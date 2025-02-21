import base64
import httpx
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import shutil
import csv
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Optional
import easyocr
import numpy as np
from dateutil import parser
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from config import *
from tasks import *

# data is initialized through Dockerfile
app = FastAPI()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)




# POST `/run?task=<task description>` - Executes a task described in plain English.
# The agent interprets the instruction, performs the necessary internal steps (which may involve leveraging an LLM), and generates the final output.
# - On success, returns HTTP 200 OK
# - If the task contains an error, returns HTTP 400 Bad Request
# - If an internal agent error occurs, returns HTTP 500 Internal Server Error
# - The response body may include relevant details in each case.
@app.post("/run")
def run_task(task: str) -> Response:
    try:
        if not task:
            raise ValueError("Task description is required")

        tool = get_task_tool(task, tasks_list)
        return execute_tool_calls(tool)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def execute_tool_calls(tool: Dict[str, Any]):
    if tool and "tool_calls" in tool:
        for tool_call in tool["tool_calls"]:
            function_name = tool_call["function"].get("name")
            function_args = tool_call["function"].get("arguments")

            # Ensure the function name is valid and callable
            if function_name in globals() and callable(globals()[function_name]):
                function_chosen = globals()[function_name]
                function_args = parse_function_args(function_args)

                if isinstance(function_args, dict):
                    return function_chosen(**function_args)

    raise NotImplementedError("Unknown task")


def parse_function_args(function_args: Optional[Any]):
    if function_args is not None:
        if isinstance(function_args, str):
            function_args = json.loads(function_args)

        elif not isinstance(function_args, dict):
            function_args = {"args": function_args}
    else:
        function_args = {}

    return function_args


# GET `/read?path=<file path>` - Retrieves the content of the specified file.
# This endpoint is essential for verifying the exact output.
# - On success, returns an HTTP 200 OK response with the file content as plain text.
# - If the file is not found, returns an HTTP 404 Not Found response with an empty body.
@app.get("/read")
def path_check(path):
    abs1 = os.path.abspath(path).lower()
    abs2 = os.path.abspath(DATA_DIR).lower()

    return os.path.abspath(abs1).startswith(abs2)


def read_file(path: str) -> Response:
    try:
        if not path:
            raise ValueError("File path is required")

        # dont allow path pout side DATA_DIR
        if not path_check(path):
            raise PermissionError("Acces denied")

        if not os.path.exists(path):
            raise FileNotFoundError("File not found")

        with open(path, "r") as f:
            content = f.read()

        return Response(content=content, media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Task implementations

tasks_list = json.load(open("task_data.json", "r", encoding="utf-8"))["tasks_array"]



# B3
# Fetch data from an API and save it to a file
def fetch_data(url: str, destination: str):
    if not url:
        raise ValueError("API url is required")

    if not destination:
        raise ValueError("Destination directory is required")

    if not path_check(destination):
        raise PermissionError(f"path not in {DATA_DIR}")

    response = httpx.get(url, verify=False)
    response.raise_for_status()

    with open(destination, "wb") as f:
        f.write(response.content)

    return {
        "message": "API Fetched",
        "source": url,
        "destination": destination,
        "status": "success",
    }


# B4
# Clone a get rpository to a directory and commit the changes
def clone_and_commit(url: str, destination: str):
    if not url:
        raise ValueError("Git repository url is required")

    if not destination:
        raise ValueError("Destination directory is required")

    if not path_check(destination):
        raise PermissionError(f"path not in {DATA_DIR}")

    if os.path.exists(destination):
        shutil.rmtree(destination)

    os.makedirs(destination, exist_ok=True)
    os.chmod(destination, 0o777)
    os.chdir(destination)

    result = subprocess.run(
        ["git", "clone", url, "."],
        check=True,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr)

    # commit back the changes

    with open("commit_time.txt", "w") as f:
        f.write(datetime.now().isoformat())

    result = subprocess.run(
        ["git", "add", "."], check=True, capture_output=True, text=True
    )

    if result.returncode != 0:
        logging.error(result.stderr)

    subprocess.run(
        ["git", "commit", "--no-verify", "-m", f"Changes from {USER_EMAIL}"],
        check=True,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logging.error(result.stderr)

    return {
        "message": "Git repository cloned",
        "source": url,
        "destination": destination,
        "status": "success",
    }


# B5
# Run a SQL query on a SQLite
def run_sql_query(query: str, source: str, destination: str):
    if not query:
        raise ValueError("QUERY is required")

    if not source:
        raise ValueError("DB file is required")

    db_path = source

    if not os.path.exists(db_path):
        raise FileNotFoundError("DB not found")

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()

        output_path = destination or file_rename(db_path, "-query-results.csv")

        with open(output_path, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                [description[0] for description in cursor.description]
            )  # write headers
            csv_writer.writerows(result)

    return {
        "message": "SQL query executed",
        "source": db_path,
        "result": output_path,
        "status": "success",
    }
