# LLM-based Automation Agent

## Overview
This project implements an automation agent powered by Large Language Models (LLM) to assist with various automation tasks. The agent is containerized using Docker for easy deployment and scalability.

## Prerequisites
- Docker installed on your system
- Valid AIPROXY_TOKEN (required for API authentication)
- Internet connection for pulling the Docker image

## Quick Start

### Environment Setup
1. Make sure you have your AIPROXY_TOKEN ready. 
2. Set your AIPROXY_TOKEN as an environment variable:
```bash
export AIPROXY_TOKEN=your_token_here
```

### Running the Agent
Execute the following Docker command to start the automation agent:
```bash
docker run -e AIPROXY_TOKEN=$AIPROXY_TOKEN -p 8000:8000 anantmulchandani/llm_fastapi_automation
```

This command will:
- Pull the image if not already present locally
- Map port 8000 from the container to your host machine
- Pass your AIPROXY_TOKEN to the container
- Start the automation agent service

### Accessing the Service
Once the container is running, you can access the service at:
```
http://localhost:8000
```

## API Documentation

The service provides two main endpoints:

### 1. Task Execution API
```
POST /run?task=<task description>
```
Executes a plain-English task. The agent parses the instruction, executes required steps (including LLM assistance), and produces the final output.

#### Response Codes:
- `200 OK`: Task executed successfully
- `400 Bad Request`: Error in the task description or parameters
- `500 Internal Server Error`: Agent execution error

The response body may contain additional information in all cases.

#### Example Tasks (please parse the task beforehand):
1. Finding similar comments using embeddings:
```
/run?task=/data/comments.txt contains a list of comments, one per line. Using embeddings, find the most similar pair of comments and write them to /data/comments-similar.txt, one per line
```

2. Multilingual support (Hindi) - Counting Sundays:
```
/run?task=/data/contents.log में कितने रविवार हैं? गिनो और /data/contents.dates में लिखो
```

### 2. File Reading API
```
GET /read?path=<file path>
```
Returns the content of the specified file for verification purposes.

#### Response Codes:
- `200 OK`: File read successfully (returns content as plain text)
- `404 Not Found`: File does not exist (empty body)

## Configuration
The service uses the following default configurations:
- Port: 8000
- API Version: v1
- Container Image: anantmulchandani/llm_fastapi_automation


## License
MIT License

This project was done as per https://tds.s-anand.net/#/project-1 