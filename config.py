import os 

DATA_DIR = "/data"
USER_EMAIL = "24f2001423@ds.study.iitm.ac.in"

# AI Proxy
AI_URL = "http://aiproxy.sanand.workers.dev/openai/v1"
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
AI_MODEL = "gpt-4o-mini"
AI_EMBEDDINGS_MODEL = "text-embedding-3-small"# for debugging use LLM token

if not AIPROXY_TOKEN:
    raise KeyError("AIPROXY_TOKEN environment variable is not present in the docker environment variables")

