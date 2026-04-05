import os
from dotenv import load_dotenv
import pathlib

load_dotenv()

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key":  os.getenv("GROQ_API_KEY", ""),
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key":  os.getenv("OPENROUTER_API_KEY", ""),
    },
}

MODEL_ROLES = {
    "reason": {
        "provider": "groq",
        "model":    "qwen/qwen3-32b",
    },
    "code": {
        "provider": "groq",
        "model":    "llama-3.3-70b-versatile",
    },
    "fallback": {
        "provider": "openrouter",
        "model":    "google/gemma-4-31b-it",
    },
}

MAX_RETRIES     = 3
RETRY_DELAY_SEC = 2
MAX_TOKENS      = 2048
TEMPERATURE     = 0.1

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
CACHE_DIR  = ".cache"


def safe_path(*parts) -> str:
    """Always returns forward-slash path, safe on both Windows and Linux."""
    return pathlib.Path(*parts).as_posix()

'''
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model":    "qwen/qwen3-32b",   # best free reasoning model on groq
        "api_key":  os.getenv("GROQ_API_KEY", ""),
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model":    "google/gemma-4-31b-it",
        "api_key":  os.getenv("OPENROUTER_API_KEY", ""),
    }
}
'''
