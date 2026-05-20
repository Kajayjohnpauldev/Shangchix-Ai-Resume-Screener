"""Loads environment-driven config once at import time."""
import os
from dotenv import load_dotenv

load_dotenv()

# litellm selects the provider from the model-string prefix, so a single
# env var re-points the whole app. Examples:
#   gemini/gemini-1.5-flash      Google (free tier, default)
#   gpt-4o-mini                  OpenAI
#   groq/llama-3.1-8b-instant    Groq
#   claude-3-5-haiku-20241022    Anthropic
LLM_MODEL = os.getenv("LLM_MODEL", "gemini/gemini-1.5-flash")

LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "15"))
TOP_K_EXPLAIN = int(os.getenv("TOP_K_EXPLAIN", "5"))
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
