"""Loads environment-driven config once at import time."""
import os
from dotenv import load_dotenv

load_dotenv()

# Which LLM provider LangChain should dispatch to.
# Supported: "openai", "gemini".
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# Per-provider model name. Defaults chosen for cost + quality.
LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gpt-4o-mini" if LLM_PROVIDER == "openai" else "gemini-1.5-flash",
)

LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "15"))
TOP_K_EXPLAIN = int(os.getenv("TOP_K_EXPLAIN", "5"))
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
