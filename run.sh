#!/usr/bin/env bash
# Starts FastAPI (port 8000) and Streamlit (port 8501).
# Works on macOS, Linux, and Windows under Git Bash.
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Prefer the venv's interpreter if present.
if [[ -x ".venv/Scripts/python.exe" ]]; then
  PY=".venv/Scripts/python.exe"
elif [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python"
fi

echo "[run] starting FastAPI on :8000"
"$PY" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

cleanup() {
  echo "[run] stopping backend pid $BACKEND_PID"
  kill "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Give uvicorn a moment to bind the port before Streamlit health-checks it.
sleep 2

echo "[run] starting Streamlit on :8501"
"$PY" -m streamlit run frontend/app.py --server.port 8501
