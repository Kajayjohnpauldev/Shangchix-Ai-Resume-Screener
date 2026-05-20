@echo off
REM Starts FastAPI (:8000) in a new window and Streamlit (:8501) in this window.
setlocal

cd /d "%~dp0"

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

echo [run] starting FastAPI on :8000 in a new window...
start "resume-screener-backend" cmd /k ""%PY%" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

REM Give uvicorn a moment to bind.
timeout /t 3 /nobreak >nul

echo [run] starting Streamlit on :8501...
"%PY%" -m streamlit run frontend/app.py --server.port 8501

endlocal
