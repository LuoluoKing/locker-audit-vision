@echo off
rem ============================================================
rem  Tianyan Sentinel - one-click launcher (ASCII only)
rem  %~dp0 = directory containing this file, so the project
rem  path is resolved at runtime and never hardcoded here.
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv not found. Run this first:
  echo   python -m venv .venv
  echo   .venv\Scripts\python.exe -m pip install -r code\requirements.txt
  pause
  exit /b 1
)

echo Starting Tianyan Sentinel on http://127.0.0.1:5000 ...
echo Monitoring console : http://127.0.0.1:5000/
echo Admin backend      : http://127.0.0.1:5000/admin
echo.
echo Keep this window OPEN. Press Ctrl+C to stop the service.
echo.

".venv\Scripts\python.exe" "code\app.py" --backend auto --port 5000

echo.
echo Service stopped.
pause
