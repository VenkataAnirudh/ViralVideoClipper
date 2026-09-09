@echo off
title Clip Branding Studio
cd /d "%~dp0"

REM Use the project venv if present, else fall back to system python.
set "PY=%~dp0..\venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo Starting Clip Branding Studio...
echo (a browser tab will open automatically)
echo.
"%PY%" brand_server.py

echo.
echo Server stopped.
pause
