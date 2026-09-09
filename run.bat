@echo off
setlocal
title Video Clipper
cd /d "%~dp0"

echo.
echo  =======================================
echo           Video Clipper Launcher
echo  =======================================
echo.

set "PY=venv\Scripts\python.exe"

if not exist "%PY%" goto missing_setup

"%PY%" -c "import sys" >nul 2>&1
if errorlevel 1 goto broken_venv

echo [*] Checking app dependencies...
"%PY%" -c "import flask, cv2, dotenv, numpy" >nul 2>&1
if errorlevel 1 goto missing_deps

echo [*] Cleaning up old server instances...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":5000" ^| find "LISTENING"') do taskkill /f /pid %%a >nul 2>&1

echo [*] Starting Video Clipper server...
echo [*] Browser will open automatically in 3 seconds.
echo [*] Press Ctrl+C to stop the server.
echo.

start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:5000"
"%PY%" app.py

echo.
echo [!] Server stopped or exited with an error. See the output above.
pause
exit /b

:missing_setup
echo [!] venv was not found.
echo [!] Run install.bat once, then run this file again.
pause
exit /b 1

:broken_venv
echo [!] venv exists, but its Python is broken.
echo [!] Run install.bat once to recreate it, then run this file again.
pause
exit /b 1

:missing_deps
echo [!] Required app packages are missing from the venv.
echo [!] Run install.bat once, then run this file again.
pause
exit /b 1
