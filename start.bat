@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title nanoStudio - Multi-Modal AI Studio

echo ================================================================
echo               nanoStudio Launcher (Windows)
echo      Multi-Modal Deep Learning Studio - LLM / Vision / Diffusion
echo ================================================================
echo.

REM 1. Check Python
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in system PATH!
    echo Please install Python 3.9+ and ensure "Add Python to PATH" is checked.
    echo.
    pause
    exit /b 1
)

REM 2. Activate virtual environment if present
if exist venv\Scripts\activate.bat (
    echo [Environment] Activating venv...
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    echo [Environment] Activating .venv...
    call .venv\Scripts\activate.bat
)

REM 3. Launch server
echo [Server] Starting nanoStudio Web Console...
echo [Ready] Opening browser: http://127.0.0.1:8000
echo         (Press Ctrl + C in this window to stop the server)
echo ================================================================
echo.

python server.py

if errorlevel 1 (
    echo.
    echo [Notice] Server exited or encountered an error. Press any key to exit...
    pause >nul
)
