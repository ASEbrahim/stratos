@echo off
title STRAT_OS Setup
echo.
echo ============================================================
echo    STRAT_OS - Strategic Intelligence Operating System
echo    First-Time Setup
echo ============================================================
echo.

REM Check Python
echo [1/4] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.11+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
echo [OK] Python found.
echo.

REM Check Ollama
echo [2/4] Checking Ollama installation...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama is not installed.
    echo.
    echo Ollama is required for AI-powered scoring.
    echo Please install it from: https://ollama.ai/download
    echo.
    echo After installing Ollama, run this setup again.
    echo.
    set /p continue="Continue without Ollama? (y/n): "
    if /i not "%continue%"=="y" (
        exit /b 1
    )
    echo [SKIP] Continuing without Ollama (will use fallback scoring)
) else (
    echo [OK] Ollama found.
    
    REM Pull model
    echo.
    echo [3/4] Pulling AI model (llama3.2)...
    echo This may take 5-10 minutes on first run...
    ollama pull llama3.2
    echo [OK] Model ready.
)
echo.

REM Install Python packages
echo [4/4] Installing Python dependencies...
cd /d "%~dp0"
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo [OK] Dependencies installed.
echo.

REM Create output directory
if not exist "backend\output" mkdir "backend\output"

echo ============================================================
echo    Setup Complete!
echo ============================================================
echo.
echo To start STRAT_OS, run: start.bat
echo.
pause
