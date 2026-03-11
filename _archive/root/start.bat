@echo off
title STRAT_OS
echo.
echo ============================================================
echo    STRAT_OS - Strategic Intelligence Operating System
echo ============================================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Start Ollama if not running
echo Checking Ollama...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if errorlevel 1 (
    echo Starting Ollama service...
    start /min "" ollama serve
    timeout /t 3 >nul
)

REM Start STRAT_OS
echo Starting STRAT_OS...
echo.
echo Press Ctrl+C to stop.
echo.

cd backend
python main.py --serve

pause
