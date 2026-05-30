@echo off
chcp 65001 >nul
echo ============================================
echo   AAReproduce - AI Data Analyst
echo ============================================
echo.

:: Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

:: Check if .env exists
if not exist ".env" (
    echo [INFO] No .env file found. Creating from template...
    copy .env.template .env
    echo [INFO] Please edit .env and add your API key, then run again.
    pause
    exit /b 1
)

:: Check if dependencies installed
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo [INFO] Starting backend on http://localhost:8001
python app.py
