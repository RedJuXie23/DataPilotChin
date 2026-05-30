@echo off
chcp 65001 >nul
echo ============================================
echo   DataPilot - AI Data Analyst
echo ============================================
echo.

:: Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

:: Check if .env exists, if not, create from template but DON'T exit
if not exist ".env" (
    echo [INFO] No .env file found. Creating from template...
    copy .env.template .env
    echo [INFO] You can edit .env file to set default API keys, or set them in the web UI.
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
echo [INFO] You can set API keys in the web UI after startup.
python app.py
