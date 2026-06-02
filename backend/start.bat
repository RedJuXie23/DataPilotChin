@echo off
chcp 65001 >nul
setlocal EnableExtensions

set "VENV_DIR=venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

echo ============================================
echo   DataPilot - AI Data Analyst
echo ============================================
echo.

:: Check Python availability in PATH
where python >nul 2>nul
if errorlevel 1 (
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

call :EnsureVenv
if errorlevel 1 (
    echo [ERROR] Virtual environment setup failed.
    pause
    exit /b 1
)

powershell -NoProfile -Command "if (Get-NetTCPConnection -State Listen -LocalPort 8001 -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
if not errorlevel 1 (
    echo [INFO] Backend is already listening on http://localhost:8001
    echo [INFO] Stop the existing backend process before restarting to load new code.
    exit /b 0
)

echo [INFO] Starting backend on http://localhost:8001
"%VENV_PY%" app.py
exit /b %errorlevel%


:EnsureVenv
if not exist "%VENV_PY%" (
    echo [INFO] Virtual environment not found. Creating...
    call :RebuildVenv
    exit /b %errorlevel%
)

:: Detect broken venv interpreter (e.g. points to removed base Python)
"%VENV_PY%" -c "import sys; print(sys.executable)" >nul 2>nul
if errorlevel 1 (
    echo [WARN] Existing virtual environment is invalid. Rebuilding...
    call :RebuildVenv
    exit /b %errorlevel%
)

:: Quick dependency health check
"%VENV_PY%" -c "import fastapi, uvicorn, dspy" >nul 2>nul
if errorlevel 1 (
    echo [WARN] Required packages are missing. Reinstalling dependencies...
    "%VENV_PY%" -m pip install -r requirements.txt
    if errorlevel 1 exit /b 1
)
exit /b 0


:RebuildVenv
if exist "%VENV_DIR%" (
    rmdir /s /q "%VENV_DIR%"
)
python -m venv "%VENV_DIR%"
if errorlevel 1 exit /b 1
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
exit /b 0
