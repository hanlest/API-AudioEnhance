@echo off

setlocal EnableExtensions

cd /d "%~dp0"

title API-AudioEnhance API

echo.
echo  ========================================
echo   API-AudioEnhance - API (FastAPI)
echo  ========================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no esta en el PATH.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] No hay .venv. Ejecuta start.bat una vez para crear el entorno.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

python -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo Instalando FastAPI y uvicorn ...
    pip install -r requirements-inference.txt
)

echo Swagger: http://127.0.0.1:8000/docs
echo Variable opcional: set AUDIOENHANCE_API_PORT=8000
echo.

set PYTHONUNBUFFERED=1
python api.py

set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" pause
endlocal
exit /b %EXIT_CODE%
