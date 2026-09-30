@echo off

setlocal EnableExtensions

cd /d "%~dp0"

title API-AudioEnhance API

echo.
echo  ========================================
echo   API-AudioEnhance - API (FastAPI)
echo  ========================================
echo.

set "PYTHONW=%CD%\.venv\Scripts\pythonw.exe"
set "PYTHON=%CD%\.venv\Scripts\python.exe"
set "PORT=8000"
if defined AUDIOENHANCE_API_PORT set "PORT=%AUDIOENHANCE_API_PORT%"
if not defined AUDIOENHANCE_API_PORT if defined RESEMBLE_API_PORT set "PORT=%RESEMBLE_API_PORT%"
if not defined BASE_PATH set "BASE_PATH=/docs"

where python >nul 2>&1
if errorlevel 1 goto NO_PYTHON

if not exist "%PYTHON%" (
    echo [ERROR] No hay .venv. Crea el entorno e instala dependencias:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements-inference.txt
    pause
    exit /b 1
)

if exist "%PYTHONW%" (
    set "INTERPRETER=%PYTHONW%"
) else (
    set "INTERPRETER=%PYTHON%"
)

"%PYTHON%" -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo Instalando FastAPI y uvicorn ...
    "%PYTHON%" -m pip install -r requirements-inference.txt
    if errorlevel 1 goto PIP_FAIL
)

where pm2 >nul 2>&1
if errorlevel 1 goto NO_PM2

call pm2 describe API-AudioEnhance >nul 2>&1
if not errorlevel 1 (
    echo Recreando API-AudioEnhance en puerto %PORT%...
    call pm2 delete API-AudioEnhance
)

echo Liberando puerto %PORT% si quedo un proceso huerfano...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr "0.0.0.0:%PORT%" ^| findstr LISTENING') do (
    echo Cerrando PID %%a que ocupa el puerto %PORT%...
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr "127.0.0.1:%PORT%" ^| findstr LISTENING') do (
    echo Cerrando PID %%a que ocupa el puerto %PORT%...
    taskkill /F /PID %%a >nul 2>&1
)

echo Iniciando API-AudioEnhance en PM2...
set PYTHONUNBUFFERED=1
set AUDIOENHANCE_API_PORT=%PORT%
set BASE_PATH=%BASE_PATH%
call pm2 start api.py --name API-AudioEnhance --interpreter "%INTERPRETER%" --cwd "%CD%"
if errorlevel 1 goto PM2_FAIL
goto DONE

:NO_PYTHON
echo [ERROR] Python no esta en el PATH.
pause
exit /b 1

:PIP_FAIL
echo [ERROR] pip install -r requirements-inference.txt fallo.
pause
exit /b 1

:NO_PM2
echo [ERROR] PM2 no esta instalado o no esta en el PATH.
echo Instala con: npm install -g pm2
pause
exit /b 1

:PM2_FAIL
echo [ERROR] PM2 no pudo iniciar el servicio.
echo Revisa logs: pm2 logs API-AudioEnhance --lines 50
pause
exit /b 1

:DONE
echo.
call pm2 status API-AudioEnhance
echo.
echo Servicio listo en PM2 como API-AudioEnhance
if "%BASE_PATH%"=="" (
    echo Swagger: http://127.0.0.1:%PORT%/docs
) else (
    echo Swagger: http://127.0.0.1:%PORT%%BASE_PATH%/
)
echo Puerto:  set AUDIOENHANCE_API_PORT=%PORT%
echo Prefijo: set BASE_PATH=%BASE_PATH%
endlocal
exit /b 0
