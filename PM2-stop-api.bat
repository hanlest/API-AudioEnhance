@echo off

setlocal EnableExtensions

cd /d "%~dp0"

title API-AudioEnhance API - Stop

echo.
echo  ========================================
echo   API-AudioEnhance - Detener API
echo  ========================================
echo.

where pm2 >nul 2>&1
if errorlevel 1 goto NO_PM2

call pm2 describe API-AudioEnhance >nul 2>&1
if errorlevel 1 goto NOT_FOUND

echo Deteniendo API-AudioEnhance...
call pm2 stop API-AudioEnhance
if errorlevel 1 goto STOP_FAIL

echo Eliminando API-AudioEnhance de PM2...
call pm2 delete API-AudioEnhance
if errorlevel 1 goto DELETE_FAIL

echo API-AudioEnhance detenido y eliminado de PM2.
endlocal
exit /b 0

:NOT_FOUND
echo API-AudioEnhance no esta registrado en PM2.
endlocal
exit /b 0

:NO_PM2
echo [ERROR] PM2 no esta instalado o no esta en el PATH.
endlocal
exit /b 1

:STOP_FAIL
echo [ERROR] PM2 no pudo detener el servicio.
endlocal
exit /b 1

:DELETE_FAIL
echo [ERROR] PM2 no pudo eliminar el servicio.
endlocal
exit /b 1
