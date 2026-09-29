@echo off
setlocal EnableExtensions
cd /d "%~dp0"

title API-AudioEnhance - PyTorch GPU

if not exist ".venv\Scripts\python.exe" (
    echo Ejecuta start.bat primero para crear .venv
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [ERROR] nvidia-smi no encontrado. Instala el driver NVIDIA.
    pause
    exit /b 1
)

echo GPU detectada:
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo.

rem cu126: drivers recientes. Si falla, prueba cu124 en la URL.
set "CUDA_WHL=cu126"
if not "%~1"=="" set "CUDA_WHL=%~1"

echo Instalando torch+torchaudio (%CUDA_WHL%) ...
python -m pip install --upgrade pip
pip uninstall -y torch torchaudio torchvision 2>nul
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/%CUDA_WHL%

echo.
python -c "import torch; ok=torch.cuda.is_available(); print('CUDA disponible:', ok); print('Dispositivo:', torch.cuda.get_device_name(0) if ok else 'N/A'); raise SystemExit(0 if ok else 1)"
if errorlevel 1 (
    echo.
    echo [ERROR] PyTorch sigue sin ver la GPU. Prueba: install-gpu.bat cu124
    pause
    exit /b 1
)

echo.
echo Listo. Reinicia start.bat para usar la GPU.
pause
endlocal
