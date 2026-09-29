@echo off

setlocal EnableExtensions

cd /d "%~dp0"



title API-AudioEnhance



echo.

echo  ========================================

echo   API-AudioEnhance - Demo Gradio

echo  ========================================

echo.



where python >nul 2>&1

if errorlevel 1 (

    echo [ERROR] Python no esta en el PATH. Instala Python 3.10+ desde https://www.python.org/

    pause

    exit /b 1

)



python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1

if errorlevel 1 (

    echo [ERROR] Se requiere Python 3.10 o superior.

    pause

    exit /b 1

)



if not exist ".venv\Scripts\python.exe" (

    echo [1/5] Creando entorno virtual .venv ...

    python -m venv .venv

    if errorlevel 1 (

        echo [ERROR] No se pudo crear el entorno virtual.

        pause

        exit /b 1

    )

) else (

    echo [1/5] Entorno virtual .venv encontrado.

)



call ".venv\Scripts\activate.bat"

if errorlevel 1 (

    echo [ERROR] No se pudo activar .venv

    pause

    exit /b 1

)



echo [2/5] Actualizando pip ...

python -m pip install --upgrade pip



if not defined PYTORCH_CUDA set "PYTORCH_CUDA=cu126"



echo [3/5] Instalando dependencias (inferencia) ...

if not exist ".deps_installed" (

    where nvidia-smi >nul 2>&1

    if errorlevel 1 (

        echo       Sin NVIDIA detectada: PyTorch CPU.

        pip install torch torchaudio torchvision

    ) else (

        echo       NVIDIA detectada: PyTorch %PYTORCH_CUDA% ...

        pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/%PYTORCH_CUDA%

    )

    pip install -r requirements-inference.txt

    pip install -e . --no-deps

    if errorlevel 1 (

        echo [ERROR] Fallo la instalacion de dependencias.

        pause

        exit /b 1

    )

    echo ok> ".deps_installed"

) else (

    echo       Dependencias base OK ^(.deps_installed^). Usa install-gpu.bat si necesitas CUDA.

)



echo [4/5] Comprobando dispositivo de inferencia ...

python -c "from resemble_enhance.device import describe_device, get_inference_device; d=get_inference_device(); print('  ->', describe_device(d)); import sys; sys.exit(0 if d!='cpu' or __import__('os').environ.get('RESEMBLE_ALLOW_CPU') else 0)"

where nvidia-smi >nul 2>&1

if not errorlevel 1 (

    python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1

    if errorlevel 1 (

        echo.

        echo  [AVISO] Tienes GPU NVIDIA pero PyTorch esta en modo CPU.

        echo          Ejecuta install-gpu.bat y vuelve a lanzar start.bat

        echo.

    )

)



echo [5/5] Iniciando demo web Gradio ...

echo.

echo  Variable opcional: set AUDIOENHANCE_DEVICE=cuda ^| cpu

echo  La primera inferencia puede descargar pesos desde Hugging Face.

echo  Pulsa Ctrl+C para detener el servidor.

echo.



set PYTHONUNBUFFERED=1

python app.py

set EXIT_CODE=%ERRORLEVEL%



if not "%EXIT_CODE%"=="0" (

    echo.

    echo [ERROR] La aplicacion termino con codigo %EXIT_CODE%.

    pause

)



endlocal

exit /b %EXIT_CODE%

