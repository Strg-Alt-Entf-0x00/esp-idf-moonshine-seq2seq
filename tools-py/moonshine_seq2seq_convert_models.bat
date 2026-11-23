@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM Moonshine Seq2Seq ESP32 - Smart Model Converter
REM 
REM Completely isolated Python environment - no system Python conflicts!
REM Works by:
REM   1. Creating isolated venv with best Python version
REM   2. Using venv\Scripts\python.exe directly (no activate needed!)
REM   3. Installing clean dependencies only for this project
REM   4. Auto-detecting GPU and using CUDA if available
REM 
REM Content was rephrased for compliance with licensing restrictions.
REM Source: https://stackoverflow.com/questions/47425520
REM ============================================================================

echo ================================================================================
echo MOONSHINE ESP32 - SMART MODEL CONVERTER
echo ================================================================================
echo.

cd /d "%~dp0"

set "VENV_DIR=moonshine_seq2seq_models_converter\.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

REM ============================================================================
REM STEP 1: Find best system Python (only for creating venv)
REM ============================================================================

set "SYSTEM_PYTHON="
set "CAN_USE_GPU=0"

REM Try Python 3.11 (best - has GPU support)
py -3.11 --version >nul 2>&1
if %errorlevel%==0 (
    set "SYSTEM_PYTHON=py -3.11"
    set "CAN_USE_GPU=1"
    goto :venv_check
)

REM Try Python 3.12
py -3.12 --version >nul 2>&1
if %errorlevel%==0 (
    set "SYSTEM_PYTHON=py -3.12"
    set "CAN_USE_GPU=1"
    goto :venv_check
)

REM Try Python 3.10
py -3.10 --version >nul 2>&1
if %errorlevel%==0 (
    set "SYSTEM_PYTHON=py -3.10"
    set "CAN_USE_GPU=1"
    goto :venv_check
)

REM Fallback to default Python (likely 3.13 - CPU only)
python --version >nul 2>&1
if %errorlevel%==0 (
    set "SYSTEM_PYTHON=python"
    goto :venv_check
)

echo [ERROR] No Python found! Please install Python 3.10, 3.11, or 3.12
echo Download: https://www.python.org/downloads/
pause
exit /b 1

:venv_check

REM ============================================================================
REM STEP 2: Create isolated venv (one-time setup)
REM ============================================================================

if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" --version >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] Existing Python environment is broken or points to a missing interpreter.
        echo [SETUP] Recreating the isolated Python environment...
        rmdir /s /q "%VENV_DIR%"
        if exist "%VENV_DIR%" (
            echo [ERROR] Could not remove the broken environment: %VENV_DIR%
            pause
            exit /b 1
        )
    )
)

if not exist "%VENV_PYTHON%" (
    echo [SETUP] Creating isolated Python environment...
    echo This is a one-time setup and takes ~3 minutes.
    echo.
    
    if "%CAN_USE_GPU%"=="1" (
        echo Creating venv with GPU support...
    ) else (
        echo Creating venv with CPU support...
    )
    
    %SYSTEM_PYTHON% -m venv "%VENV_DIR%"
    
    if not exist "%VENV_PYTHON%" (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    
    "%VENV_PYTHON%" --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] The new Python environment cannot start.
        pause
        exit /b 1
    )

    echo Installing pip...
    "%VENV_PYTHON%" -m pip install --quiet --upgrade pip
    
    if "%CAN_USE_GPU%"=="1" (
        echo Installing PyTorch with CUDA support...
        "%VENV_PYTHON%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        if !errorlevel! neq 0 (
            echo [WARNING] CUDA installation failed, using CPU version...
            "%VENV_PYTHON%" -m pip install torch torchvision torchaudio
        )
    ) else (
        echo Installing PyTorch...
        "%VENV_PYTHON%" -m pip install torch torchvision torchaudio
    )
    
    echo Installing transformers and numpy...
    "%VENV_PYTHON%" -m pip install "transformers<5" numpy
    
    echo.
    echo [OK] Isolated environment ready!
    echo.
)

REM ============================================================================
REM STEP 3: Detect GPU in isolated environment
REM ============================================================================

echo [1/3] Using isolated Python environment...
echo.

echo [2/3] Detecting hardware...

set "USE_GPU=0"
set "GPU_NAME="

REM Check NVIDIA GPU with nvidia-smi
nvidia-smi --query-gpu=name --format=csv,noheader >nul 2>&1
if %errorlevel%==0 (
    for /f "delims=" %%g in ('nvidia-smi --query-gpu=name --format=csv^,noheader 2^>nul') do set "GPU_NAME=%%g"
)

REM Check if venv Python can use CUDA
if not "%GPU_NAME%"=="" (
    "%VENV_PYTHON%" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
    if !errorlevel!==0 (
        echo [OK] GPU detected: !GPU_NAME!
        echo [OK] Using CUDA acceleration ^(3-4x faster^)
        set "USE_GPU=1"
    ) else (
        echo [INFO] GPU found but CUDA not available in venv
        echo [INFO] Using CPU mode
    )
) else (
    echo [INFO] No GPU detected - using optimized CPU
)

REM ============================================================================
REM STEP 4: Convert models
REM ============================================================================

echo.
echo [3/3] Converting models...
echo.

set "MODEL_DIR=..\models\py"
set "OUTPUT_DIR=..\models\espm"
set "CONVERTER=moonshine_seq2seq_models_converter\export_moonshine_espm.py"

REM Check models directory
if not exist "%MODEL_DIR%" (
    echo [ERROR] Model directory not found: %MODEL_DIR%
    echo.
    echo Please download models first:
    echo   cd ..
    echo   python moonshine_seq2seq_models_downloader\moonshine_seq2seq_download_models_py.py --all
    echo.
    pause
    exit /b 1
)

REM Count models
set MODEL_COUNT=0
for /d %%M in ("%MODEL_DIR%\moonshine-seq2seq-*") do (
    set /a MODEL_COUNT+=1
)

if %MODEL_COUNT%==0 (
    echo [ERROR] No models found in %MODEL_DIR%
    echo.
    echo Download models first:
    echo   cd ..
    echo   python moonshine_seq2seq_models_downloader\moonshine_seq2seq_download_models_py.py --all
    echo.
    pause
    exit /b 1
)

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo Found %MODEL_COUNT% model^(s^) to convert
echo.
echo ================================================================================

REM Convert complete models directly to the supported native ESPM quantizations.
set "QUANTS=int8 int8-per-channel"

for %%Q in (%QUANTS%) do (
    if "%%Q"=="int8" (
        echo [1/2] Generating INT8 ESPM models...
    ) else if "%%Q"=="int8-per-channel" (
        echo [2/2] Generating INT8-per-channel ESPM models...
    )
    
    for /d %%M in ("%MODEL_DIR%\moonshine-seq2seq-*") do (
        set "model_name=%%~nxM"
        set "model_name=!model_name:moonshine-seq2seq-py-=!"
        echo   Converting !model_name!...
        
        "%VENV_PYTHON%" "%CONVERTER%" ^
            --model-dir "%%~fM" ^
            --output "%OUTPUT_DIR%" ^
            --quantize %%Q
        
        if !errorlevel! neq 0 (
            echo [ERROR] Failed to convert !model_name! ^(%%Q^)
            pause
            exit /b 1
        )
    )
    echo.
)

REM ============================================================================
REM DONE
REM ============================================================================

echo ================================================================================
echo SUCCESS!
echo ================================================================================
echo.
echo Native ESPM models converted to: %OUTPUT_DIR%\
echo.
echo Quantizations:
echo   - INT8
echo   - INT8-per-channel
echo Tokenizer and native tensor index are embedded in each .espm file.
echo.
if "%USE_GPU%"=="1" (
    echo Hardware used: GPU ^(CUDA^) - Fast conversion!
) else (
    echo Hardware used: CPU ^(optimized^)
)
echo.
echo Models are ready to use with ESP32-P4!
echo.

pause
