@echo off
setlocal
title Video Clipper Setup
cd /d "%~dp0"

echo.
echo  =======================================
echo        Video Clipper One-Time Setup
echo        (GPU Accelerated Edition)
echo  =======================================
echo.

set "PY=venv\Scripts\python.exe"

:: ---------------------------------------------------------------
:: GUARD: If a previous install completed successfully, skip it.
:: Delete .install_complete manually to force a full reinstall.
:: ---------------------------------------------------------------
if exist ".install_complete" (
    echo [*] Setup already completed. Skipping full install.
    echo [*] Delete ".install_complete" in this folder to force a reinstall.
    echo.
    goto verify_and_exit
)

:: ---------------------------------------------------------------
:: VENV: Create or heal the virtual environment
:: ---------------------------------------------------------------
if exist "%PY%" (
    "%PY%" -c "import sys" >nul 2>&1
    if errorlevel 1 (
        echo [!] Existing venv is broken. Recreating it now...
        rmdir /s /q venv
    ) else (
        echo [*] Existing venv looks usable.
    )
)

if not exist "%PY%" (
    echo [*] Creating virtual environment...
    py -3.11 -m venv venv >nul 2>&1
    if errorlevel 1 py -3 -m venv venv >nul 2>&1
    if errorlevel 1 python -m venv venv
    if errorlevel 1 (
        echo [!] Failed to create venv.
        echo [!] Install Python 3.11+ from https://www.python.org/downloads/
        pause
        exit /b 1
    )
)

:: ---------------------------------------------------------------
:: PIP TOOLING
:: ---------------------------------------------------------------
echo.
echo [*] Upgrading pip tooling...
"%PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto install_failed

:: ---------------------------------------------------------------
:: PYTORCH: Only install if not already present with CUDA support.
:: This is the biggest download (~3 GB) so we skip it when possible.
:: ---------------------------------------------------------------
echo.
echo [*] Checking PyTorch + CUDA status...
"%PY%" -c "import torch; assert torch.cuda.is_available(), 'no cuda'" >nul 2>&1
if errorlevel 1 (
    echo [*] PyTorch with CUDA not found — installing now...
    "%PY%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    if errorlevel 1 (
        echo [!] CUDA PyTorch install failed. Trying default PyTorch package...
        "%PY%" -m pip install torch torchvision torchaudio
        if errorlevel 1 goto install_failed
    )
    :: Re-check after install
    "%PY%" -c "import torch; print('[*] PyTorch', torch.__version__, '— CUDA:', torch.cuda.is_available())"
) else (
    :: Verify torch and torchvision are from the same build (prevents circular-import crash)
    "%PY%" -c "import torch, torchvision; tv=torchvision.__version__; tt=torch.__version__; assert tv.split('+')[0].startswith(tt.split('.')[0]+'.'), f'version mismatch torch={tt} torchvision={tv}'" >nul 2>&1
    if errorlevel 1 (
        echo [!] torch/torchvision version mismatch detected — reinstalling matched set...
        "%PY%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
        if errorlevel 1 goto install_failed
    ) else (
        "%PY%" -c "import torch; print('[*] PyTorch', torch.__version__, 'already installed — CUDA OK, skipping.')"
    )
)

:: ---------------------------------------------------------------
:: CUDA VERSION DETECTION (for CuPy and OpenCV GPU)
:: ---------------------------------------------------------------
echo.
echo [*] Detecting CUDA version for GPU dependencies...
set "CUDA_VER="
"%PY%" -c "import torch; v=torch.version.cuda; print(v.replace('.','')[:3])" > temp_cuda.txt
set /p CUDA_VER=<temp_cuda.txt
del temp_cuda.txt

if "%CUDA_VER%"=="" (
    echo [!] Could not detect CUDA version. Assuming CUDA 12.x
    set "CUDA_VER=12x"
)

echo [*] Detected CUDA version: %CUDA_VER%

:: Map CUDA version to CuPy package
if "%CUDA_VER%"=="126" set "CUPY_PKG=cupy-cuda12x>=12.0.0"
if "%CUDA_VER%"=="121" set "CUPY_PKG=cupy-cuda12x>=12.0.0"
if "%CUDA_VER%"=="120" set "CUPY_PKG=cupy-cuda12x>=12.0.0"
if "%CUDA_VER%"=="118" set "CUPY_PKG=cupy-cuda11x>=11.0.0"
if "%CUDA_VER%"=="117" set "CUPY_PKG=cupy-cuda11x>=11.0.0"
if "%CUPY_PKG%"=="" set "CUPY_PKG=cupy-cuda12x>=12.0.0"

echo [*] Will install CuPy package: %CUPY_PKG%

:: ---------------------------------------------------------------
:: CORE REQUIREMENTS: GPU-optimized stack
:: ---------------------------------------------------------------
echo.
echo [*] Checking core app dependencies...
"%PY%" -c "import flask, cv2, dotenv, numpy, faster_whisper, google, anthropic, av, scenedetect" >nul 2>&1
if errorlevel 1 (
    echo [*] One or more packages missing — running requirements install...
    "%PY%" -m pip install --prefer-binary -r requirements.txt
    if errorlevel 1 goto install_failed
) else (
    echo [*] Core packages already installed.
)

:: ---------------------------------------------------------------
:: OPENCV GPU CONTRIB (for cudacodec and CUDA modules)
:: ---------------------------------------------------------------
echo.
echo [*] Installing OpenCV with GPU support...
"%PY%" -c "import cv2; assert cv2.cuda.getCudaEnabledDeviceCount() > 0, 'no cuda'" >nul 2>&1
if errorlevel 1 (
    echo [*] OpenCV CUDA modules not detected — installing opencv-contrib-python...
    "%PY%" -m pip install --prefer-binary opencv-contrib-python>=4.9
    if errorlevel 1 (
        echo [!] opencv-contrib-python install failed. GPU decode will be unavailable.
        echo [!] Face tracking will use CPU fallback for frame decoding.
    ) else (
        echo [*] opencv-contrib-python installed successfully.
    )
) else (
    echo [*] OpenCV CUDA modules already available.
)

:: ---------------------------------------------------------------
:: CUPY (GPU array computing for batch operations)
:: ---------------------------------------------------------------
echo.
echo [*] Installing CuPy GPU acceleration...
"%PY%" -c "import cupy" >nul 2>&1
if errorlevel 1 (
    echo [*] CuPy not found — installing %CUPY_PKG%...
    "%PY%" -m pip install --prefer-binary %CUPY_PKG%
    if errorlevel 1 (
        echo [!] CuPy install failed. Will use CPU NumPy for array operations.
        echo [!] This reduces face scoring performance significantly.
    ) else (
        echo [*] CuPy installed successfully.
    )
) else (
    echo [*] CuPy already installed.
)

:: ---------------------------------------------------------------
:: INSIGHTFACE: GPU face detector (replaces MediaPipe)
:: ---------------------------------------------------------------
echo.
echo [*] Checking GPU face detector (InsightFace)...
"%PY%" -c "import insightface" >nul 2>&1
if errorlevel 1 (
    echo [*] InsightFace not found — installing GPU face detector...
    
    :: Try prebuilt wheel first
    echo [*] Trying InsightFace prebuilt Windows wheel for Python 3.11...
    "%PY%" -m pip install https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp311-cp311-win_amd64.whl
    if errorlevel 1 (
        :: Fallback to pip install
        echo [!] Prebuilt wheel failed — trying pip install...
        "%PY%" -m pip install --prefer-binary insightface>=0.7.3
        if errorlevel 1 (
            echo [!] InsightFace install completely failed.
            echo [!] Face tracking will fall back to CPU MediaPipe (slower).
            echo [!] MediaPipe will be installed as fallback...
            "%PY%" -m pip install "protobuf>=3.20.0,<4.0.0" mediapipe==0.10.9
        ) else (
            echo [*] InsightFace installed via pip.
        )
    ) else (
        echo [*] InsightFace installed from prebuilt wheel.
    )
) else (
    echo [*] InsightFace already installed.
)

:: ---------------------------------------------------------------
:: ONNX RUNTIME GPU (for InsightFace backend)
:: ---------------------------------------------------------------
echo.
echo [*] Installing ONNX Runtime GPU...
"%PY%" -c "import onnxruntime; assert 'CUDA' in onnxruntime.get_available_providers(), 'no cuda'" >nul 2>&1
if errorlevel 1 (
    echo [*] ONNX Runtime GPU not found — installing...
    "%PY%" -m pip install --prefer-binary onnxruntime-gpu>=1.16.0
    if errorlevel 1 (
        echo [!] ONNX Runtime GPU install failed. InsightFace may use CPU backend.
    ) else (
        echo [*] ONNX Runtime GPU installed.
    )
) else (
    echo [*] ONNX Runtime GPU already available.
)

:: ---------------------------------------------------------------
:: MEDIAPIPE: Only install as fallback if InsightFace failed
:: ---------------------------------------------------------------
echo.
echo [*] Checking face detection capability...
"%PY%" -c "import insightface; print('OK')" >nul 2>&1
if errorlevel 1 (
    echo [*] InsightFace not available — installing MediaPipe fallback...
    
    :: Check if mediapipe already installed
    "%PY%" -c "import mediapipe" >nul 2>&1
    if errorlevel 1 (
        echo [*] Installing protobuf and MediaPipe...
        "%PY%" -m pip install "protobuf>=3.20.0,<4.0.0" mediapipe==0.10.9
        if errorlevel 1 (
            echo [!] MediaPipe install failed. Face tracking will be disabled.
        ) else (
            echo [*] MediaPipe fallback installed.
        )
    ) else (
        echo [*] MediaPipe already present as fallback.
    )
    
    :: Verify protobuf version
    "%PY%" -c "import google.protobuf; v=google.protobuf.__version__; major=int(v.split('.')[0]); assert major < 4, f'protobuf {v} is >=4 — mediapipe will crash!'; print('[*] protobuf', v, 'OK')" >nul 2>&1
    if errorlevel 1 (
        echo [!] protobuf version conflict detected — forcing downgrade...
        "%PY%" -m pip install "protobuf>=3.20.0,<4.0.0" --force-reinstall
        if errorlevel 1 goto install_failed
    )
) else (
    echo [*] GPU face detection available (InsightFace). MediaPipe not needed.
)

:: ---------------------------------------------------------------
:: PYANNOTE: Optional diarization (only if HF_TOKEN is configured)
:: ---------------------------------------------------------------
echo.
findstr /r /i "^HF_TOKEN=hf_" .env >nul 2>&1
if errorlevel 1 (
    echo [!] HF_TOKEN not found in .env — skipping pyannote.audio.
    echo [!] To enable speaker diarization: add HF_TOKEN=hf_xxxxxxxxxxxx to .env
    echo [!] then delete ".install_complete" and rerun install.bat.
) else (
    "%PY%" -c "import pyannote.audio" >nul 2>&1
    if errorlevel 1 (
        echo [*] Installing pyannote.audio for speaker diarization...
        "%PY%" -m pip install --prefer-binary "pyannote.audio>=3.1.0" --extra-index-url https://download.pytorch.org/whl/cu126
        if errorlevel 1 (
            echo [!] pyannote.audio install failed. App will run without diarization.
        )
    ) else (
        echo [*] pyannote.audio already installed.
    )
)

:: ---------------------------------------------------------------
:: GPU VERIFICATION
:: ---------------------------------------------------------------
echo.
echo [*] Verifying GPU acceleration stack...
echo.
"%PY%" -c "
import sys
errors = []

# Check CUDA
try:
    import torch
    if torch.cuda.is_available():
        print(f'✓ PyTorch CUDA: Available (GPU: {torch.cuda.get_device_name(0)})')
    else:
        print('✗ PyTorch CUDA: Not available')
        errors.append('PyTorch CUDA')
except Exception as e:
    print(f'✗ PyTorch: {e}')
    errors.append('PyTorch')

# Check OpenCV CUDA
try:
    import cv2
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print(f'✓ OpenCV CUDA: {cv2.cuda.getCudaEnabledDeviceCount()} GPU(s)')
    else:
        print('✗ OpenCV CUDA: Not available (install opencv-contrib-python)')
        errors.append('OpenCV CUDA')
except Exception as e:
    print(f'✗ OpenCV: {e}')
    errors.append('OpenCV')

# Check CuPy
try:
    import cupy
    print(f'✓ CuPy: {cupy.__version__}')
except Exception as e:
    print(f'✗ CuPy: Not available (install cupy-cuda)')
    errors.append('CuPy')

# Check InsightFace
try:
    import insightface
    print('✓ InsightFace: Available')
except Exception as e:
    print(f'✗ InsightFace: Not available (fallback to MediaPipe)')
    errors.append('InsightFace')

# Check ONNX GPU
try:
    import onnxruntime
    providers = onnxruntime.get_available_providers()
    if 'CUDAExecutionProvider' in providers:
        print('✓ ONNX Runtime: GPU backend available')
    else:
        print('✗ ONNX Runtime: GPU backend not available')
        errors.append('ONNX GPU')
except Exception as e:
    print(f'✗ ONNX Runtime: {e}')
    errors.append('ONNX')

if errors:
    print(f'\n[!] {len(errors)} GPU component(s) missing. Some operations will run on CPU.')
else:
    print('\n✓ Full GPU acceleration stack ready!')

sys.exit(0)
"

:: ---------------------------------------------------------------
:: FONTS
:: ---------------------------------------------------------------
echo.
echo [*] Ensuring caption fonts are downloaded...
if exist "download_fonts.py" (
    "%PY%" download_fonts.py
) else (
    echo [!] download_fonts.py not found — skipping font download.
)

:: ---------------------------------------------------------------
:: DONE — stamp the sentinel so future runs skip all of the above
:: ---------------------------------------------------------------
echo.
echo [*] Writing .install_complete sentinel...
echo Installation completed successfully. Delete this file to force a reinstall. > .install_complete

echo.
echo  =======================================
echo   Setup complete. Run run.bat to start.
echo  =======================================
echo.
pause
exit /b 0

:: ---------------------------------------------------------------
:: VERIFY (fast path when sentinel already exists)
:: ---------------------------------------------------------------
:verify_and_exit
if not exist "%PY%" (
    echo [!] Sentinel exists but venv is missing — something went wrong.
    echo [!] Delete ".install_complete" and rerun install.bat.
    pause
    exit /b 1
)

:: Quick GPU verification
echo [*] Quick GPU check...
"%PY%" -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available())" >nul 2>&1
if errorlevel 1 (
    echo [!] Could not verify PyTorch — packages may be broken.
    echo [!] Delete ".install_complete" and rerun install.bat to reinstall.
    pause
    exit /b 1
)

echo [*] All checks passed. Run run.bat to start the app.
echo.
pause
exit /b 0

:install_failed
echo.
echo [!] Setup failed. Check the pip output above for details.
echo [!] Common fixes:
echo [!]   - No internet connection
echo [!]   - Disk full (PyTorch needs ~5 GB free)
echo [!]   - Python version mismatch (need 3.11+)
echo [!]   - CUDA not installed (install NVIDIA CUDA Toolkit 12.x)
echo [!]   - GPU drivers outdated (update from nvidia.com/drivers)
echo.
echo [!] GPU-specific troubleshooting:
echo [!]   - Run 'nvidia-smi' in terminal to check GPU availability
echo [!]   - Ensure CUDA_PATH is set in environment variables
echo [!]   - Try manual install: pip install cupy-cuda12x
pause
exit /b 1
