@echo off
REM ============================================================
REM Auto-install hyper-yolo env on Windows
REM Detects existing Anaconda, creates hyper-yolo env, installs deps
REM ============================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo  Hyper-YOLO Environment Installer
echo ============================================================
echo.

REM ---- Find Anaconda root ----
set CONDA_ROOT=
for %%P in (
    "D:\ProgramData\anaconda3"
    "C:\ProgramData\anaconda3"
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\miniconda3"
    "D:\anaconda3"
    "C:\anaconda3"
    "E:\anaconda3"
) do (
    if "!CONDA_ROOT!"=="" (
        if exist %%P (
            set CONDA_ROOT=%%~P
        )
    )
)

if "!CONDA_ROOT!"=="" (
    echo [ERROR] No Anaconda installation found.
    echo.
    echo Please install Anaconda first:
    echo   https://www.anaconda.com/download
    echo.
    pause
    exit /b 1
)

echo [OK] Anaconda found: !CONDA_ROOT!
echo.

set CONDA=!CONDA_ROOT!\Scripts\conda.exe
if not exist "!CONDA!" (
    echo [ERROR] conda.exe not found at !CONDA!
    pause
    exit /b 1
)

REM ---- Create env (skip if already exists) ----
echo [1/4] Creating hyper-yolo env (python 3.10)...
"!CONDA!" create -n hyper-yolo python=3.10 -y
if errorlevel 1 (
    echo [WARN] conda create returned error, env may already exist
)

REM ---- Direct path to env python (avoid activate.bat complexity) ----
set ENV_PY=!CONDA_ROOT!\envs\hyper-yolo\python.exe
set ENV_PIP=!CONDA_ROOT!\envs\hyper-yolo\Scripts\pip.exe

if not exist "!ENV_PY!" (
    echo [ERROR] Env python not found at !ENV_PY!
    pause
    exit /b 1
)

echo [OK] Env python: !ENV_PY!

REM ---- Install deps ----
echo [2/4] Upgrading pip...
"!ENV_PY!" -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [WARN] pip upgrade failed, continuing anyway
)

echo [3/4] Configuring Tsinghua mirror (fast in China)...
"!ENV_PY!" -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
"!ENV_PY!" -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

echo [3/4] Installing pytorch (GPU cu121, pinned 2.5.1) + ultralytics + cv2 + av (retry up to 3 times)...
echo        pinning torch==2.5.1 to match training env (avoids torch 2.6+ weights_only bug)
set ATTEMPT=0
:install_retry
set /a ATTEMPT+=1
REM ---- Pin GPU torch (matching training env: torch==2.5.1+cu121) ----
"!ENV_PY!" -m pip install --quiet "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 goto install_continue_ultralytics
REM ---- Then install ultralytics (already has cuda torch via line above) ----
"!ENV_PY!" -m pip install --quiet ultralytics==8.0.227 opencv-python pillow av ttkbootstrap
goto install_done
:install_continue_ultralytics
if !ATTEMPT! geq 3 goto install_fail
echo [WARN] Attempt !ATTEMPT! failed, retrying in 5s...
timeout /t 5 /nobreak >nul
goto install_retry
:install_done
echo [OK] pip install succeeded
goto install_verify
:install_fail
echo [ERROR] pip install failed after 3 attempts.
pause
exit /b 1
:install_verify
if errorlevel 1 (
    echo [ERROR] pip install failed
    pause
    exit /b 1
)

REM ---- Verify ----
echo [4/4] Verifying installation...
"!ENV_PY!" -c "import ultralytics; import cv2; import av; import PIL; import ttkbootstrap; print('OK all imports')"
if errorlevel 1 (
    echo [WARN] Some imports failed. Manual check:
    echo   !ENV_PY! -c "import ultralytics; import cv2; import av"
    pause
    exit /b 1
)

REM ---- Show GPU info ----
"!ENV_PY!" -c "import torch; print(f'torch={torch.__version__}  cuda={torch.cuda.is_available()}' + (f'  dev={torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '  (CPU only)'))"

echo.
echo ============================================================
echo  OK hyper-yolo env ready!
echo.
echo  Python: !ENV_PY!
echo.
echo  Next: double-click run_windows.bat to launch GUI
echo ============================================================
echo.
pause