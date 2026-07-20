@echo off
REM ============================================================
REM Coil Tip Detection GUI - PyInstaller packaging script
REM Run ONCE on a Windows machine with Python + ultralytics
REM Output: dist\CoilTipViz\ (self-contained, no Python needed)
REM
REM Designed for both manual dev builds (looks for hyper-yolo conda env)
REM and GitHub Actions automated builds (uses system python with installed deps).
REM All source paths are relative to this script's directory.
REM ============================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

REM ---- Step 1: Find Python (must have ultralytics + cv2 + av) ----
REM Search order: conda hyper-yolo env -> system python in PATH (CI)
set PYTHON_EXE=
for %%P in (
    "D:\ProgramData\anaconda3\envs\hyper-yolo\python.exe"
    "C:\ProgramData\anaconda3\envs\hyper-yolo\python.exe"
    "%USERPROFILE%\anaconda3\envs\hyper-yolo\python.exe"
    "D:\anaconda3\envs\hyper-yolo\python.exe"
    "C:\anaconda3\envs\hyper-yolo\python.exe"
) do (
    if "!PYTHON_EXE!"=="" if exist %%P set PYTHON_EXE=%%~P
)

REM Fallback: system python in PATH (used by GitHub Actions runner)
if "!PYTHON_EXE!"=="" (
    for /f "delims=" %%I in ('where python 2^>nul') do (
        if "!PYTHON_EXE!"=="" if exist "%%I" set PYTHON_EXE=%%I
    )
)

if "!PYTHON_EXE!"=="" (
    echo [ERROR] No Python found with ultralytics installed.
    echo.
    echo Please install first:
    echo   1. Install Python 3.10 from python.org OR Anaconda
    echo   2. pip install ultralytics==8.0.227 opencv-python pillow av
    echo.
    pause
    exit /b 1
)

echo [OK] Python: !PYTHON_EXE!

REM Verify deps
"!PYTHON_EXE!" -c "import ultralytics, cv2, av" 1>nul 2>nul
if errorlevel 1 (
    echo [ERROR] Missing deps. Please run:
    echo   "!PYTHON_EXE!" -m pip install ultralytics==8.0.227 opencv-python pillow av
    pause
    exit /b 1
)

REM ---- Step 2: Install PyInstaller if missing ----
"!PYTHON_EXE!" -c "import PyInstaller" 1>nul 2>nul
if errorlevel 1 (
    echo [1/4] Installing PyInstaller...
    "!PYTHON_EXE!" -m pip install pyinstaller --quiet
)

REM ---- Step 3: Clean old artifacts ----
echo [2/4] Cleaning old build artifacts...
if exist build rmdir /s /q build
if exist dist  rmdir /s /q dist
if exist CoilTipViz.spec del /q CoilTipViz.spec

REM ---- Step 4: Find weights (try v18.3 deploy, then v26) ----
set WEIGHTS=
for %%W in (
    "..\..\runs\deploy_best\v18_3_epoch60_hard_neg_weak_aug.pt"
    "..\..\runs\coil_panet_ablation\v26_mid_strong_full_300ep\weights\best.pt"
) do (
    if "!WEIGHTS!"=="" if exist %%W set WEIGHTS=%%~W
)

REM If weights missing (e.g. CI build without LFS weights), skip packaging them.
REM End user will pick .pt via GUI's "Select Weight" button on first run.
set WEIGHTS_ARG=
if not "!WEIGHTS!"=="" (
    set WEIGHTS_ARG=--add-data "!WEIGHTS!;weights" ^
    echo [OK] Weights: !WEIGHTS!
) else (
    echo [WARN] No deploy weights found - GUI will prompt to pick one on first run.
)

REM ---- Step 5: Package (this takes 5-10 minutes) ----
echo [3/4] PyInstaller packaging (5-10 minutes, do not close this window)...
"!PYTHON_EXE!" -m PyInstaller ^
    --noconfirm ^
    --noconsole ^
    --onedir ^
    --name CoilTipViz ^
    --collect-all ultralytics ^
    --collect-all torch ^
    --collect-all av ^
    --hidden-import cv2 ^
    --hidden-import PIL ^
    --hidden-import numpy ^
    --paths "%~dp0" ^
    --paths "%~dp0framediff" ^
    --add-data "%~dp0hyper_inference.py;." ^
    --add-data "%~dp0frame_diff_wrapper.py;." ^
    --add-data "%~dp0framediff\frame_diff_detector.py;framediff" ^
    --add-data "%~dp0framediff\change_capture.py;framediff" ^
    --add-data "%~dp0framediff\pyav_reader.py;framediff" ^
    !WEIGHTS_ARG!
    "%~dp0coil_tip_viz_gui.py"

if errorlevel 1 (
    echo [ERROR] PyInstaller failed. See above for details.
    pause
    exit /b 1
)

REM ---- Step 6: Create launcher bat and README ----
echo [4/4] Creating launcher and README...

REM Launcher bat (Chinese named for convenience)
(
    echo @echo off
    echo title Coil Tip Detection GUI
    echo cd /d %%~dp0
    echo CoilTipViz.exe
    echo if errorlevel 1 pause
) > dist\CoilTipViz\启动.bat

REM Distribution README (English, pure ASCII to avoid GBK issues)
(
    echo ============================================================
    echo  Coil Tip Detection GUI - End User Guide
    echo ============================================================
    echo.
    echo  Quick start:
    echo    1. Double-click CoilTipViz.exe
    echo    2. Click "Select Weight" - pick a .pt file ^(or use bundled default^)
    echo    3. Click "Select Video" - pick an .mp4 file
    echo    4. Click "Start" button
    echo.
    echo  File list:
    echo    CoilTipViz.exe   - Main program, double-click to launch
    echo    启动.bat         - Same launcher, Chinese-named
    echo    _internal\       - Dependency files, DO NOT DELETE
    echo.
    echo  System requirements:
    echo    - Windows 10/11 ^(64-bit^)
    echo    - VC++ Redistributable ^(preinstalled on 99%% of PCs^)
    echo    - NVIDIA GPU + driver ^(optional, for GPU acceleration^)
    echo.
    echo  No Python, Anaconda, ultralytics, or torch needed!
    echo.
    echo  Troubleshooting:
    echo    "MSVCP140.dll missing" - install VC++ Redistributable:
    echo      https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo    "No module named cv2" - this build is broken, re-run build_exe.bat
    echo    GUI starts but slow - CPU mode active, install NVIDIA GPU
    echo.
) > dist\CoilTipViz\README.txt

echo.
echo ============================================================
echo  Build complete!
echo.
echo  Output: %cd%\dist\CoilTipViz\
echo    CoilTipViz.exe   (double-click to launch)
echo    启动.bat         (alternate launcher)
echo    README.txt       (end-user guide, ship with .exe)
echo    _internal\       (dependencies - DO NOT modify)
echo.
echo  Total size: ~1.5 GB
echo.
echo  Distribution steps:
echo    1. Right-click dist\CoilTipViz\ folder
echo    2. Send to -^> Compressed (zipped) folder
echo    3. Send CoilTipViz.zip to any Windows PC
echo    4. User unzips and double-clicks CoilTipViz.exe
echo    5. No setup needed - works on fresh Windows install!
echo ============================================================
echo.
pause