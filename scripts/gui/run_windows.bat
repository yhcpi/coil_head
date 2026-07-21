@echo off
REM ============================================================
REM Coil Tip Detection GUI - Windows double-click launcher
REM Auto-detect: anaconda env / system Python / WSL fallback
REM ============================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

set PYTHON_EXE=
set PYTHON_SOURCE=

REM ---- Strategy 1: Try hyper-yolo env in common Anaconda locations ----
for %%P in (
    "C:\ProgramData\anaconda3\envs\hyper-yolo\python.exe"
    "%USERPROFILE%\anaconda3\envs\hyper-yolo\python.exe"
    "D:\anaconda3\envs\hyper-yolo\python.exe"
    "C:\anaconda3\envs\hyper-yolo\python.exe"
    "D:\ProgramData\anaconda3\envs\hyper-yolo\python.exe"
    "E:\anaconda3\envs\hyper-yolo\python.exe"
) do (
    if "!PYTHON_EXE!"=="" (
        if exist %%P (
            set PYTHON_EXE=%%~P
            set PYTHON_SOURCE=anaconda-hyper-yolo
        )
    )
)

REM ---- Strategy 2: Parse `where python` and verify it has ultralytics/cv2 ----
if "!PYTHON_EXE!"=="" (
    for /f "delims=" %%I in ('where python 2^>nul') do (
        if "!PYTHON_EXE!"=="" (
            if exist "%%I" (
                REM Check if this python has cv2 (proxy for full install)
                "%%I" -c "import cv2, ultralytics, av" 1>nul 2>nul
                if not errorlevel 1 (
                    set PYTHON_EXE=%%I
                    set PYTHON_SOURCE=system-path-with-deps
                )
            )
        )
    )
)

REM ---- Strategy 3: WSL fallback (already-tested hyper-yolo env) ----
if "!PYTHON_EXE!"=="" (
    where wsl 1>nul 2>nul
    if not errorlevel 1 (
        echo [INFO] No Windows Python with ultralytics found.
        echo [INFO] Falling back to WSL Ubuntu hyper-yolo env...
        wsl -d Ubuntu -- bash -lc "test -x /home/pi/anaconda3/envs/hyper-yolo/bin/python && /home/pi/anaconda3/envs/hyper-yolo/bin/python /mnt/d/gui/coil_tip_viz_gui.py"
        exit /b !errorlevel!
    )
)

if "!PYTHON_EXE!"=="" (
    echo.
    echo ============================================================
    echo [ERROR] No usable Python found.
    echo.
    echo Tried:
    echo   - 6 common Anaconda hyper-yolo env paths
    echo   - Any python.exe in PATH
    echo   - WSL Ubuntu hyper-yolo env
    echo.
    echo Run install_hyper_yolo.bat to create hyper-yolo env.
    echo See scripts\gui\WINDOWS_INSTALL.md for details.
    echo ============================================================
    pause
    exit /b 1
)

echo [OK] Python: %PYTHON_EXE%  (source: %PYTHON_SOURCE%)

REM ---- Frame diff dir (optional) ----
if exist "C:\projects\mm\FrameDiff" (
    set FRAMEDIFF_DIR=C:\projects\mm\FrameDiff
) else if exist "C:\projects\mm\framediff" (
    set FRAMEDIFF_DIR=C:\projects\mm\framediff
) else (
    set FRAMEDIFF_DIR=
)

if not "%FRAMEDIFF_DIR%"=="" (
    echo [OK] Frame diff: %FRAMEDIFF_DIR%
    set PYTHONPATH=%FRAMEDIFF_DIR%;%~dp0
) else (
    echo [WARN] Frame diff dir not found, model inference still works
    REM 路线 B (zip 解压): framediff/ 在 %~dp0framediff 下, 需把它加到 path
    set PYTHONPATH=%~dp0;%~dp0framediff
)

REM ---- Optional: drag video file as %1 ----
set VIDEO_ARG=
if not "%~1"=="" set VIDEO_ARG=--video "%~1"

REM ---- Launch ----
echo [LAUNCH] Starting GUI (close this window to quit)...
"%PYTHON_EXE%" "%~dp0coil_tip_viz_gui.py" %VIDEO_ARG%

if errorlevel 1 (
    echo.
    echo [ERROR] GUI exited with code %errorlevel%
    echo Common fixes:
    echo   1. Run install_hyper_yolo.bat to install deps
    echo   2. See scripts\gui\WINDOWS_INSTALL.md troubleshooting
    pause
)