@echo off
REM ============================================================
REM Coil Tip Detection GUI - PyInstaller launcher
REM This bat is a thin wrapper that finds Python and calls build_exe.py
REM which handles all quoting/argv properly.
REM ============================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

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

if "!PYTHON_EXE!"=="" (
    for /f "delims=" %%I in ('where python 2^>nul') do (
        if "!PYTHON_EXE!"=="" if exist "%%I" set PYTHON_EXE=%%I
    )
)

if "!PYTHON_EXE!"=="" (
    echo [ERROR] No Python found. Run install_hyper_yolo.bat first.
    pause
    exit /b 1
)

"!PYTHON_EXE!" "%~dp0build_exe.py"
exit /b %errorlevel%