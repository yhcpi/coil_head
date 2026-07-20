@echo off
REM Create desktop shortcut (Windows) - run_windows.bat -> Desktop icon

setlocal
cd /d "%~dp0"

set SCRIPT=%~dp0run_windows.bat
set SHORTCUT=%USERPROFILE%\Desktop\CoilTipGUI.lnk

REM Single-line PowerShell (avoid ^ line-continuation fragility)
powershell -NoProfile -Command "$s = (New-Object -COM WScript.Shell).CreateShortcut('%SHORTCUT%'); $s.TargetPath = '%SCRIPT%'; $s.WorkingDirectory = '%~dp0.'; $s.IconLocation = 'shell32.dll,13'; $s.Description = 'Coil Tip Detection GUI (Hyper-YOLO)'; $s.Save(); Write-Host ('OK shortcut: ' + $s.TargetPath)"

if %errorlevel% equ 0 (
    echo.
    echo Desktop shortcut created: %SHORTCUT%
    echo Double-click it to launch GUI.
) else (
    echo.
    echo [ERROR] Shortcut creation failed. Try right-click -^> Run as administrator.
)

pause