@echo off
setlocal
set "APP_DIR=%~dp0"
set "PYTHON=%APP_DIR%python314\python.exe"
if not exist "%PYTHON%" (
    echo The bundled Python runtime is missing. Extract the complete release ZIP again.
    pause
    exit /b 1
)
"%PYTHON%" -I -B "%APP_DIR%install_offline_dependencies.py"
if errorlevel 1 goto failed
"%PYTHON%" -I -B "%APP_DIR%verify_offline_bundle.py"
if errorlevel 1 goto failed
echo Offline installation and startup verification passed.
exit /b 0
:failed
echo Offline installation failed. No network or pip was used.
pause
exit /b 1
