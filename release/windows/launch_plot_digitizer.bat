@echo off
setlocal
set "APP_DIR=%~dp0"
set "NATIVE_PYTHON=%APP_DIR%python314\python.exe"
set "NATIVE_PYTHONW=%APP_DIR%python314\pythonw.exe"
set "LOG=%APP_DIR%plot_digitizer_launch.log"
if not exist "%NATIVE_PYTHON%" goto missing_python
if not exist "%NATIVE_PYTHONW%" goto missing_python
if not exist "%APP_DIR%runtime_deps\py314\PySide6" goto install_runtime
if not exist "%APP_DIR%runtime_deps\py314\numpy" goto install_runtime
if not exist "%APP_DIR%runtime_deps\py314\matplotlib" goto install_runtime
if not exist "%APP_DIR%runtime_deps\py314\cv2" goto install_runtime
goto ready
:install_runtime
if not exist "%APP_DIR%wheelhouse_py314" (
    echo Runtime files are missing. Re-extract the ready package or use the offline-wheels package.
    pause
    exit /b 1
)
call "%APP_DIR%install_dependencies.bat"
if errorlevel 1 exit /b 1
:ready
"%NATIVE_PYTHON%" -I -B "%APP_DIR%verify_offline_bundle.py" > "%LOG%" 2>&1
if errorlevel 1 (
    echo Startup verification failed. Details:
    type "%LOG%"
    pause
    exit /b 1
)
start "PlotDigitizer" /D "%APP_DIR%" "%NATIVE_PYTHONW%" -I -B "%APP_DIR%app_launcher.py"
exit /b 0
:missing_python
echo The bundled Python runtime is missing. Extract the complete release ZIP again.
pause
exit /b 1
