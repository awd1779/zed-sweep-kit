@echo off
rem Double-click to record a sweep, then verify it automatically.
rem Sessions are saved in the "captures" folder next to this file.
rem Extra arguments pass through, e.g.:  capture.bat --duration 20

cd /d "%~dp0"

rem Uses the "python" on PATH. If ZED's Python API lives in another
rem environment (e.g. conda), put its python.exe path here instead.
set PY=python

%PY% -c "import pyzed.sl" 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: Python cannot import pyzed. Install the ZED SDK and its Python
    echo API first - see README.md, step 1.
    echo.
    pause
    exit /b 1
)

%PY% scripts\zed_capture_session.py --res HD1080 --fps 15 --lock-wb 5600 --outdir captures %*

echo.
pause
