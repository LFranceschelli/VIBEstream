@echo off
setlocal
REM ===========================================================================
REM  VIBEstream installer for Windows.  Double-click it (or run it in a terminal).
REM
REM  It creates a virtual environment in this folder (.venv) and installs
REM  VIBEstream and all its Python packages into it.  VibeStream.bat then uses
REM  that environment automatically.
REM
REM  NOT installed by this script (not on PyPI, see README.md):
REM    - Metavision SDK or OpenEB: camera and .raw files.  Install it FIRST.
REM      Metavision 5.x supports Python 3.10 - 3.12: use the same Python here.
REM    - Digilent WaveForms: only for the Analog Discovery (laser trigger, pump).
REM
REM  The environment is created with --system-site-packages, as Prophesee
REM  recommends, so the Metavision Python modules of the SDK stay importable.
REM
REM  To use a specific Python, set it here, e.g.  set "PYTHON=py -3.11"
REM  or the full path of python.exe.
REM ===========================================================================
set "PYTHON="

cd /d "%~dp0"
if defined PYTHON goto :have_python
where py >nul 2>nul
if not errorlevel 1 set "PYTHON=py -3"
if defined PYTHON goto :have_python
where python >nul 2>nul
if not errorlevel 1 set "PYTHON=python"
if defined PYTHON goto :have_python
echo.
echo   Python was not found. Install Python 3.10-3.12 from python.org
echo   (tick "Add python.exe to PATH"), then run this file again.
goto :fail

:have_python
echo.
echo   Using:
%PYTHON% -c "import sys; print('   ', sys.version.split()[0], sys.executable)"
if errorlevel 1 goto :fail

if exist ".venv\Scripts\python.exe" goto :venv_ok
echo.
echo   Creating the environment .venv ...
%PYTHON% -m venv --system-site-packages .venv
if errorlevel 1 goto :fail
:venv_ok
set "VPY=.venv\Scripts\python.exe"

echo.
echo   Installing VIBEstream and its packages ...
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail
"%VPY%" -m pip install -e ".[fast]"
if errorlevel 1 goto :fail

echo.
echo   Check
echo   -----
"%VPY%" -c "import numpy, scipy, cv2, h5py, vibe, vibe_hr; print('    VIBEstream and core packages: OK')"
if errorlevel 1 goto :fail
"%VPY%" -c "import metavision_hal, metavision_core; print('    Metavision: OK')" 2>nul
if errorlevel 1 echo     Metavision: NOT importable - install the Metavision SDK or OpenEB, then run this file again. Offline tools and the resolution enhancement work without it.

echo.
echo   Done.  Start VIBEstream by double-clicking VibeStream.bat
echo   (or run  .venv\Scripts\vibestream ).  Full diagnosis:
echo       .venv\Scripts\python tools\check_env.py
echo.
pause
exit /b 0

:fail
echo.
echo   Installation FAILED - the error is printed above.
echo.
pause
exit /b 1
