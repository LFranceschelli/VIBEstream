@echo off
setlocal
REM ===========================================================================
REM  VibeStream - EBIV run configuration
REM
REM  Double-click this file.  It opens the parameter window and keeps this
REM  console open so you can read the log and any error.
REM
REM  ---------------------------------------------------------------------
REM  IF IT PICKS THE WRONG PYTHON, SET IT HERE.
REM
REM  This matters.  The Metavision HAL plugins are C++ DLLs; whether they
REM  load depends on which DLLs are ahead of them on PATH, so the SAME code
REM  can open the camera under one interpreter and fail with
REM      "No se encuentra el punto de entrada ... hal_plugin_prophesee.dll"
REM  under another.  Anaconda is the usual culprit: its Library\bin shadows
REM  Prophesee's own dependencies.
REM
REM  In the environment where the camera DOES work, run
REM      python -c "import sys; print(sys.executable)"
REM  and paste the printed path between the quotes of PYTHON_EXE below.
REM
REM  If that environment is a conda environment, leave PYTHON_EXE EMPTY and
REM  put the environment name in CONDA_ENV instead.  A conda environment has
REM  to be activated, not just have its python.exe called - activation is
REM  what puts its DLL folders on PATH, and that is the whole problem here.
REM ===========================================================================

set "PYTHON_EXE="
set "CONDA_ENV="

REM ---------------------------------------------------------------------------
REM  MV_HAL_PLUGIN_PATH decides WHICH HAL plugins Metavision looks for.  A
REM  process launched from Explorer can inherit a STALE copy of it (Explorer
REM  caches the environment), which is how the same machine can work from a
REM  cmd prompt and fail from a double-click.  Pin it here to remove the
REM  ambiguity.  Leave empty to inherit whatever the system has.
REM ---------------------------------------------------------------------------
set "MV_PLUGINS="

REM ===========================================================================
cd /d "%~dp0"
if defined MV_PLUGINS set "MV_HAL_PLUGIN_PATH=%MV_PLUGINS%"
title VibeStream - EBIV

set "PY="

if defined CONDA_ENV goto :activate_conda
goto :find_python

REM --- conda environment: activate it, then use whatever python it gives ----
:activate_conda
set "ACT="
for %%A in (
    "%USERPROFILE%\anaconda3\Scripts\activate.bat"
    "%USERPROFILE%\miniconda3\Scripts\activate.bat"
    "C:\ProgramData\anaconda3\Scripts\activate.bat"
    "C:\ProgramData\miniconda3\Scripts\activate.bat"
    "%LOCALAPPDATA%\Continuum\anaconda3\Scripts\activate.bat"
) do (
    if not defined ACT if exist %%A set "ACT=%%~A"
)
if not defined ACT goto :no_conda
echo.
echo   Activating conda environment: %CONDA_ENV%
call "%ACT%" %CONDA_ENV%
goto :find_python

:no_conda
echo.
echo   CONDA_ENV is set to "%CONDA_ENV%" but no Anaconda/Miniconda activate.bat
echo   was found in the usual places.  Either clear CONDA_ENV and set
echo   PYTHON_EXE instead, or add your activate.bat path to the list above.
echo.
pause
exit /b 1

REM --- pick the interpreter --------------------------------------------------
:find_python
if not defined PYTHON_EXE if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not defined PYTHON_EXE goto :auto_python
if not exist "%PYTHON_EXE%" goto :bad_python_exe
set PY="%PYTHON_EXE%"
goto :have_python

:bad_python_exe
echo.
echo   PYTHON_EXE is set to:
echo       %PYTHON_EXE%
echo   but that file does not exist.  Fix it at the top of this file, or
echo   clear it to fall back to auto-detection.
echo.
pause
exit /b 1

:auto_python
where python >nul 2>nul
if not errorlevel 1 set "PY=python"
if defined PY goto :have_python

where py >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if defined PY goto :have_python

for %%P in (
    "%USERPROFILE%\anaconda3\python.exe"
    "%USERPROFILE%\miniconda3\python.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python39\python.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python310\python.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python311\python.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Continuum\anaconda3\python.exe"
    "C:\ProgramData\anaconda3\python.exe"
    "C:\Python39\python.exe"
) do (
    if not defined PY if exist %%P set PY="%%~P"
)
if defined PY goto :have_python

echo.
echo   Could not find Python.
echo.
echo   Set PYTHON_EXE at the top of this file to the full path of the
echo   python.exe you use for this project, or run by hand:
echo       python EBIV_Main.py --gui
echo.
pause
exit /b 1

REM --- report what we are about to use, and whether it can do the job -------
:have_python
echo.
echo   VibeStream
echo   Folder: %CD%
echo.
echo   Interpreter
echo   -----------
%PY% -c "import sys;print('   python     :',sys.version.split()[0]);print('   executable :',sys.executable)" 2>nul
if errorlevel 1 goto :python_broken

%PY% -c "import numpy,scipy,cv2" 2>nul
if errorlevel 1 (echo    numpy/scipy/cv2 : MISSING - the PIV path will not run) else (echo    numpy/scipy/cv2 : OK)

%PY% -c "import metavision_hal,metavision_core" 2>nul
if errorlevel 1 (echo    metavision      : NOT IMPORTABLE - live stream and RAW playback will not run) else (echo    metavision      : import OK)

echo.
echo   MV_HAL_PLUGIN_PATH
echo   ------------------
if defined MV_HAL_PLUGIN_PATH (echo    %MV_HAL_PLUGIN_PATH%) else (echo    NOT SET - HAL will fall back to its install directory, which is a
echo    common cause of the "entry point not found" dialog.)

echo.
echo   If the interpreter above is not the one you normally use for this
echo   project, set PYTHON_EXE or CONDA_ENV at the top of this file.
echo   For a full diagnosis run:  %PY% tools\check_env.py
echo.
echo   Leave this console open: the run log appears here.
echo.

%PY% -u "EBIV_Main.py" --gui
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" (
    echo   ---------------------------------------------------------------
    echo   VibeStream exited with code %RC%.
    echo   The error is printed above. Common causes:
    echo     - the wrong Python was used ^(see the interpreter block above^)
    echo     - a package is missing in this environment
    echo     - the WaveForms application is open and is holding the
    echo       Analog Discovery
    echo   ---------------------------------------------------------------
) else (
    echo   VibeStream closed normally.
)
echo.
pause
exit /b %RC%

:python_broken
echo.
echo   The interpreter %PY% could not be run.
echo   Set PYTHON_EXE at the top of this file to a working python.exe.
echo.
pause
exit /b 1
