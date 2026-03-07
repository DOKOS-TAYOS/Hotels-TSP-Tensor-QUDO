@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "EXTRAS=%~1"
if "%EXTRAS%"=="" set "EXTRAS=dev,ui,cirq"

echo [install.bat] Step 1/3: Checking prerequisites (Python 3.12+ and Git)

where git >nul 2>&1
if errorlevel 1 (
  echo [install.bat] ERROR: Git was not found in PATH.
  echo [install.bat] Install Git from https://git-scm.com/download/win and rerun.
  exit /b 1
)

where python >nul 2>&1
if errorlevel 1 (
  echo [install.bat] ERROR: Python was not found in PATH.
  echo [install.bat] Install Python 3.12+ from https://www.python.org/downloads/windows/ and rerun.
  exit /b 1
)

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1
if errorlevel 1 (
  echo [install.bat] ERROR: Python 3.12+ is required.
  echo [install.bat] Current Python does not satisfy project requirements.
  exit /b 1
)

echo [install.bat] Step 2/3: Prerequisites OK
echo [install.bat] Step 3/3: Running setup script
call "%SCRIPT_DIR%bin\setup.bat" "%EXTRAS%"
if errorlevel 1 exit /b 1

echo [install.bat] Installer completed successfully.

endlocal
