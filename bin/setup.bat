@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"
set "EXTRAS=%~1"
if "%EXTRAS%"=="" set "EXTRAS=dev,ui,cirq"

echo [setup.bat] Project root resolved to: %PROJECT_ROOT%
echo [setup.bat] Requested extras: %EXTRAS%

where python >nul 2>&1
if errorlevel 1 (
  echo [setup.bat] ERROR: Python was not found in PATH.
  exit /b 1
)

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1
if errorlevel 1 (
  echo [setup.bat] ERROR: Python 3.12+ is required.
  exit /b 1
)

cd /d "%PROJECT_ROOT%"

if exist ".venv" (
  echo [setup.bat] Step 1/4: Reusing existing .venv
) else (
  echo [setup.bat] Step 1/4: Creating virtual environment in .venv
  python -m venv .venv
  if errorlevel 1 exit /b 1
)

echo [setup.bat] Step 2/4: Upgrading pip
.venv\Scripts\python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

echo [setup.bat] Step 3/4: Installing project dependencies
if "%EXTRAS%"=="" (
  .venv\Scripts\python -m pip install -e .
) else (
  .venv\Scripts\python -m pip install -e .[%EXTRAS%]
)
if errorlevel 1 exit /b 1

echo [setup.bat] Step 4/4: Preparing environment file
if not exist ".env" (
  if exist ".env.example" (
    copy ".env.example" ".env" >nul
    echo [setup.bat] Created .env from .env.example
  )
)

echo [setup.bat] Setup completed. Activate with: .venv\Scripts\activate
endlocal
