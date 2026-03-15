@echo off
setlocal

if "%~1"=="" goto help

if /i "%~1"=="setup" goto setup
if /i "%~1"=="lint" goto lint
if /i "%~1"=="test" goto test
if /i "%~1"=="app" goto app
if /i "%~1"=="clean" goto clean
goto help

:setup
if "%~2"=="" (
  call "%~dp0..\bin\setup.bat" dev,ui,cirq
) else (
  call "%~dp0..\bin\setup.bat" "%~2"
)
goto end

:lint
.venv\Scripts\python -m ruff check .
goto end

:test
.venv\Scripts\python -m pytest
goto end

:app
.venv\Scripts\streamlit run src\streamlit_app\app.py
goto end

:clean
echo [make.bat] Cleaning Python caches and temporary artifacts...

if exist ".pytest_cache" rd /s /q ".pytest_cache" >nul 2>&1
if exist ".ruff_cache" rd /s /q ".ruff_cache" >nul 2>&1
if exist "htmlcov" rd /s /q "htmlcov" >nul 2>&1
if exist ".tmp" rd /s /q ".tmp" >nul 2>&1
if exist "tests\.tmp" rd /s /q "tests\.tmp" >nul 2>&1
if exist ".coverage" del /q ".coverage" >nul 2>&1

for /d /r %%D in (__pycache__) do (
  if exist "%%D" rd /s /q "%%D" >nul 2>&1
)

del /s /q *.pyc >nul 2>&1
del /s /q *.pyo >nul 2>&1

for /d %%D in (pytest-cache-files-*) do (
  if exist "%%D" rd /s /q "%%D" >nul 2>&1
)

echo [make.bat] Clean completed.
goto end

:help
echo Usage: scripts\make.bat ^<setup^|lint^|test^|app^|clean^> [extras]
exit /b 1

:end
endlocal
