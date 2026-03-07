@echo off
setlocal

if "%~1"=="" goto help

if /i "%~1"=="setup" goto setup
if /i "%~1"=="lint" goto lint
if /i "%~1"=="test" goto test
if /i "%~1"=="app" goto app
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

:help
echo Usage: scripts\make.bat ^<setup^|lint^|test^|app^> [extras]
exit /b 1

:end
endlocal
