@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0\..\.."

if not exist .venv (
  echo [INFO] Creating virtualenv...
  py -3 -m venv .venv
)

call .venv\Scripts\activate
if errorlevel 1 (
  echo [ERROR] Failed to activate .venv
  exit /b 1
)

python -m pip install --upgrade pip
python -m pip install -r requirements_full_windows.txt
if errorlevel 1 (
  echo [ERROR] Failed to install dependencies.
  exit /b 1
)

pyinstaller --noconfirm --clean build\windows\histoanalyzer.spec
if errorlevel 1 (
  echo [ERROR] PyInstaller build failed.
  exit /b 1
)

echo [OK] EXE build finished: dist\HistoAnalyzer\HistoAnalyzer.exe
endlocal
