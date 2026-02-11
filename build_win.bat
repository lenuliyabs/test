@echo off
setlocal
if not exist .venv (
  echo Please create virtualenv and install dependencies first.
  exit /b 1
)
call .venv\Scripts\activate
pyinstaller --noconfirm --clean HistoAnalyzer.spec
if %errorlevel% neq 0 exit /b %errorlevel%
echo Build completed. See dist\HistoAnalyzer\HistoAnalyzer.exe
endlocal
