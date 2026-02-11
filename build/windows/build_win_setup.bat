@echo off
setlocal

cd /d "%~dp0\..\.."

call build\windows\build_win_exe.bat
if errorlevel 1 exit /b 1

set ISCC_PATH=
for %%I in (iscc.exe) do set ISCC_PATH=%%~$PATH:I

if "%ISCC_PATH%"=="" (
  if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" set ISCC_PATH=C:\Program Files (x86)\Inno Setup 6\ISCC.exe
)
if "%ISCC_PATH%"=="" (
  if exist "C:\Program Files\Inno Setup 6\ISCC.exe" set ISCC_PATH=C:\Program Files\Inno Setup 6\ISCC.exe
)

if "%ISCC_PATH%"=="" (
  echo [WARN] Inno Setup Compiler ^(iscc.exe^) not found.
  echo Install Inno Setup: https://jrsoftware.org/isinfo.php
  echo Then add ISCC.exe to PATH or install to default folder.
  exit /b 1
)

"%ISCC_PATH%" installer\HistoAnalyzer.iss
if errorlevel 1 (
  echo [ERROR] Inno Setup build failed.
  exit /b 1
)

echo [OK] Setup build finished: release\Setup_HistoAnalyzer.exe
endlocal
