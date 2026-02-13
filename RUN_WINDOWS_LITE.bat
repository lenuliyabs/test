@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "LOG_FILE=run_lite_install_log.txt"
set "TARGET_PY=3.11"
> "%LOG_FILE%" echo [%DATE% %TIME%] RUN_WINDOWS_LITE start

echo [1/7] Проверка Python %TARGET_PY%...
py -%TARGET_PY% -c "import sys; print(sys.version)" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo [ERROR] Python %TARGET_PY% (x64) не найден.
    echo Установите Python %TARGET_PY% и повторите запуск.
    goto :error
)

echo [2/7] Проверка существующей .venv...
if exist ".venv" (
    if not exist ".venv\.pyver" (
        echo .venv найдена без маркера версии, пересоздание...>> "%LOG_FILE%"
        rmdir /s /q ".venv" >> "%LOG_FILE%" 2>&1
        if errorlevel 1 goto :error
    ) else (
        set "VENV_VER="
        set /p VENV_VER=<".venv\.pyver"
        if not "!VENV_VER!"=="%TARGET_PY%" (
            echo .venv на версии !VENV_VER!, пересоздание на %TARGET_PY%...>> "%LOG_FILE%"
            rmdir /s /q ".venv" >> "%LOG_FILE%" 2>&1
            if errorlevel 1 goto :error
        )
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo [3/7] Создание .venv на Python %TARGET_PY%...
    py -%TARGET_PY% -m venv .venv >> "%LOG_FILE%" 2>&1
    if errorlevel 1 goto :error
    > ".venv\.pyver" echo %TARGET_PY%
)

echo [4/7] Активация окружения...
call ".venv\Scripts\activate.bat" >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto :error

echo [5/7] Обновление pip/setuptools/wheel...
python -m pip install --upgrade pip setuptools wheel >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto :error

echo [6/7] Установка requirements.txt...
python -m pip install -r requirements.txt >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto :error

echo [7/7] Запуск приложения...
python -m app >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto :error

exit /b 0

:error
echo.
echo [ERROR] Установка/запуск завершились с ошибкой.
echo ===== ЛОГ (%LOG_FILE%) =====
type "%LOG_FILE%"
echo =============================
pause
exit /b 1
