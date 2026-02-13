@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "LOG_FILE=run_full_ai_install_log.txt"
> "%LOG_FILE%" echo [%%DATE%% %%TIME%%] RUN_WINDOWS_FULL_AI start

echo [1/7] Проверка Python 3.11...
py -3.11 -c "import sys; print(sys.version)" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python 3.11 не найден через py launcher.
    echo Установите Python 3.11 (x64) и повторите запуск.
    echo.
    type "%LOG_FILE%"
    pause
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    set "VENV_PY_VER="
    for /f "usebackq delims=" %%V in (`".venv\Scripts\python.exe" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"`) do set "VENV_PY_VER=%%V"
    echo Existing .venv version: !VENV_PY_VER!>> "%LOG_FILE%"
    if not "!VENV_PY_VER!"=="3.11" (
        echo [2/7] Найдена .venv на Python !VENV_PY_VER!, пересоздаю на 3.11...
        rmdir /s /q ".venv" >> "%LOG_FILE%" 2>&1
        if exist ".venv" (
            echo [ERROR] Не удалось удалить старую .venv>> "%LOG_FILE%"
            goto :error
        )
    ) else (
        echo [2/7] Найдена корректная .venv (Python 3.11).
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo [3/7] Создание .venv через Python 3.11...
    py -3.11 -m venv .venv >> "%LOG_FILE%" 2>&1
    if errorlevel 1 goto :error
)

echo [4/7] Активация окружения...
call ".venv\Scripts\activate.bat" >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto :error

echo [5/7] Обновление pip/setuptools/wheel...
python -m pip install --upgrade pip setuptools wheel >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto :error

if not exist "requirements.txt" (
    echo [ERROR] requirements.txt не найден в корне проекта.>> "%LOG_FILE%"
    goto :error
)
if not exist "requirements_full_windows.txt" (
    echo [ERROR] requirements_full_windows.txt не найден в корне проекта.>> "%LOG_FILE%"
    goto :error
)

echo [6/7] Установка requirements.txt...
python -m pip install -r requirements.txt >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto :error

echo [7/7] Установка requirements_full_windows.txt...
python -m pip install -r requirements_full_windows.txt >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto :error

echo Запуск приложения...
python -m app.main >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto :error

echo.
echo Готово. Лог установки: %LOG_FILE%
exit /b 0

:error
echo.
echo [ERROR] Установка/запуск завершились с ошибкой.
echo Полный лог: %LOG_FILE%
echo.
type "%LOG_FILE%"
pause
exit /b 1
