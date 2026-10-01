@echo off
setlocal

set "TARGET_DIR=%LOCALAPPDATA%\Learn2RAG"
set "FORCE=0"

if /i "%~1"=="-y" set "FORCE=1"
if /i "%~1"=="/y" set "FORCE=1"

if not exist "%TARGET_DIR%" (
    echo No user data found at %TARGET_DIR%.
    exit /b 0
)

if "%FORCE%"=="0" (
    echo WARNING: This will permanently delete all user data in %TARGET_DIR%.
    set /p CONFIRM="Are you sure? (y/N): "
    if /i not "%CONFIRM%"=="y" (
        echo Aborted. No data was deleted.
        exit /b 1
    )
)

echo Deleting user data...
rd /s /q "%TARGET_DIR%"
echo User data removed.
endlocal