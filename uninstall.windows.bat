@echo off
setlocal

set "SCRIPT_DIR=%~dp0"

if exist "%SCRIPT_DIR%remove-learn2rag-software.bat" (
    call "%SCRIPT_DIR%remove-learn2rag-software.bat"
) else (
    echo Error: %SCRIPT_DIR%remove-learn2rag-software.bat not found.
    exit /b 1
)

echo.

set /p PURGE_DATA="Do you also want to remove all personal user data (databases, configs)? (y/N): "

if /i "%PURGE_DATA%"=="y" (
    if exist "%SCRIPT_DIR%remove-user-data.bat" (
        call "%SCRIPT_DIR%remove-user-data.bat" /y
    ) else (
        echo Error: %SCRIPT_DIR%remove-user-data.bat not found.
        exit /b 1
    )
) else (
    echo User data preserved.
)

echo Uninstall process finished.
endlocal