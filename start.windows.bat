@echo off
setlocal

echo Starting Learn2RAG...
echo Initializing, please wait...
echo ========================================

echo 1/3 Initializing open-webui...
call services\start-open-webui main --version
if %ERRORLEVEL% neq 0 goto cleanup

echo ========================================
echo 2/3 Preparing configurator...
set FLASK_DEBUG=1
.\configurator --prepare-only
if %ERRORLEVEL% neq 0 goto cleanup

echo ========================================
echo 3/3 Starting Learn2RAG...
.\configurator

goto :eof

:cleanup
echo ========================================
echo Initialization failed. Cleaning up pyapp related files...
.\configurator self restore
exit /b 1