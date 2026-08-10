@echo off
setlocal

echo Initializing, please wait...
echo ========================================

echo 1/3 Initializing open-webui...
.\services\start-open-webui main --version
if %ERRORLEVEL% neq 0 goto cleanup

echo ========================================
echo 2/3 Initializing Learn2RAG...
.\configurator learn2rag.noop
if %ERRORLEVEL% neq 0 goto cleanup

echo ========================================
echo 3/3 Starting Learn2RAG...
.\configurator

goto :eof

:cleanup
echo ========================================
echo Initialization failed. Cleaning up, please try to run it again
.\services\start-open-webui self remove
.\configurator self remove
exit /b 1
