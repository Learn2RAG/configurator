@echo off
setlocal

echo Initializing, please wait...
.\configurator learn2rag.noop
if %ERRORLEVEL% neq 0 goto cleanup

echo Starting Learn2RAG...
.\configurator

goto :eof

:cleanup
echo ========================================
echo Initialization failed. Cleaning up, please try to run it again
.\configurator self remove
exit /b 1
