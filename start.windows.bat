@echo off
setlocal

echo Initializing, please wait...
.\configurator learn2rag.noop
if %ERRORLEVEL% neq 0 goto cleanup

echo Downloading the embedding model, the first time this takes a while...
.\configurator learn2rag.download_models
if %ERRORLEVEL% neq 0 echo Downloading the embedding model failed, imports will try again

echo Starting Learn2RAG...
.\configurator

goto :eof

:cleanup
echo ========================================
echo Initialization failed. Cleaning up, please try to run it again
.\configurator self remove
exit /b 1
