@echo off
setlocal enabledelayedexpansion

"%LEARN2RAG_PATH%\configurator.exe" learn2rag.importer --config "%IMPORTER_CONFIG%" --logging-config "%STORAGE_PATH%\logging_config.yml"
if %errorlevel% neq 0 exit /b %errorlevel%

"%LEARN2RAG_PATH%\configurator.exe" learn2rag.pipeline.ingestion --logging-config "%STORAGE_PATH%\logging_config.yml"
if %errorlevel% neq 0 exit /b %errorlevel%

if exist "%STORAGE_PATH%\loaded_documents.json" (
    del /f /q "%STORAGE_PATH%\loaded_documents.json"
)