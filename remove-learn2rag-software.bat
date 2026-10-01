@echo off
setlocal

echo Removing Learn2RAG application runtimes...
if exist "%LOCALAPPDATA%\pyapp\data\learn2rag" rd /s /q "%LOCALAPPDATA%\pyapp\data\learn2rag"
if exist "%LOCALAPPDATA%\pyapp\data\open-webui" rd /s /q "%LOCALAPPDATA%\pyapp\data\open-webui"
if exist "%LOCALAPPDATA%\pyapp\data\open-webui-pipelines" rd /s /q "%LOCALAPPDATA%\pyapp\data\open-webui-pipelines"

echo Software removed successfully.
endlocal