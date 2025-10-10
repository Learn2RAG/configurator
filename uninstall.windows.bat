@echo off
rem
rem TODO: use https://ofek.dev/pyapp/latest/runtime/#remove ?
rd /s /q "%LOCALAPPDATA%/pyapp/data/learn2rag"
rd /s /q "%LOCALAPPDATA%/pyapp/data/importer"
rd /s /q "%LOCALAPPDATA%/pyapp/data/basic-pipeline"
rd /s /q "%LOCALAPPDATA%/pyapp/data/open-webui"
rd /s /q "%LOCALAPPDATA%/pyapp/data/open-webui-pipelines"
echo Done
