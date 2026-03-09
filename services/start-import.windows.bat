"%LEARN2RAG_PATH%/configurator.exe" learn2rag.importer --config "%STORAGE_PATH%/importer_config.json" --logging-config "%STORAGE_PATH%/logging_config.yml"
"%LEARN2RAG_PATH%/configurator.exe" learn2rag.pipeline.ingestion --logging-config "%STORAGE_PATH%/logging_config.yml"
