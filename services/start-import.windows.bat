"%LEARN2RAG_PATH%/configurator.exe" learn2rag.importer --config "%IMPORTER_CONFIG%" --logging-config "%STORAGE_PATH%/logging_config.yml"
"%LEARN2RAG_PATH%/configurator.exe" learn2rag.pipeline.ingestion --logging-config "%STORAGE_PATH%/logging_config.yml"
