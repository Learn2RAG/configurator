"""
main.py

Description:
This is the main script of the learn2rag-importer project, which is designed to import and process data for the learn2rag application.

Author: Kyrill MeyerInstitution: IFDT
Version: 0.0.1
Creation Date: June 10, 2025
"""

import json
import os
import logging
from config.config_constants import LOGGING_CONFIG_PATH, JSON_CONFIG_PATH, LOGS_DIR, VERSION
from utils.logging_setup import setup_logging
from utils.config_loader import load_json_config, validate_config_entry
from loaders.process_loaders import process_configuration_entries

#main function to run the application
def main():

    # Display a small textual description about the app
    print("------------------------------------------------------------")
    print("Learn2RAG Importer - DataImporter for Learn2RAG.")
    print(f"Version: {VERSION} | Author: IFDT (KM) | Date: July 21, 2025\n")
    print("https://github.com/Learn2RAG/")
    print("------------------------------------------------------------\n")


    # Ensure the logs directory exists
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    # Check if the logging configuration file exists
    if not os.path.exists(LOGGING_CONFIG_PATH):
        logging.basicConfig()
        logging.error("Logging configuration file not found at %s", LOGGING_CONFIG_PATH)
    else:
        # Set up logging configuration
        setup_logging(LOGGING_CONFIG_PATH)
    logger = logging.getLogger("Learn2RAGImporter")
    logger.info("Application started.")

    # Load JSON configuration
    try:
        config = load_json_config(JSON_CONFIG_PATH)
        logger.info("Configuration loaded successfully, starting validation...")

        # Validate each entry in the configuration
        validation_errors = False
        for index, entry in enumerate(config.get("loaders", []), start=1): 
            try:
                logger.info(f"Validated configuration configuration entry {index}: {entry}")
                validate_config_entry(entry)
            except ValueError as e:
                logger.error(f"Validation error in configuration entry {index}: {e}")
                validation_errors = True

        # Process configuration entries and load documents
        if not validation_errors:
            all_documents = process_configuration_entries(config.get("loaders", []))
            logger.info(f"Total documents loaded: {len(all_documents)}")

            # Optional: Speichern der Dokumente in einer Datei
            output_path = "loaded_documents.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump([{"metadata": doc.metadata, "content": doc.page_content} for doc in all_documents], f, ensure_ascii=False, indent=4)

            print(f"Loaded documents saved to {output_path}")
        else:
            logger.error("Configuration validation failed. No documents were processed.")

    except Exception as e:
        logger.error(f"Error loading configuration: {e}")

    

if __name__ == "__main__":
    main()