"""
main.py

Description:
This is the main script of the learn2rag-importer project, which is designed to import and process data for the learn2rag application.

Author: Kyrill Meyer
Institution: IFDT
Version: 0.0.3
Creation Date: June 10, 2025
Last Modified: February 20, 2026
"""

import argparse
import json
import os
import logging
import sys
import importlib.resources
import warnings  
from pathlib import Path
from .config.config_constants import LOGS_DIR, VERSION
from .utils.logging_setup import setup_logging
from .utils.config_loader import load_json_config, validate_config_entry
from .loaders.process_loaders import process_configuration_entries


logger = logging.getLogger("Learn2RAGImporter")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="magic") # Suppress SyntaxWarnings from the 'magic' module


class ImporterArgumentParser(argparse.ArgumentParser):
    def __init__(self) -> None:
        super().__init__()
        json_config_path = importlib.resources.files("learn2rag.importer.config") / "config.json"
        self.add_argument('--config', default=str(json_config_path))


def init(args):
    # Display a small textual description about the app
    print("------------------------------------------------------------")
    print("Learn2RAG Importer - DataImporter for Learn2RAG.")
    print(f"Version: {VERSION} | Author: IFDT (KM) | Date: February 20, 2026\n")
    print("https://github.com/Learn2RAG/")
    print("------------------------------------------------------------\n")


    # Ensure the logs directory exists
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

     # Paket-relativ path to logging configuration file
    logging_config_path = importlib.resources.files("learn2rag.importer.config") / "logging.yaml"

    # Check if the logging configuration file exists
    if not Path(str(logging_config_path)).exists():
        logging.basicConfig()
        logging.error("Logging configuration file not found at %s", logging_config_path)
    else:
        # Set up logging configuration
        setup_logging(str(logging_config_path))  
    logger.info("Application started.")


#main function to run the application
def main(args: argparse.Namespace) -> None:
    # Load JSON configuration
    try:
        config = load_json_config(args.config)
        logger.info("Configuration loaded successfully, starting validation...")

        # Validate each entry in the configuration
        validation_errors = False
        for index, entry in enumerate(config.get("loaders", []), start=1): 
            try:
                loader_type = entry.get("loader_type", "Unknown")
                logger.info(f"Validated configuration entry {index}: {loader_type}")
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
    args = ImporterArgumentParser().parse_args()
    init(args)
    main(args)
