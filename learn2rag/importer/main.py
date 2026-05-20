"""
main.py

Description:
This is the main script of the learn2rag-importer project, which is designed to import and process data for the learn2rag application.

Author: Kyrill Meyer
Institution: IFDT
Version: 0.0.4
Creation Date: June 10, 2025
Last Modified: May 20, 2026
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
from .loaders.process_loaders import process_configuration_entries, process_delta_imports
from .utils.import_state import ImportState
from learn2rag.pipeline.ingestion import index


logger = logging.getLogger("Learn2RAGImporter")
statusLogger = logging.getLogger('status')
warnings.filterwarnings("ignore", category=SyntaxWarning, module="magic") # Suppress SyntaxWarnings from the 'magic' module


class ImporterArgumentParser(argparse.ArgumentParser):
    """Argument parser for the Learn2RAG importer.

    Arguments:
        --config          Path to the importer config JSON file.
                          Default: config.json bundled with the package.
        --state-file      Path to the import-state JSON file that persists per-loader
                          last-import timestamps across runs.
                          Default: import_state.json placed next to --config.
        --delta           Run a delta import instead of a full import.
                          Intelligent loaders (Drupal, SharePoint) fetch only documents
                          changed since the last run; plain loaders (Directory, HTML, CSV)
                          perform a SHA-256 content-hash comparison against the current
                          Qdrant index and only update changed documents.
                          On the very first run (no state file) a full import is performed
                          automatically.
        --save-documents  Write all loaded documents to loaded_documents.json in the
                          current working directory after a full import.
                          Intended for debugging and backwards compatibility only;
                          disabled by default.

    Environment variables (read by main(), not CLI arguments):
        PIPELINE_USER_CONFIG  Path to the pipeline user_config.json.
                              Default: learn2rag/pipeline/user_config.json
        PIPELINE_OPT_CONFIG   Path to the pipeline opt_config.json.
                              Default: learn2rag/pipeline/opt_config.json
    """

    def __init__(self) -> None:
        super().__init__()
        json_config_path = importlib.resources.files("learn2rag.importer.config") / "config.json"
        self.add_argument('--config', default=str(json_config_path),
                          help='path to the importer config JSON (default: bundled config.json)')
        self.add_argument('--state-file', default=None,
                          help='path to the import state JSON file (default: import_state.json next to --config)')
        self.add_argument('--delta', action='store_true',
                          help='perform a delta import instead of a full import')
        self.add_argument('--save-documents', action='store_true',
                          help='write loaded documents to loaded_documents.json (debug/backwards-compat only)')


def init(args: argparse.Namespace) -> None:
    # Display a small textual description about the app
    print("------------------------------------------------------------")
    print("Learn2RAG Importer - DataImporter for Learn2RAG.")
    print(f"Version: {VERSION} | Author: IFDT (KM) | Date: May 20, 2026\n")
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
    statusLogger.info('Import started')
    try:
        config = load_json_config(args.config)
        logger.debug("Configuration loaded successfully, starting validation...")

        # load pipeline configuration for user and opt settings, needed for delta-import and indexing
        user_config_path = os.environ.get("PIPELINE_USER_CONFIG", "learn2rag/pipeline/user_config.json")
        opt_config_path = os.environ.get("PIPELINE_OPT_CONFIG", "learn2rag/pipeline/opt_config.json")
        with open(user_config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        with open(opt_config_path, "r", encoding="utf-8") as f:
            opt_config = json.load(f)
        logger.debug("Pipeline configuration loaded from '%s' and '%s'.", user_config_path, opt_config_path)

        # save import state file next to the importer config if not explicitly specified

        state_file_path = args.state_file if args.state_file else str(Path(args.config).parent / "import_state.json")
        import_state = ImportState(state_file_path)

        # validate configuration entries before processing to avoid partial imports and ensure all issues are caught upfront
        validation_errors = False
        for entry_idx, entry in enumerate(config.get("loaders", []), start=1):
            try:
                loader_type = entry.get("loader_type", "Unknown")
                logger.debug(f"Validated configuration entry {entry_idx}: {loader_type}")
                validate_config_entry(entry)
            except ValueError as e:
                logger.error(f"Validation error in configuration entry {entry_idx}: {e}")
                validation_errors = True

        if not validation_errors:
            if args.delta:
                # Delta-Import: Hash-/Timestamp-comparison, direct ingest in Qdrant,
                logger.info("Running delta import (state file: %s)", state_file_path)
                process_delta_imports(
                    config_entries=config.get("loaders", []),
                    user_config=user_config,
                    opt_config=opt_config,
                    import_state=import_state,
                )
            else:
                # full import: all documents load and directly ingest in Qdrant
                logger.info("Running full import")
                all_documents = process_configuration_entries(config.get("loaders", []))
                logger.debug(f"Total documents loaded: {len(all_documents)}")
                index(all_documents, user_config, opt_config)

                # JSON-Dump für Rückwärtskompatibilität (nur mit --save-documents)
                if args.save_documents:
                    output_path = "loaded_documents.json"
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump([{"metadata": doc.metadata, "content": doc.page_content} for doc in all_documents], f, ensure_ascii=False, indent=4)
                    logger.debug('Documents saved to: %s', output_path)

            statusLogger.info('Import finished')
        else:
            logger.error("Configuration validation failed. No documents were processed.")
            statusLogger.error('Import failed')
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        statusLogger.error('Import failed')
        sys.exit(1)


if __name__ == "__main__":
    args = ImporterArgumentParser().parse_args()
    init(args)
    main(args)
