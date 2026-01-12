"""
process_loaders.py

Description:
This module processes configuration entries and delegates loading to specific loader functions.

Author: Kyrill Meyer
Institution: IFDT
Version: 0.0.3
Creation Date: June 10, 2025
Last Modified: Jan 12, 2026
"""

import logging
from ..globals import stop_loading
from .directory_loader import load_from_directory
from .csv_loader import load_from_csv
from .html_loader import load_html_content

#
# initialize logger
logger = logging.getLogger("Learn2RAGImporter")

def process_configuration_entries(config_entries):
    """
    Process configuration entries and load documents based on loader type.

    Args:
        config_entries (list): List of configuration entries.

    Returns:
        list: List of loaded documents.
    """

    all_documents = []

    for entry in config_entries:
        loader_type = entry.get("loader_type")

        if not loader_type:
            logger.error(f"Invalid configuration entry: {entry}")
            continue

        try:
            logger.info(f"Processing entry: {entry}, please wait...")
            if loader_type == "DirectoryLoader":
                path = entry.get("path")
                recursive = entry.get("recursive", False)
                if not path:
                    logger.error("Missing 'path' for 'DirectoryLoader' in configuration entry.")
                    continue
                documents = load_from_directory(path, recursive=recursive)
                logger.info(f"Loaded {len(documents)} documents from {path} using {loader_type}.")
            elif loader_type == "CSVLoader":
                if not path:
                    logger.error("Missing 'path' for 'CSVLoader' in configuration entry.")
                    continue
                documents = load_from_csv(path)
                logger.info(f"Loaded {len(documents)} documents from {path} using {loader_type}.")
            elif loader_type == "HTMLLoader":
                url = entry.get("url")
                depth = entry.get("depth", 0)
                if not url or not isinstance(depth, int) or depth < 0:
                    logger.error(f"Invalid configuration for HTMLLoader: {entry}")
                    continue
                documents = load_html_content(url, depth=depth)
                logger.info(f"Loaded {len(documents)} documents from {url} using {loader_type}.")
            else:
                logger.error(f"Unknown loader type: {loader_type}")
                continue

            all_documents.extend(documents)
        except Exception as e:
            logger.error(f"Error processing entry {entry}: {e}")

    return all_documents

