"""
process_loaders.py

Description:
This module processes configuration entries and delegates loading to specific loader functions.

Author: Kyrill Meyer
Institution: IFDT
Version: 0.0.5
Creation Date: June 10, 2025
Last Modified: March 17, 2026
"""

import logging
from typing import List, Dict, Any
from ..globals import stop_loading
from langchain_core.documents import Document
from .directory_loader import load_from_directory
from .csv_loader import load_from_csv
from .html_loader import load_html_content
from .sharepoint_loader import load_from_sharepoint
from .drupal_loader import load_from_drupal

#
# initialize logger
logger = logging.getLogger("Learn2RAGImporter")

def process_configuration_entries(config_entries: List[Dict[str, Any]]) -> List[Document]:
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
            loader_id = entry.get("loader_id") or ""
            if not loader_id:
                logger.warning(f"No loader_id specified for entry: {entry}.\nIt is recommended to set a unique loader_id for each loader.")
            if loader_type == "DirectoryLoader":
                path = entry.get("path")
                recursive = entry.get("recursive", False)
                silent_errors = entry.get("silent_errors", True)
                if not path:
                    logger.error("Missing 'path' for 'DirectoryLoader' in configuration entry.")
                    continue
                documents = load_from_directory(path, recursive=recursive, silent_errors=silent_errors, loader_id=loader_id)
                logger.info(f"Loaded {len(documents)} documents from {path} using {loader_type} for configuration entry with loader_id: {loader_id}.")
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
                documents = load_html_content(url, depth=depth, loader_id=loader_id)
                logger.info(f"Loaded {len(documents)} documents from {url} using {loader_type}.")
            elif loader_type == "SharepointLoader":
                client_id = entry.get("client_id")
                client_secret = entry.get("client_secret")
                document_library_id = entry.get("document_library_id")
                tenant_id = entry.get("tenant_id", "common")
                folder_path = entry.get("folder_path")
                folder_id = entry.get("folder_id")
                site_id = entry.get("site_id")
                object_ids = entry.get("object_ids")

                recursive = entry.get("recursive", False)
                if isinstance(recursive, str):
                    recursive = recursive.lower() == "true"

                auth_with_token = entry.get("auth_with_token", False)
                if isinstance(auth_with_token, str):
                    auth_with_token = auth_with_token.lower() == "true"

                reset_token = entry.get("reset_token", False)
                if isinstance(reset_token, str):
                    reset_token = reset_token.lower() == "true"

                if not client_id or not client_secret or not tenant_id or not document_library_id:
                    logger.error(
                        f"Invalid configuration for SharepointLoader: Missing required parameters. Entry: {entry}")
                    continue

                documents = load_from_sharepoint(
                    client_id=client_id,
                    client_secret=client_secret,
                    document_library_id=document_library_id,
                    folder_path=folder_path,
                    folder_id=folder_id,
                    object_ids=object_ids,
                    recursive=recursive,
                    auth_with_token=auth_with_token,
                    reset_token=reset_token,
                    tenant_id=tenant_id,
                    site_id=site_id,
                    loader_id=loader_id
                )
                logger.info(f"Loaded {len(documents)} documents from SharePoint using {loader_type}.")

            elif loader_type == "DrupalLoader":
                base_url = entry.get("base_url")
                content_types = entry.get("content_types", [])
                if not base_url or not content_types:
                    logger.error(f"Invalid configuration for DrupalLoader: Missing 'base_url' or 'content_types'. Entry: {entry}")
                    continue
                auth_type = str(entry.get("auth_type", "none"))
                username = str(entry.get("username", ""))
                password = str(entry.get("password", ""))
                token = str(entry.get("token", ""))
                text_fields = entry.get("text_fields")  # None → loader uses default
                page_size = int(entry.get("page_size", 50))
                language = str(entry.get("language", ""))
                documents = load_from_drupal(
                    base_url=str(base_url),
                    content_types=list(content_types),
                    loader_id=loader_id,
                    auth_type=auth_type,
                    username=username,
                    password=password,
                    token=token,
                    text_fields=text_fields,
                    page_size=page_size,
                    language=language,
                )
                logger.info(f"Loaded {len(documents)} documents from Drupal ({base_url}) using {loader_type}.")
            else:
                logger.error(f"Unknown loader type: {loader_type}")
                continue
            for doc in documents:
                doc.metadata["loader_id"] = loader_id
            all_documents.extend(documents)
        except Exception as e:
            logger.error(f"Error processing entry {entry}: {e}")

    return all_documents
