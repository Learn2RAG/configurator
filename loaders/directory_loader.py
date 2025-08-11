"""
directory_loader.py

Description:
This module handles loading documents from directories.

Author: Kyrill Meyer
Version: 0.0.2
Institution: IFDT
Creation Date: June 10, 2025
"""
import hashlib
import logging
from datetime import datetime
from globals import stop_loading
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document

# supress pdfminer-Warnings
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# initialize logger
logger = logging.getLogger("Learn2RAGImporter")


def load_from_directory(path, recursive) -> list[Document]:
    """
    Load documents from a directory and set metadata.

    Args:
        path (str): Path to the directory.
        recursive (bool): Whether to load documents recursively.

    Returns:
        list: List of documents with metadata.
    """

    documents = []
    if isinstance(recursive, str):
        recursive = recursive.lower() == "true"

    text_loader_kwargs = {"autodetect_encoding": True, "detect_language_per_element": False}
    loader = DirectoryLoader(path, show_progress=True, silent_errors=True, loader_kwargs=text_loader_kwargs, recursive=recursive, glob=["*.csv", "*.doc", "*.docx", "*.eml", "*.epub", "*.html", "*.json", "*.md", "*.odt", "*.pdf", "*.ppt", "*.pptx", "*.rst", "*.rtf", "*.txt", "*.tsv", "*.cls", "*.xlsx", "*.xml"])
   
    #loader = DirectoryLoader(path, show_progress=True, silent_errors=True, recursive=False)
    try:
        loaded_documents = loader.load()
        for doc in loaded_documents:
            if stop_loading:
                logger.info("Loading process stopped by user.")
                break
            # generate a unique hash for the document content
            if isinstance(doc, Document):
                content_hash = hashlib.sha256(doc.page_content.encode('utf-8')).hexdigest()
                doc.metadata["content_hash"] = content_hash
            else:
                logger.warning(f"Document is not of type Document: {type(doc)}. Skipping.")
                continue

            # set metadata for each document
            doc.metadata["source_path"] = path
            doc.metadata["file_extension"] = doc.metadata.get("source", "").split(".")[-1]
            doc.metadata["process_date"] = datetime.now().strftime("%Y-%m-%d")
            doc.metadata["process_time"] = datetime.now().strftime("%H:%M:%S")
            doc.metadata["loader_type"] = "DirectoryLoader"
            documents.append(doc)
            logger.debug(f"Loaded file: {doc.metadata.get('source', 'Unknown')}")

    except Exception as e:
        logger.error(f"Error loading documents from directory: {e}")
    if documents:
        file_types = [doc.metadata.get("file_extension", "unknown") for doc in documents]
        type_count = {ext: file_types.count(ext) for ext in set(file_types)}
        logger.info(f"Loaded {len(documents)} documents from '{path}'. File types: {type_count}")
    else:
        logger.warning(f"No documents found in directory: {path}")
        raise ValueError(f"No documents found in directory: {path}")

    return documents



