"""
directory_loader.py

Description:
This module handles loading documents from directories.

Author: Kyrill Meyer
Version: 0.0.4
Institution: IFDT
Creation Date: June 10, 2025
Last Modified: March 17, 2026
"""
import hashlib
import logging
import os
from datetime import datetime
from typing import List, Union
from ..globals import stop_loading
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document

# supress pdfminer-Warnings
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# initialize logger
logger = logging.getLogger("Learn2RAGImporter")

def ensure_pandoc_available() -> None:
    """Checks if pandoc is available and downloads it with pypandoc if necessary."""
    try:
        import pypandoc # type: ignore
        try:
            version = pypandoc.get_pandoc_version()
            logger.info(f"Pandoc found, version: {version}")
        except OSError:
            logger.warning("Pandoc not found, trying to download with pypandoc, please be patient...")
            try:
                pypandoc.download_pandoc()
                logger.info("Pandoc successfully downloaded and installed.")
            except Exception as e:
                logger.error(f"Could not download pandoc: {e}")
    except ImportError:
        logger.warning("pypandoc is not installed. Install it with 'poetry add pypandoc' to manage pandoc automatically.")


def load_from_directory(path: str, recursive: Union[bool, str], silent_errors: bool = False, loader_id: str = "N/A") -> List[Document]:
    """
    Load documents from a directory and set metadata.

    Args:
        path (str): Path to the directory.
        recursive (bool): Whether to load documents recursively.

    Returns:
        list: List of documents with metadata.
    """

    # Check if pandoc is available (for RTF, DOCX etc.)
    ensure_pandoc_available()

    documents = []
    if isinstance(recursive, str):
        recursive = recursive.lower() == "true"


    text_loader_kwargs = {"autodetect_encoding": True, "detect_language_per_element": False}
    loader = DirectoryLoader(
        path,
        show_progress=True,
        loader_kwargs=text_loader_kwargs,
        recursive=recursive,
        silent_errors=silent_errors,
        glob=[
            "*.docx",
            "*.pptx",
            "*.xlsx",
            "*.pdf",
            "*.txt",
            "*.csv",
            "*.html",
            "*.md",
            "*.rtf",
            "*.odt",
            "*.epub",
        ]
    )
   
    #loader = DirectoryLoader(path, show_progress=True, loader_kwargs=text_loader_kwargs, recursive=recursive, glob=["*.csv", "*.docx", "*.eml", "*.epub", "*.html", "*.json", "*.md", "*.odt", "*.pdf", "*.ppt", "*.pptx", "*.rst", "*.rtf", "*.txt", "*.tsv", "*.cls", "*.xlsx", "*.xml"])
   
    #external dependencies 
    # doc - requires libreoffice
    # epub - requires pandoc

    #loader = DirectoryLoader(path, show_progress=True, silent_errors=True, recursive=False)
    try:
        loaded_documents = loader.load()
    except Exception as e:
        logger.error(f"Error loading documents from directory: {e}")
        return []
    
    for doc in loaded_documents:
        if stop_loading:
            logger.info("Loading process stopped by user.")
            break
        try:
            # generate a unique hash for the document content
            if isinstance(doc, Document):
                content_hash = hashlib.sha256(doc.page_content.encode('utf-8')).hexdigest()
                doc.metadata["content_hash"] = content_hash

                # get file metadata
                try:
                    stat_info = os.stat(doc.metadata["source"])
                    doc.metadata["file_size"] = stat_info.st_size
                    doc.metadata["file_mtime"] = datetime.fromtimestamp(stat_info.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                except OSError as e:
                    logger.warning(f"Could not get file metadata for {doc.metadata['source']}: {e}")
                    doc.metadata["file_size"] = "N/A"
                    doc.metadata["file_mtime"] = "N/A"
            else:
                logger.warning(f"Document is not of type Document: {type(doc)}. Skipping.")
                continue

            # set metadata for each document
            doc.metadata["source_path"] = path
  
            relative_path = os.path.relpath(doc.metadata["source"], path)
            doc.metadata["relative_path"] = relative_path
            doc.metadata["file_extension"] = doc.metadata.get("source", "").split(".")[-1]
            doc.metadata["process_date"] = datetime.now().strftime("%Y-%m-%d")
            doc.metadata["process_time"] = datetime.now().strftime("%H:%M:%S")
            doc.metadata["loader_type"] = "DirectoryLoader"
            doc.metadata["loader_id"] = loader_id

            documents.append(doc)
            logger.debug(f"Loaded file: {doc.metadata.get('source', 'Unknown')}")
        except Exception as e:
            error_str = str(e)
            doc_source = getattr(doc, 'metadata', {}).get('source', 'Unknown')
            if (
                "tesseract is not installed" in error_str
                or "pandoc is not installed" in error_str
                or "libreoffice" in error_str or "soffice" in error_str
            ):
                logger.warning(f"System dependency missing for file: {doc_source}\nError: {e}")
                continue
            else:
                logger.error(f"Error processing document {doc_source}: {e}")
                continue
    if documents:
        file_types = [doc.metadata.get("file_extension", "unknown") for doc in documents]
        type_count = {ext: file_types.count(ext) for ext in set(file_types)}
        logger.info(f"Loaded {len(documents)} documents from '{path}'. File types: {type_count}")

    else:
        logger.warning(f"No documents found in directory: {path}")
    return documents



