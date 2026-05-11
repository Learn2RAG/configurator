"""
html_loader.py

Description:
This module handles loading documents from HTML sources.

Author: Kyrill Meyer
Institution: IFDT
Version: 0.0.6
Creation Date: July 28, 2025
Last Modified: May 05, 2026
"""

import hashlib
import os
import tempfile
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from datetime import datetime

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
from typing import List, Optional, Set
from urllib.parse import urlparse, urljoin
from ..globals import stop_loading
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.documents import Document
import logging
import requests


# initialize logger
logger = logging.getLogger("Learn2RAGImporter")


def _is_same_site(url: str, base_url: str) -> bool:
    """Check if url is on the same domain and under the same path prefix as base_url."""
    parsed_url = urlparse(url)
    parsed_base = urlparse(base_url)
    if parsed_url.scheme not in ('http', 'https'):
        return False
    if parsed_url.netloc != parsed_base.netloc:
        return False
    base_path = parsed_base.path.rstrip('/')
    return not base_path or parsed_url.path.startswith(base_path)


def load_html_content(url: str, depth: int = 0, visited: Optional[Set[str]] = None, loader_id: str = "N/A", _base_url: Optional[str] = None, skipped: Optional[Set[str]] = None) -> List[Document]:
    """
    Load HTML content from a URL and optionally follow links recursively.

    Args:
        url (str): The URL of the HTML page to load.
        depth (int): The depth of link traversal (default is 0).
            Use -1 to crawl the entire site (all links on the same domain
            and under the same path as the root URL).
        visited (set): A set of visited URLs to avoid duplicates.
        _base_url: Internal parameter to track the root URL for domain filtering.

    Returns:
        list: A list of LangChain Document objects with extracted content.
    """
    if visited is None:
        visited = set()

    if _base_url is None:
        _base_url = url

    if skipped is None:
        skipped = set()

    if url in visited:
        logger.info(f"Skipping already visited URL: {url}")
        return []

    visited.add(url)
    documents = []
    try:
        # Load the main page content
        response = requests.get(url)
        response.raise_for_status()

        # Save the HTML content to a temporary file for UnstructuredHTMLLoader
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", encoding="utf-8", delete=False) as f:
            f.write(response.text)
            temp_file = f.name

        # Use UnstructuredHTMLLoader to extract content
        loader = UnstructuredHTMLLoader(temp_file)
        page_documents = loader.load()
        # Extract meta properties using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        meta_tags = {meta.get("name", meta.get("property", "")): meta.get("content", "")
                     for meta in soup.find_all("meta") if meta.get("content")}
        
        for doc in page_documents:
            if stop_loading:
                logger.info("Loading process stopped by user.")
                break
            # Generate a unique hash for the document content
            if isinstance(doc, Document):
                content_hash = hashlib.sha256(doc.page_content.encode('utf-8')).hexdigest()
                doc.metadata["content_hash"] = content_hash
            else:
                logger.warning(f"Document is not of type Document: {type(doc)}. Skipping.")
                continue
            doc.metadata["source"] = url
            doc.metadata["process_date"] = datetime.now().strftime("%Y-%m-%d")
            doc.metadata["process_time"] = datetime.now().strftime("%H:%M:%S")
            doc.metadata["loader_type"] = "HTMLLoader"
            doc.metadata["meta_properties"] = meta_tags  
            doc.metadata["loader_id"] = loader_id
        documents.extend(page_documents)

        logger.info(f"Loaded content from {url}")

        # If depth > 0 or depth == -1, extract links and process them recursively
        if depth > 0 or depth == -1:
            soup = BeautifulSoup(response.text, "html.parser")
            links = [a["href"] for a in soup.find_all("a", href=True)]
            for link in links:
                if stop_loading:
                    break
                if isinstance(link, str):
                    # Resolve relative URLs
                    absolute_link = urljoin(url, link)
                    if depth == -1:
                        # Only follow links on the same domain and under the root path
                        if not _is_same_site(absolute_link, _base_url):
                            skipped.add(absolute_link)
                            continue
                        documents.extend(load_html_content(absolute_link, depth=-1, visited=visited, loader_id=loader_id, _base_url=_base_url, skipped=skipped))
                    else:
                        documents.extend(load_html_content(absolute_link, depth=depth - 1, visited=visited, loader_id=loader_id, _base_url=_base_url, skipped=skipped))

    except Exception as e:
        logger.error(f"Error loading content from {url}: {e}")

    finally:
        # Delete the temporary file
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)
            logger.debug(f"Temporary file {temp_file} deleted.")

        if documents:
            meta_count = sum(len(doc.metadata.get("meta_properties", {})) for doc in documents)
            logger.info(
                f"Loaded {len(documents)} documents from '{url}'. "
                f"Total meta tags: {meta_count}. "
                f"Depth: {depth}."
            )
        else:
            logger.warning(f"No documents found for URL: {url}")

    return documents
