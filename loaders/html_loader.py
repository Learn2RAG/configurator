"""
html_loader.py

Description:
This module handles loading documents from HTML sources.

Author: Kyrill Meyer
Institution: IFDT
Version: 0.0.3
Creation Date: July 28, 2025
"""

import hashlib
from bs4 import BeautifulSoup
from datetime import datetime
from globals import stop_loading
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.documents import Document
import logging
import os
import requests


# initialize logger
logger = logging.getLogger("Learn2RAGImporter")

def load_html_content(url, depth=0, visited=None) -> list[Document]:
    """
    Load HTML content from a URL and optionally follow links recursively.

    Args:
        url (str): The URL of the HTML page to load.
        depth (int): The depth of link traversal (default is 0).
        visited (set): A set of visited URLs to avoid duplicates.

    Returns:
        list: A list of LangChain Document objects with extracted content.
    """
    if visited is None:
        visited = set()

    if url in visited:
        logger.info(f"Skipping already visited URL: {url}")
        return []

    visited.add(url)
    documents = []
    temp_file = "temp.html"

    try:
        # Load the main page content
        response = requests.get(url)
        response.raise_for_status()

        # Save the HTML content to a temporary file for UnstructuredHTMLLoader
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(response.text)

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
        documents.extend(page_documents)

        logger.info(f"Loaded content from {url}")

        # If depth > 0, extract links and process them recursively
        if depth > 0:
            soup = BeautifulSoup(response.text, "html.parser")
            links = [a["href"] for a in soup.find_all("a", href=True)]
            for link in links:
                # Resolve relative URLs
                absolute_link = requests.compat.urljoin(url, link)
                documents.extend(load_html_content(absolute_link, depth=depth - 1, visited=visited))

    except Exception as e:
        logger.error(f"Error loading content from {url}: {e}")

    finally:
        # Delete the temporary file
        if os.path.exists(temp_file):
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