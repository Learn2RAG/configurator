"""
csv_loader.py

Description:
This module handles loading documents from CSV files.

Author: Kyrill Meyer
Institution: IFDT
Version: 0.0.1
Creation Date: June 10, 2025
"""
from datetime import datetime
from langchain_community.document_loaders import CSVLoader

def load_from_csv(path):
    """
    Load documents from a CSV file and set metadata.

    Args:
        path (str): Path to the CSV file.

    Returns:
        list: List of documents with metadata.
    """

    loader = CSVLoader(file_path=path)
    documents = loader.load()
    if not documents:
        raise ValueError(f"No document found in CSV file: {path}")

    # Set metadata for each document
    for doc in documents:
        doc.metadata["source"] = path
        doc.metadata["loader_type"] = "CSVLoader"
        doc.metadata["file_name"] = doc.metadata.get("source", "").split("/")[-1] if "source" in doc.metadata else ""
        doc.metadata["file_extension"] = doc.metadata.get("file_name", "").split(".")[-1] if "file_name" in doc.metadata else ""
        doc.metadata["file_size"] = doc.metadata.get("file_size", "unknown")
        doc.metadata["file_type"] = doc.metadata.get("file_extension", "unknown")
        doc.metadata["file_creation_date"] = doc.metadata.get("file_creation_date", "unknown")
        doc.metadata["file_modification_date"] = doc.metadata.get("file_modification_date", "unknown")
        doc.metadata["file_access_date"] = doc.metadata.get("file_access_date", "unknown")
        doc.metadata["file_owner"] = doc.metadata.get("file_owner", "unknown")
        doc.metadata["file_permissions"] = doc.metadata.get("file_permissions", "unknown")
        doc.metadata["file_hash"] = doc.metadata.get("file_hash", "unknown")
        doc.metadata["file_encoding"] = doc.metadata.get("file_encoding", "unknown")
        doc.metadata["file_language"] = doc.metadata.get("file_language", "unknown")
        doc.metadata["file_description"] = doc.metadata.get("file_description", "unknown")
        doc.metadata["file_keywords"] = doc.metadata.get("file_keywords", [])
        doc.metadata["file_tags"] = doc.metadata.get("file_tags", [])
        doc.metadata["file_category"] = doc.metadata.get("file_category", "unknown")
         # Add process date and time
        doc.metadata["process_date"] = datetime.now().strftime("%Y-%m-%d")
        doc.metadata["process_time"] = datetime.now().strftime("%H:%M:%S")


    return documents