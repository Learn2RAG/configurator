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
import os
import platform
import stat
from datetime import datetime
from ..globals import stop_loading
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
    loader = DirectoryLoader(path, show_progress=True, loader_kwargs=text_loader_kwargs, recursive=recursive, glob=["*.csv", "*.doc", "*.docx", "*.eml", "*.epub", "*.html", "*.json", "*.md", "*.odt", "*.pdf", "*.ppt", "*.pptx", "*.rst", "*.rtf", "*.txt", "*.tsv", "*.cls", "*.xlsx", "*.xml"])
   
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

                # get file metadata
                try:
                    stat_info = os.stat(doc.metadata["source"])
                    doc.metadata["file_permissions"] = stat.filemode(stat_info.st_mode)
                    doc.metadata["file_size"] = stat_info.st_size
                    doc.metadata["file_mtime"] = datetime.fromtimestamp(stat_info.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    # owner and group
                    if platform.system() == 'Windows':
                        try:
                            import win32api
                            import win32security
                            # Get owner SID
                            sd = win32security.GetFileSecurity(doc.metadata["source"], win32security.OWNER_SECURITY_INFORMATION)
                            owner_sid = sd.GetSecurityDescriptorOwner()
                            owner_name, domain, type = win32security.LookupAccountSid(None, owner_sid)
                            doc.metadata["file_owner"] = f"{domain}\\{owner_name}"
                            # For group
                            sd_group = win32security.GetFileSecurity(doc.metadata["source"], win32security.GROUP_SECURITY_INFORMATION)
                            group_sid = sd_group.GetSecurityDescriptorGroup()
                            group_name, group_domain, type = win32security.LookupAccountSid(None, group_sid)
                            doc.metadata["file_group"] = f"{group_domain}\\{group_name}"
                            # For permissions, get DACL summary
                            sd_dacl = win32security.GetFileSecurity(doc.metadata["source"], win32security.DACL_SECURITY_INFORMATION)
                            dacl = sd_dacl.GetSecurityDescriptorDacl()
                            if dacl:
                                permissions = []
                                for ace_idx in range(dacl.GetAceCount()):
                                    ace = dacl.GetAce(ace_idx)
                                    trustee_name, trustee_domain, type = win32security.LookupAccountSid(None, ace[2])
                                    permissions.append(f"{trustee_domain}\\{trustee_name}: {ace[1]}")
                                doc.metadata["file_permissions_detailed"] = "; ".join(permissions)
                            else:
                                doc.metadata["file_permissions_detailed"] = "No DACL"
                        except ImportError:
                            doc.metadata["file_owner"] = "N/A (install pywin32 for Windows details)"
                            doc.metadata["file_group"] = "N/A (install pywin32 for Windows details)"
                            doc.metadata["file_permissions_detailed"] = "N/A (install pywin32 for Windows details)"
                        except Exception as e:
                            logger.warning(f"Error getting Windows file security info: {e}")
                            doc.metadata["file_owner"] = "Error"
                            doc.metadata["file_group"] = "Error"
                            doc.metadata["file_permissions_detailed"] = "Error"
                    else:
                        try:
                            import pwd
                            doc.metadata["file_owner"] = pwd.getpwuid(stat_info.st_uid).pw_name
                        except ImportError:
                            doc.metadata["file_owner"] = "N/A"
                        try:
                            import grp
                            doc.metadata["file_group"] = grp.getgrgid(stat_info.st_gid).gr_name
                        except ImportError:
                            doc.metadata["file_group"] = "N/A"
                        doc.metadata["file_permissions_detailed"] = doc.metadata["file_permissions"]  # Unix permissions are already detailed enough
                except OSError as e:
                    logger.warning(f"Could not get file metadata for {doc.metadata['source']}: {e}")
                    doc.metadata["file_permissions"] = "N/A"
                    doc.metadata["file_size"] = "N/A"
                    doc.metadata["file_mtime"] = "N/A"
                    doc.metadata["file_owner"] = "N/A"
                    doc.metadata["file_group"] = "N/A"
                    doc.metadata["file_permissions_detailed"] = "N/A"

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



