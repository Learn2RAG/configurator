"""
sharepoint_loader.py

Description:
This module handles loading documents from SharePoint.
It includes robust handling for App-Only Authentication (Client Credentials)
and Site-Specific contexts.

Author: Kyrill Meyer
Version: 0.0.6
Institution: IFDT
Creation Date: January 14, 2026
Last Modified Date: April 24, 2026
"""
import hashlib
import logging
import os
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Any, Union
from langchain_community.document_loaders import UnstructuredFileLoader, TextLoader, UnstructuredExcelLoader, PyPDFLoader
from langchain_core.documents import Document
from O365 import Account, FileSystemTokenBackend  # type: ignore
from ..globals import stop_loading

# initialize logger
logger = logging.getLogger("Learn2RAGImporter")

def _parse_file(file_path: Path, original_item: Any, loader_id: str = "N/A") -> List[Document]:
    """
    Parses file using the robust UnstructuredFileLoader.
    """
    docs: List[Document] = []
    # One hash for the entire file so all chunks share the same value,
    # enabling unambiguous deduplication by source URL in get_documents_by_loader_id.
    file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
    
    # Check file extension (lowercase)
    suffix = file_path.suffix.lower()

    try:
        loader: Union[UnstructuredExcelLoader, PyPDFLoader, UnstructuredFileLoader, TextLoader]
        
        # SPECIAL HANDLING FOR EXCEL
        if suffix in [".xlsx", ".xls"]:
            logger.info(f"Detected Excel file: {file_path.name} - using UnstructuredExcelLoader")
            # mode="single" combines the sheet into one document
            try:
                loader = UnstructuredExcelLoader(str(file_path), mode="single")
                docs = loader.load()
            except Exception as e_xls:
                # Fallback to Pandas if Unstructured fails (e.g. missing msoffcrypto)
                if "msoffcrypto" in str(e_xls) or "ImportError" in str(type(e_xls)):
                    logger.warning(f"Unstructured failed for Excel ({e_xls}). Attempting Pandas fallback...")
                    import pandas as pd
                    # Read all sheets; requires 'openpyxl' or 'xlrd'
                    dfs = pd.read_excel(str(file_path), sheet_name=None)
                    text_content = []
                    for sheet_name, df in dfs.items():
                        # Convert dataframe to string representation
                        text_content.append(f"--- Sheet: {sheet_name} ---\n{df.to_string(index=False)}")
                    
                    if not text_content:
                            text_content = ["Empty Excel file"]
                            
                    docs = [Document(
                        page_content="\n\n".join(text_content),
                        metadata={"source": original_item.web_url}
                    )]
                else:
                    raise e_xls

        # SPECIAL HANDLING FOR PDF (using PyPDFLoader to avoid unstructured dependencies)
        elif suffix == ".pdf":
            logger.info(f"Detected PDF file: {file_path.name} - using PyPDFLoader")
            loader = PyPDFLoader(str(file_path))
            _pages = loader.load()
            # Merge all pages into one Document so delta-import deduplication works on
            # a 1:1 source→document basis (same as all other loaders).
            _merged_pdf = Document(
                page_content="\n\n".join(p.page_content for p in _pages),
                metadata={**(_pages[0].metadata if _pages else {}), "total_pages": len(_pages)},
            )
            _merged_pdf.metadata.pop("page", None)
            _merged_pdf.metadata.pop("page_label", None)
            docs = [_merged_pdf]

        # SPECIAL HANDLING FOR IMAGES (Skip due to broken OCR environment)
        elif suffix in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            logger.warning(f"Image processing skipped for {file_path.name}: 'unstructured-inference' environment is incompatible.")
            # Return placeholder to avoid errors and TextLoader fallback attempts
            docs = [Document(
                page_content="[Image content skipped due to system dependency error]", 
                metadata={
                    "source": original_item.web_url, 
                    "name": original_item.name
                }
            )]

        else:
            # STANDARD: Smart Parsing with Unstructured for PDF, Word, HTML etc.
            # mode="single" combines the file content into one document
            loader = UnstructuredFileLoader(str(file_path),mode="single")
            docs = loader.load()

        # Enrich metadata for all extracted chunks/pages
        for doc in docs:
            # delete existing 'file_directory' metadata to avoid confusion since the local temp path is not relevant
            if "file_directory" in doc.metadata:
                del doc.metadata["file_directory"]

            # We keep existing metadata from Unstructured and add our own
            doc.metadata.update({
                "source": original_item.web_url,
                "document_id": original_item.object_id,
                "name": original_item.name,
                "created": str(original_item.created),
                "modified": str(original_item.modified),
                "loader": "SharePointLoader",
                "loader_id": loader_id,
                "content_hash": file_hash,
            })
        return docs

    except Exception as e_unstructured:
        logger.warning(f"Unstructured parsing failed for {file_path.name}: {e_unstructured}")
        
        # FALLBACK to simple TextLoader (for .txt, .csv, .py, .json, Logs, etc.)
        try:
             # Skip fallback for binary formats where text loader produces garbage
            if suffix in [".pdf", ".docx", ".doc", ".xlsx", ".xls", ".png", ".jpg"]:
                 raise Exception("Skipping TextLoader fallback for binary format.")

            logger.info(f"Attempting fallback to TextLoader for {file_path.name}...")
            loader = TextLoader(str(file_path), encoding="utf-8", autodetect_encoding=True)
            docs = loader.load()
        except Exception as e_text:
            logger.warning(f"TextLoader fallback also failed: {e_text}")
            # Fallback: Create empty document with error info so we know something is missing
            return [Document(
                page_content="", 
                metadata={
                    "source": original_item.web_url, 
                    "name": original_item.name, 
                    "error": str(e_unstructured)
                }
            )]
        
        # Enrich metadata for fallback docs as well
        for doc in docs:
            doc.metadata.update({
                "source": original_item.web_url,
                "document_id": original_item.object_id,
                "name": original_item.name,
                "created": str(original_item.created),
                "modified": str(original_item.modified),
                "loader": "SharePointLoader",
                "loader_id": loader_id,
                "content_hash": file_hash,
            })
        
        return docs
    
def reset_o365_token() -> None:
    """
    Reset the O365 token by deleting the token file.
    This forces a new authentication on the next load attempt.
    """
    token_path = Path.home() / ".credentials" / "o365_token.txt"
    if token_path.exists():
        token_path.unlink()
        logger.info("O365 token has been reset (file deleted).")
    else:
        logger.info("O365 token file does not exist; nothing to reset.")

def _authenticate_directly_with_o365(client_id: str, client_secret: str, tenant_id: str) -> None:
    """
    Helper function to authenticate directly using the O365 library.
    This ensures the tenant_id is correctly used for URL generation.
    """
    credentials = (client_id, client_secret)
    token_path_obj = Path.home() / ".credentials" / "o365_token.txt"
    token_backend = FileSystemTokenBackend(token_path=Path.home() / ".credentials", token_filename="o365_token.txt")
    
    account = Account(credentials, auth_flow_type='credentials', tenant_id=tenant_id, token_backend=token_backend)
    
    if token_path_obj.exists():
        logger.info(f"ℹ️ Found existing token file at: {token_path_obj} (Size: {token_path_obj.stat().st_size} bytes)")

    if not account.is_authenticated:
        logger.info("Attempting automatic background login (Client Credentials Flow)...")
        if account.authenticate():
            logger.info("✅ Automatic login successful!")

            if token_path_obj.exists():
                logger.info(f"💾 Token file successfully created/updated at: {token_path_obj}")
                logger.info(f"   New Size: {token_path_obj.stat().st_size} bytes")
            else:
                logger.warning("⚠️ Authentication succeeded, but the token file was NOT found on disk.")
        else:
            logger.error("❌ Automatic login failed. Check Azure 'Application Permissions' and Admin Consent.")
    else:
        logger.info("O365 Library reports: Already authenticated.")

def _list_available_drives(account: Account, search_term: Optional[str] = None) -> None:
    """
    Helper to list available drives/sites visible to the app.
    This helps in finding the correct Drive ID.
    """
    try:
        logger.info("Attempting to list available Drives (Document Libraries) for debugging...")
        sp = account.sharepoint() 
        
        # case 1: try root
        site = sp.get_root_site()
        if site:
            logger.info(f"Root Site found: {site.name} (ID: {site.object_id})")
            try:
                drives = site.list_document_libraries()
                for d in drives:
                    logger.info(f" -> Found Drive: Name='{d.name}', ID='{d.object_id}', WebUrl='{d.web_url}'")
            except Exception as e:
                logger.warning(f"Could not list libraries for root: {e}")
        else:
            logger.warning("Could not retrieve Root Site.")

        # case 2: search
        if search_term:
            logger.info(f"[2/2] Searching for sites containing '{search_term}'...")
            found_sites = sp.search_site(search_term)
            
            if found_sites:
                for site in found_sites:
                    logger.info(f"   ► Site match: '{site.name}' (ID: {site.object_id})")
                    try:
                        drives = site.list_document_libraries()
                        if drives:
                            for d in drives:
                                logger.info(f"      -> Drive found: Name='{d.name}', ID='{d.object_id}', Url='{d.web_url}'")
                        else:
                            logger.info("      (No Document Libraries found in this site)")
                    except Exception as drive_err:
                        logger.warning(f"      Could not list drives for this site: {drive_err}")
            else:
                logger.info(f"   No sites found matching '{search_term}'.")
        
        logger.info("---------------- DIAGNOSIS END ----------------")

    except Exception as e:
        logger.error(f"Error while listing available drives: {e}")

def _load_items_manual_traversal(drive: Any, folder_id: Optional[str] = None, recursive: bool = True, loader_id: str = "N/A", since: Optional[datetime] = None) -> List[Document]:
    """
    Internal helper to manually traverse and load items into Document objects.
    This bypasses LangChain's internal 'storage()' call which fails in App-Only context.
    It initiates downloads and uses Unstructured for parsing.
    """
    documents: List[Document] = []
    
    # 1. Determine entry point
    if folder_id:
        try:
            logger.info(f"Accessing folder by ID: {folder_id}")
            target = drive.get_item(folder_id)
            if not target.is_folder:
                 items = [target] 
            else:
                 items = target.get_items()
        except Exception as e:
            logger.error(f"Could not access folder_id {folder_id}: {e}")
            return []
    else:
        logger.info("Accessing root of the drive...")
        items = drive.get_items()

    # Create temporary directory for downloads
    temp_dir = Path(tempfile.mkdtemp())
    logger.info(f"Created temp directory for downloads: {temp_dir}")

    try:
        def _process_folder_items(item_list: Any) -> None:
            for item in item_list:
                if stop_loading: return

                if item.is_folder and recursive:
                    try:
                        # Recursive call for folders
                        _process_folder_items(item.get_items())
                    except Exception as e:
                        logger.warning(f"Skipping folder {item.name}: {e}")
                
                elif item.is_file:
                    try:
                        # Seit-Filter: Dateien überspringen, die vor `since` zuletzt geändert wurden
                        if since is not None:
                            item_modified = item.modified
                            if item_modified is not None:
                                # Sicherstellen, dass beide tz-aware sind
                                since_utc = since.astimezone(timezone.utc) if since.tzinfo else since.replace(tzinfo=timezone.utc)
                                item_modified_utc = item_modified.astimezone(timezone.utc) if item_modified.tzinfo else item_modified.replace(tzinfo=timezone.utc)
                                if item_modified_utc < since_utc:
                                    logger.debug(f"Skipping unchanged file (modified={item_modified_utc.isoformat()}): {item.name}")
                                    continue

                        # 2. Download file
                        download_success = item.download(to_path=temp_dir)
                        
                        if download_success:
                            local_file_path = temp_dir / item.name
                            
                            # 3. Parse file using Unstructured
                            if local_file_path.exists():
                                logger.info(f"Parsing with Unstructured: {item.name} ...")
                                parsed_docs = _parse_file(local_file_path, item, loader_id=loader_id)
                                documents.extend(parsed_docs)
                                
                                # Clean up immediately to save space
                                local_file_path.unlink()
                        else:
                            logger.warning(f"Download returned False for: {item.name}")

                    except Exception as e:
                        logger.error(f"Error processing file {item.name}: {e}")

        _process_folder_items(items)

    finally:
        # 4. Clean up the entire temp folder
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.debug("Cleaned up temp directory.")

    return documents


def _list_items_web_urls(drive: Any, folder_id: Optional[str] = None, recursive: bool = True) -> List[str]:
    """
    Traverse SharePoint files without downloading and collect their web URLs.

    Intended for deletion detection in the 2-pass delta import: compare the returned
    set against the paths stored in Qdrant to find files that have been removed.

    Args:
        drive (Any): Authenticated O365 Drive object.
        folder_id (Optional[str]): Object ID of the folder to start from.
                                    Uses the drive root when ``None``.
        recursive (bool): Whether to traverse sub-folders recursively (default ``True``).

    Returns:
        List[str]: Web URLs of all files found, e.g.
                   ``["https://tenant.sharepoint.com/.../file.pdf"]``.
    """
    urls: List[str] = []

    try:
        if folder_id:
            folder = drive.get_item(folder_id)
        else:
            folder = drive.get_root_folder()

        items = folder.get_items()
    except Exception as e:
        logger.error(f"_list_items_web_urls: error accessing folder: {e}")
        return urls

    for item in items:
        try:
            if item.is_folder and recursive:
                sub_urls = _list_items_web_urls(drive, folder_id=item.object_id, recursive=recursive)
                urls.extend(sub_urls)
            elif item.is_file:
                if item.web_url:
                    urls.append(item.web_url)
        except Exception as e:
            logger.warning(f"_list_items_web_urls: error processing item {getattr(item, 'name', '?')}: {e}")

    return urls


def get_all_sharepoint_document_ids(
    client_id: str,
    client_secret: str,
    document_library_id: str,
    folder_path: Optional[str] = None,
    folder_id: Optional[str] = None,
    recursive: bool = False,
    auth_with_token: bool = True,
    reset_token: bool = False,
    tenant_id: str = "common",
    site_id: Optional[str] = None,
) -> List[str]:
    """
    Retrieve the web URL for every file in a SharePoint document library without loading content.

    Intended for deletion detection in the 2-pass delta import: compare the returned
    set against the paths stored in Qdrant to find files that have been removed.

    Args:
        client_id (str): Azure AD application (client) ID.
        client_secret (str): Azure AD client secret.
        document_library_id (str): GUID of the SharePoint document library (Drive ID).
        folder_path (Optional[str]): Slash-separated path to a sub-folder relative to
                                      the library root, e.g. ``"Docs/Reports"``.
        folder_id (Optional[str]): Object ID of the entry-point folder; takes precedence
                                    over ``folder_path`` when both are provided.
        recursive (bool): Whether to traverse sub-folders recursively (default ``False``).
        auth_with_token (bool): Use cached O365 token when available (default ``True``).
        reset_token (bool): Delete the cached token before authenticating (default ``False``).
        tenant_id (str): Azure AD tenant ID or ``"common"`` (default).
        site_id (Optional[str]): SharePoint site ID; when provided, the library is looked
                                  up on that specific site rather than the root site.

    Returns:
        List[str]: Web URLs of all files found, e.g.
                   ``["https://tenant.sharepoint.com/.../file.pdf"]``.
    """
    if reset_token:
        reset_o365_token()

    token_path = Path.home() / ".credentials" / "o365_token.txt"
    token_backend = FileSystemTokenBackend(token_path=Path.home() / ".credentials", token_filename="o365_token.txt")

    if (not auth_with_token) or (not token_path.exists()):
        if tenant_id and tenant_id != "common":
            _authenticate_directly_with_o365(client_id, client_secret, tenant_id)
        else:
            logger.error("get_all_sharepoint_document_ids: No valid authentication method available.")
            return []

    account = Account((client_id, client_secret), token_backend=token_backend)

    if not account.is_authenticated:
        logger.error("get_all_sharepoint_document_ids: Authentication failed.")
        return []

    try:
        if site_id:
            sp = account.sharepoint()
            site = sp.get_site(site_id)
            storage = site.storage
        else:
            storage = account.storage()

        drive = storage.get_drive(document_library_id)
        if drive is None:
            logger.error(f"get_all_sharepoint_document_ids: Drive not found: {document_library_id}")
            return []

        # Optionaler Unterordner-Start
        effective_folder_id = folder_id
        if folder_path and not folder_id:
            root = drive.get_root_folder()
            for part in folder_path.strip("/").split("/"):
                found = None
                for child in root.get_items():
                    if child.is_folder and child.name == part:
                        found = child
                        break
                if found:
                    root = found
                else:
                    logger.warning(f"get_all_sharepoint_document_ids: folder part '{part}' not found")
                    return []
            effective_folder_id = root.object_id

        return _list_items_web_urls(drive, folder_id=effective_folder_id, recursive=recursive)

    except Exception as e:
        logger.error(f"get_all_sharepoint_document_ids: error: {e}")
        return []


def load_from_sharepoint(client_id: str, client_secret: str, document_library_id: str, 
                         folder_path: Optional[str] = None, folder_id: Optional[str] = None, 
                         object_ids: Optional[List[str]] = None, recursive: bool = False, 
                         auth_with_token: bool = True, load_extended_metadata: bool = True,
                         reset_token: bool = False, tenant_id: str = "common",
                         site_id: Optional[str] = None, loader_id: str = "N/A",
                         since: Optional[datetime] = None) -> List[Document]:
    """
    Load documents from SharePoint and set metadata.
    """

    documents: List[Document] = []

    # Reset token if requested
    if reset_token:
        reset_o365_token()

    token_path = Path.home() / ".credentials" / "o365_token.txt"
    token_backend = FileSystemTokenBackend(token_path=Path.home() / ".credentials", token_filename="o365_token.txt")
    
    # Check if we need to authenticate
    if (not auth_with_token) or (not token_path.exists()):
        if tenant_id and tenant_id != "common":
            logger.info("Performing O365 authentication using Tenant ID")
            _authenticate_directly_with_o365(client_id, client_secret, tenant_id)
            auth_with_token = True
        else:
            logger.warning("No Tenant ID provided or Tenant ID is 'common'.")
            auth_with_token = False
    else:
        logger.info(f"Using existing O365 token found at: {token_path}")
        logger.info("To force a new authentication (e.g., if token expired), set 'reset_token': 'True' in your configuration.")

    # --- Robust Logic Implementation ---
    # We construct the O365 Account object to manually fetch the Drive.
    # This avoids the "There isn't a Drive with id" error common in standard LangChain loader with App-Only Auth.
    
    try:
        credentials = (client_id, client_secret)
        account = Account(credentials, auth_flow_type='credentials', tenant_id=tenant_id, token_backend=token_backend)
        
        if not account.is_authenticated:
            # Fallback 
             logger.warning("Account seemingly not authenticated in main load logic. Check logs.")

        drive = None
        
        # 1. Try: specific Site ID (if provided)
        if site_id:
            logger.info(f"Connecting to specific Site ID: {site_id[:20]}...")
            site = account.sharepoint().get_site(site_id)
            drive = site.get_document_library(document_library_id)
        
        # 2. Try: Document Library directly from Root
        else:
            logger.info("Connecting to Root Site...")
            site = account.sharepoint().get_root_site()
            drive = site.get_document_library(document_library_id)
        
        if not drive:
            raise Exception(f"Drive/Library not found with ID {document_library_id}")

        logger.info(f"Successfully connected to Library: {drive.name}")
        
        # Load documents using internal helper function
        # Use folder_id if provided, otherwise use Root of the Drive
        loaded_docs = _load_items_manual_traversal(drive, folder_id=folder_id, recursive=recursive, loader_id=loader_id, since=since)
        
        logger.info(f"Found {len(loaded_docs)} documents.")

        for doc in loaded_docs:
            if stop_loading:
                logger.info("Loading process stopped by user.")
                break
            documents.append(doc)

    except Exception as e:
        logger.error(f"Error loading documents from SharePoint: {e}")
        
        # --- Diagnosis Helper ---
        err_str = str(e)
        if "There isn't a Drive with id" in err_str or "404 Client Error" in err_str or "Drive/Library not found" in err_str:
            logger.info("Diagnosis: The provided Document Library ID seems incorrect or not accessible.")
            
            if account and account.is_authenticated:
                 # Hardcode search term here or intelligently guess
                 # For production use, consider reading from config, but here we leave "frost" as an example
                 logger.info("Starting diagnosis scan...")
                 _list_available_drives(account, search_term="frost") 
        # ---------------------------------------

    return documents