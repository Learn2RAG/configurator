"""
process_loaders.py

Description:
This module processes configuration entries and delegates loading to specific loader functions.

Author: Kyrill Meyer
Institution: IFDT
Version: 0.0.8
Creation Date: June 10, 2025
Last Modified: May 18, 2026
"""

import hashlib
import logging
from typing import Dict, List, Any, TYPE_CHECKING
if TYPE_CHECKING:
    from learn2rag.importer.utils.import_state import ImportState
from learn2rag.pipeline.ingestion import index
from learn2rag.pipeline.store import get_documents, delete_documents, update_documents
from ..globals import stop_loading
from langchain_core.documents import Document
from .directory_loader import load_from_directory
from .csv_loader import load_from_csv
from .html_loader import load_html_content
from .sharepoint_loader import load_from_sharepoint, get_all_sharepoint_document_ids
from .drupal_loader import load_from_drupal, get_all_drupal_document_ids
from .jira_loader import load_from_jira, get_all_jira_document_ids

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
                path = entry.get("path")
                if not path:
                    logger.error("Missing 'path' for 'CSVLoader' in configuration entry.")
                    continue
                documents = load_from_csv(path)
                # CSVLoader does not set loader_id, source or content_hash — populate them here
                for doc in documents:
                    doc.metadata["loader_id"] = loader_id
                    doc.metadata["source"] = path
                    doc.metadata["content_hash"] = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
                logger.info(f"Loaded {len(documents)} documents from {path} using {loader_type}.")
            elif loader_type == "HTMLLoader":
                url = entry.get("url")
                depth = entry.get("depth", 0)
                if not url or not isinstance(depth, int) or depth < -1:
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
            elif loader_type == "JiraLoader":
                base_url = str(entry.get("base_url", ""))
                auth_type = str(entry.get("auth_type", "basic"))
                username = str(entry.get("username", ""))
                password = str(entry.get("password", ""))
                token = str(entry.get("token", ""))
                jql = str(entry.get("jql", ""))
                projects = entry.get("projects", [])
                issue_types = entry.get("issue_types", [])
                page_size = int(entry.get("page_size", 50))
                include_comments = entry.get("include_comments", False)
                if isinstance(include_comments, str):
                    include_comments = include_comments.lower() == "true"

                if not base_url:
                    logger.error(f"Invalid configuration for JiraLoader: Missing 'base_url'. Entry: {entry}")
                    continue

                documents = load_from_jira(
                    base_url=base_url,
                    loader_id=loader_id,
                    auth_type=auth_type,
                    username=username,
                    password=password,
                    token=token,
                    jql=jql,
                    projects=list(projects) if isinstance(projects, list) else [],
                    issue_types=list(issue_types) if isinstance(issue_types, list) else [],
                    page_size=page_size,
                    include_comments=bool(include_comments),
                )
                logger.info(f"Loaded {len(documents)} documents from Jira ({base_url}) using {loader_type}.")
            else:
                logger.error(f"Unknown loader type: {loader_type}")
                continue
            for doc in documents:
                doc.metadata["loader_id"] = loader_id
            all_documents.extend(documents)
        except Exception as e:
            logger.error(f"Error processing entry {entry}: {e}")

    return all_documents


def process_delta_imports(
    config_entries: List[Dict[str, Any]],
    user_config: Dict[str, Any],
    opt_config: Dict[str, Any],
    import_state: "ImportState",
) -> None:
    """
    Perform a delta import for all configured loaders.

    Dispatches between two strategies:

    - **Intelligent loaders** (DrupalLoader, SharepointLoader): 2-pass approach —
      fetch all current document IDs to detect deletions, then load only changed
      documents using a server-side timestamp filter.
    - **Normal loaders** (DirectoryLoader, HTMLLoader, CSVLoader): full load followed
      by content-hash comparison to detect additions, changes, and deletions.

    The import timestamp is only persisted on successful completion, so a failed run
    will be retried in full on the next call.

    Args:
        config_entries (List[Dict[str, Any]]): Loader configuration entries from the
                                               importer config file.
        user_config (Dict[str, Any]): User configuration dict (must contain
                                      ``collection_name``).
        opt_config (Dict[str, Any]): Optimisation configuration dict.
        import_state (ImportState): ImportState instance for timestamp management.
    """
    from datetime import datetime, timezone

    for entry in config_entries:
        if stop_loading:
            logger.info("Delta import stopped by user.")
            break

        loader_type = entry.get("loader_type")
        loader_id = entry.get("loader_id") or ""

        if not loader_type or not loader_id:
            logger.error(f"process_delta_imports: entry missing loader_type or loader_id: {entry}")
            continue

        try:
            last_import_time = import_state.get_last_import_time(loader_id)
            import_start = datetime.now(timezone.utc)
            import_state.record_import_start(loader_id, import_start)

            # Retrieve existing Qdrant documents for this loader as {source: content_hash} map
            existing_map: Dict[str, str] = get_documents(loader_id, user_config, opt_config)
            is_initial = len(existing_map) == 0

            logger.info(
                f"Delta import '{loader_id}': is_initial={is_initial}, "
                f"last_import_time={last_import_time}, existing_docs={len(existing_map)}"
            )

            # ----------------------------------------------------------------
            # INTELLIGENT LOADERS: Drupal / SharePoint
            # 2-pass: (1) fetch all current IDs → detect deletions,
            #          (2) load only changed documents via timestamp filter
            # ----------------------------------------------------------------
            if loader_type == "DrupalLoader":
                base_url = entry.get("base_url")
                if not base_url:
                    logger.error(f"DrupalLoader '{loader_id}': missing 'base_url'")
                    continue
                base_url = str(base_url)
                content_types = entry.get("content_types", [])
                auth_type = str(entry.get("auth_type", "none"))
                username = str(entry.get("username", ""))
                password = str(entry.get("password", ""))
                token = str(entry.get("token", ""))
                text_fields = entry.get("text_fields")
                page_size = int(entry.get("page_size", 50))
                language = str(entry.get("language", ""))

                if is_initial or last_import_time is None:
                    # No prior state: fall back to full load + hash comparison
                    logger.info(f"Drupal '{loader_id}': full load (initial={is_initial})")
                    all_docs = load_from_drupal(
                        base_url=base_url, content_types=content_types, loader_id=loader_id,
                        auth_type=auth_type, username=username, password=password, token=token,
                        text_fields=text_fields, page_size=page_size, language=language,
                    )
                    if is_initial:
                        index(all_docs, user_config, opt_config)
                    else:
                        # Hash comparison: replace changed, remove deleted
                        _delta_by_source(all_docs, existing_map, loader_id, user_config, opt_config)
                else:
                    # 2-pass delta
                    logger.info(f"Drupal '{loader_id}': 2-pass delta since {last_import_time.isoformat()}")
                    # Pass 1: fetch all current IDs to detect deleted documents
                    current_ids = set(get_all_drupal_document_ids(
                        base_url=base_url, content_types=content_types,
                        auth_type=auth_type, username=username, password=password, token=token,
                        page_size=page_size, language=language,
                    ))
                    deleted_paths = [p for p in existing_map if p not in current_ids]
                    if deleted_paths:
                        logger.info(f"Drupal '{loader_id}': deleting {len(deleted_paths)} removed documents")
                        delete_documents(loader_id, deleted_paths, user_config, opt_config)

                    # Pass 2: load and index changed documents
                    changed_docs = load_from_drupal(
                        base_url=base_url, content_types=content_types, loader_id=loader_id,
                        auth_type=auth_type, username=username, password=password, token=token,
                        text_fields=text_fields, page_size=page_size, language=language,
                        since=last_import_time,
                    )
                    sources_to_delete = [doc.metadata.get("source", "") for doc in changed_docs]
                    if sources_to_delete:
                        delete_documents(loader_id, sources_to_delete, user_config, opt_config)
                    index(changed_docs, user_config, opt_config)
                    logger.info(f"Drupal '{loader_id}': {len(deleted_paths)} deleted, {len(changed_docs)} updated")

            elif loader_type == "SharepointLoader":
                client_id = entry.get("client_id", "")
                client_secret = entry.get("client_secret", "")
                document_library_id = entry.get("document_library_id", "")
                folder_path = entry.get("folder_path")
                folder_id = entry.get("folder_id")
                recursive = entry.get("recursive", False)
                auth_with_token = entry.get("auth_with_token", True)
                reset_token = entry.get("reset_token", False)
                tenant_id = entry.get("tenant_id", "common")
                site_id = entry.get("site_id")

                if is_initial or last_import_time is None:
                    logger.info(f"SharePoint '{loader_id}': full load (initial={is_initial})")
                    all_docs = load_from_sharepoint(
                        client_id=client_id, client_secret=client_secret,
                        document_library_id=document_library_id, folder_path=folder_path,
                        folder_id=folder_id, recursive=recursive, auth_with_token=auth_with_token,
                        reset_token=reset_token, tenant_id=tenant_id, site_id=site_id,
                        loader_id=loader_id,
                    )
                    if is_initial:
                        index(all_docs, user_config, opt_config)
                    else:
                        _delta_by_source(all_docs, existing_map, loader_id, user_config, opt_config)
                else:
                    logger.info(f"SharePoint '{loader_id}': 2-pass delta since {last_import_time.isoformat()}")
                    # Pass 1: fetch all current URLs to detect deleted documents
                    current_ids = set(get_all_sharepoint_document_ids(
                        client_id=client_id, client_secret=client_secret,
                        document_library_id=document_library_id, folder_path=folder_path,
                        folder_id=folder_id, recursive=recursive, auth_with_token=auth_with_token,
                        reset_token=reset_token, tenant_id=tenant_id, site_id=site_id,
                    ))
                    deleted_paths = [p for p in existing_map if p not in current_ids]
                    if deleted_paths:
                        logger.info(f"SharePoint '{loader_id}': deleting {len(deleted_paths)} removed documents")
                        delete_documents(loader_id, deleted_paths, user_config, opt_config)

                    # Pass 2: load and index changed documents
                    changed_docs = load_from_sharepoint(
                        client_id=client_id, client_secret=client_secret,
                        document_library_id=document_library_id, folder_path=folder_path,
                        folder_id=folder_id, recursive=recursive, auth_with_token=auth_with_token,
                        reset_token=reset_token, tenant_id=tenant_id, site_id=site_id,
                        loader_id=loader_id, since=last_import_time,
                    )
                    sources_to_delete = [doc.metadata.get("source", "") for doc in changed_docs]
                    if sources_to_delete:
                        delete_documents(loader_id, sources_to_delete, user_config, opt_config)
                    index(changed_docs, user_config, opt_config)
                    logger.info(f"SharePoint '{loader_id}': {len(deleted_paths)} deleted, {len(changed_docs)} updated")

            elif loader_type == "JiraLoader":
                base_url = str(entry.get("base_url", ""))
                auth_type = str(entry.get("auth_type", "basic"))
                username = str(entry.get("username", ""))
                password = str(entry.get("password", ""))
                token = str(entry.get("token", ""))
                jql = str(entry.get("jql", ""))
                projects = entry.get("projects", [])
                issue_types = entry.get("issue_types", [])
                page_size = int(entry.get("page_size", 50))
                include_comments = entry.get("include_comments", False)
                if isinstance(include_comments, str):
                    include_comments = include_comments.lower() == "true"

                safe_projects = list(projects) if isinstance(projects, list) else []
                safe_issue_types = list(issue_types) if isinstance(issue_types, list) else []

                if is_initial or last_import_time is None:
                    logger.info(f"Jira '{loader_id}': full load (initial={is_initial})")
                    all_docs = load_from_jira(
                        base_url=base_url,
                        loader_id=loader_id,
                        auth_type=auth_type,
                        username=username,
                        password=password,
                        token=token,
                        jql=jql,
                        projects=safe_projects,
                        issue_types=safe_issue_types,
                        page_size=page_size,
                        include_comments=bool(include_comments),
                    )
                    if is_initial:
                        index(all_docs, user_config, opt_config)
                    else:
                        _delta_by_source(all_docs, existing_map, loader_id, user_config, opt_config)
                else:
                    logger.info(f"Jira '{loader_id}': 2-pass delta since {last_import_time.isoformat()}")
                    current_ids = set(get_all_jira_document_ids(
                        base_url=base_url,
                        auth_type=auth_type,
                        username=username,
                        password=password,
                        token=token,
                        jql=jql,
                        projects=safe_projects,
                        issue_types=safe_issue_types,
                        page_size=page_size,
                    ))
                    deleted_paths = [p for p in existing_map if p not in current_ids]
                    if deleted_paths:
                        logger.info(f"Jira '{loader_id}': deleting {len(deleted_paths)} removed documents")
                        delete_documents(loader_id, deleted_paths, user_config, opt_config)

                    changed_docs = load_from_jira(
                        base_url=base_url,
                        loader_id=loader_id,
                        auth_type=auth_type,
                        username=username,
                        password=password,
                        token=token,
                        jql=jql,
                        projects=safe_projects,
                        issue_types=safe_issue_types,
                        page_size=page_size,
                        include_comments=bool(include_comments),
                        since=last_import_time,
                    )
                    sources_to_delete = [doc.metadata.get("source", "") for doc in changed_docs]
                    if sources_to_delete:
                        delete_documents(loader_id, sources_to_delete, user_config, opt_config)
                    index(changed_docs, user_config, opt_config)
                    logger.info(f"Jira '{loader_id}': {len(deleted_paths)} deleted, {len(changed_docs)} updated")

            # ----------------------------------------------------------------
            # NORMAL LOADERS: Directory / HTML / CSV — hash comparison
            # ----------------------------------------------------------------
            elif loader_type == "DirectoryLoader":
                path = entry.get("path")
                if not path:
                    logger.error(f"DirectoryLoader '{loader_id}': missing 'path'")
                    continue
                all_docs = load_from_directory(
                    path,
                    recursive=entry.get("recursive", False),
                    silent_errors=entry.get("silent_errors", True),
                    loader_id=loader_id,
                )
                if is_initial:
                    index(all_docs, user_config, opt_config)
                else:
                    _delta_by_source(all_docs, existing_map, loader_id, user_config, opt_config)

            elif loader_type == "HTMLLoader":
                url = entry.get("url")
                depth = entry.get("depth", 0)
                if not url:
                    logger.error(f"HTMLLoader '{loader_id}': missing 'url'")
                    continue
                all_docs = load_html_content(url, depth=depth, loader_id=loader_id)
                if is_initial:
                    index(all_docs, user_config, opt_config)
                else:
                    _delta_by_source(all_docs, existing_map, loader_id, user_config, opt_config)

            elif loader_type == "CSVLoader":
                path = entry.get("path")
                if not path:
                    logger.error(f"CSVLoader '{loader_id}': missing 'path'")
                    continue
                all_docs = load_from_csv(path)
                for doc in all_docs:
                    doc.metadata["loader_id"] = loader_id
                    doc.metadata["source"] = path
                    doc.metadata["content_hash"] = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
                if is_initial:
                    index(all_docs, user_config, opt_config)
                else:
                    _delta_by_source(all_docs, existing_map, loader_id, user_config, opt_config)

            else:
                logger.error(f"process_delta_imports: unknown loader_type '{loader_type}' for loader_id '{loader_id}'")
                continue

            import_state.save_success(loader_id)
            logger.info(f"Delta import '{loader_id}': completed successfully.")

        except Exception as e:
            logger.error(f"process_delta_imports: error processing loader '{loader_id}': {e}", exc_info=True)


def _delta_by_source(
    all_docs: List[Document],
    existing_map: Dict[str, str],
    loader_id: str,
    user_config: Dict[str, Any],
    opt_config: Dict[str, Any],
) -> None:
    """
    Hash-based delta import for normal loaders (DirectoryLoader, HTMLLoader, CSVLoader).

    Each loader guarantees exactly one Document per source (PDF pages merged, HTML
    elements merged), so ``content_hash`` on that Document is the raw-file hash and
    directly comparable to the value stored in Qdrant by ``get_documents()``.

    - Deletes Qdrant chunks (bulk) for sources that no longer exist in the new load.
    - Calls ``update_documents`` for sources whose content_hash has changed or that
      are entirely new (update_documents handles delete-then-reindex internally).
    - Leaves unchanged sources untouched.

    Args:
        all_docs (List[Document]): All documents returned by the loader for this run.
        existing_map (Dict[str, str]): Mapping of ``{source: content_hash}``.
        loader_id (str): Unique loader identifier.
        user_config (Dict[str, Any]): User configuration dict (must contain
                                      ``collection_name``).
        opt_config (Dict[str, Any]): Optimisation configuration dict.
    """
    # Group freshly loaded documents by source (1 source = 1 Document after loader merging)
    new_docs_by_source: Dict[str, List[Document]] = {}
    for doc in all_docs:
        source: str = doc.metadata.get("source", "")
        new_docs_by_source.setdefault(source, []).append(doc)

    # All loaders guarantee exactly one Document per source (PDF pages merged,
    # HTML elements merged), so content_hash on docs[0] is directly comparable
    # to the value returned by get_documents() from Qdrant.
    new_hash_by_source: Dict[str, str] = {
        source: docs[0].metadata.get("content_hash", "")
        for source, docs in new_docs_by_source.items()
    }

    # Bulk-delete sources that are no longer present in the fresh load
    deleted_sources: List[str] = [s for s in existing_map if s not in new_docs_by_source]
    if deleted_sources:
        delete_documents(loader_id, deleted_sources, user_config, opt_config)

    # Update changed and new sources via update_documents (delete-then-reindex)
    changed_docs: List[Document] = []
    for source, docs in new_docs_by_source.items():
        if existing_map.get(source) != new_hash_by_source[source]:
            changed_docs.extend(docs)

    if changed_docs:
        update_documents(loader_id, changed_docs, user_config, opt_config)

    changed_source_count = len(set(d.metadata.get("source", "") for d in changed_docs))
    logger.info(
        f"_delta_by_source '{loader_id}': {len(deleted_sources)} deleted, "
        f"{len(changed_docs)} chunks re-indexed from {changed_source_count} changed sources"
    )
