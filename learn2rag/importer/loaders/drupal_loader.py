"""
drupal_loader.py

Description:
This module handles loading documents from a Drupal CMS instance via the JSON:API.
Supports authentication (none, Basic Auth, Bearer token) and automatic pagination.

Drupal requirements:
- Drupal 8/9/10/11 with the JSON:API module enabled (core module, check via the "EXTEND" page in Drupal admin).
- Content types must be accessible via /jsonapi/node/<content_type>

Author: Kyrill Meyer
Institution: IFDT
Version: 0.0.2
Creation Date: March 17, 2026
Last Modified: April 24, 2026
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup
from langchain_core.documents import Document
import requests
from ..globals import stop_loading

logger = logging.getLogger("Learn2RAGImporter")

# JSON:API index paths to probe (in order) when auto-discovering endpoints
_JSONAPI_INDEX_CANDIDATES = ["/jsonapi", "/en/jsonapi", "/de/jsonapi"]


def _discover_endpoint_map(base_url: str, session: requests.Session) -> Dict[str, str]:
    """
    Fetch the Drupal JSON:API index and return a mapping of
    resource type (e.g. 'node--article') to its full endpoint URL.
    Probes several common paths to handle language-prefixed Drupal installations.
    """
    base = base_url.rstrip("/")
    for path in _JSONAPI_INDEX_CANDIDATES:
        url = base + path
        try:
            response = session.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                links: Dict[str, Any] = data.get("links", {})
                endpoint_map: Dict[str, str] = {}
                for key, value in links.items():
                    if isinstance(value, dict):
                        href = value.get("href", "")
                    elif isinstance(value, str):
                        href = value
                    else:
                        continue
                    if href:
                        endpoint_map[key] = href
                if endpoint_map:
                    logger.info(f"DrupalLoader: discovered {len(endpoint_map)} endpoint(s) from {url}")
                    return endpoint_map
        except requests.exceptions.RequestException:
            continue
    logger.warning("DrupalLoader: could not auto-discover JSON:API endpoints, falling back to /jsonapi path")
    return {}


def _build_session(auth_type: str, username: str, password: str, token: str) -> requests.Session:
    """Build a requests Session with the configured authentication."""
    session = requests.Session()
    session.headers.update({"Accept": "application/vnd.api+json"})
    if auth_type == "basic":
        session.auth = (username, password)
    elif auth_type == "token":
        session.headers.update({"Authorization": f"Bearer {token}"})
    return session


def _html_to_text(html: str) -> str:
    """Strip HTML tags and return plain text."""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def _extract_page_content(attributes: Dict[str, Any], text_fields: List[str]) -> str:
    """
    Extract and concatenate text content from the specified fields.
    Handles both plain strings and Drupal processed-text objects ({"value": ..., "processed": ...}).
    """
    parts: List[str] = []
    for field in text_fields:
        value = attributes.get(field)
        if value is None:
            continue
        if isinstance(value, dict):
            # Drupal body field: {"value": "...", "processed": "<p>...</p>", "summary": "..."}
            processed = value.get("processed") or value.get("value") or ""
            parts.append(_html_to_text(str(processed)))
        elif isinstance(value, str):
            parts.append(_html_to_text(value))
    return "\n\n".join(filter(None, parts))


def _autodetect_text_content(attributes: Dict[str, Any]) -> str:
    """
    Fallback: scan all attributes and concatenate any field that contains readable text.
    Covers plain text fields, Drupal formatted-text dicts, and list-of-dicts (paragraphs).
    """
    skip_keys = {"drupal_internal__nid", "drupal_internal__vid", "langcode", "status",
                 "created", "changed", "promote", "sticky", "revision_timestamp",
                 "revision_log", "revision_translation_affected", "default_langcode",
                 "path", "content_translation_source", "content_translation_outdated"}
    parts: List[str] = []
    for key, value in attributes.items():
        if key in skip_keys or key == "title":
            continue
        if isinstance(value, dict) and ("value" in value or "processed" in value):
            text = value.get("processed") or value.get("value") or ""
            cleaned = _html_to_text(str(text))
            if cleaned:
                parts.append(cleaned)
        elif isinstance(value, str) and len(value) > 30:
            parts.append(_html_to_text(value))
        elif isinstance(value, list):
            for sub in value:
                if isinstance(sub, dict):
                    text = sub.get("processed") or sub.get("value") or ""
                    cleaned = _html_to_text(str(text))
                    if cleaned:
                        parts.append(cleaned)
    return "\n\n".join(filter(None, parts))


def load_from_drupal(
    base_url: str,
    content_types: List[str],
    loader_id: str = "",
    auth_type: str = "none",
    username: str = "",
    password: str = "",
    token: str = "",
    text_fields: Optional[List[str]] = None,
    page_size: int = 50,
    language: str = "",
    since: Optional[datetime] = None,
) -> List[Document]:
    """
    Load documents from a Drupal instance via the JSON:API.

    Args:
        base_url (str): Base URL of the Drupal site, e.g. "https://example.com".
        content_types (list): List of machine names of content types to load,
                              e.g. ["article", "page"].
        loader_id (str): Unique identifier for this loader (used in metadata).
        auth_type (str): Authentication type: "none", "basic", or "token".
        username (str): Username for Basic Auth.
        password (str): Password for Basic Auth.
        token (str): Bearer token for token auth.
        text_fields (list): Fields to use as page_content. Defaults to ["title", "body"].
        page_size (int): Number of items per API page request (default 50).
        language (str): Optional language filter, e.g. "de" or "en". Uses Drupal's
                        content-language filtering via the Accept-Language header.

    Returns:
        list: List of LangChain Document objects.
    """
    if text_fields is None:
        text_fields = ["title", "field_body", "body"]  # fallback covers both naming conventions

    session = _build_session(auth_type, username, password, token)
    if language:
        session.headers.update({"Accept-Language": language})

    # Auto-discover real endpoint URLs (handles language-prefixed Drupal installations)
    endpoint_map = _discover_endpoint_map(base_url, session)

    all_documents: List[Document] = []

    for content_type in content_types:
        if stop_loading:
            logger.info("Loading process stopped by user.")
            break

        resource_key = f"node--{content_type}"
        if resource_key in endpoint_map:
            endpoint = endpoint_map[resource_key]
            logger.debug(f"DrupalLoader: using discovered endpoint for '{content_type}': {endpoint}")
        else:
            endpoint = f"{base_url.rstrip('/')}/jsonapi/node/{content_type}"
            logger.debug(f"DrupalLoader: no discovered endpoint for '{content_type}', using fallback: {endpoint}")

        # Do NOT use sparse fieldsets (fields[...]) – unknown field names cause 404/400.
        # Fetch all attributes and extract only the configured text_fields locally.
        params: Dict[str, Any] = {
            "page[limit]": page_size,
            "page[offset]": 0,
        }

        # Timestamp filter: only documents changed on or after `since`
        if since is not None:
            # Ensure the timestamp is timezone-aware and formatted as ISO 8601 for JSON:API
            since_utc = since.astimezone(timezone.utc) if since.tzinfo else since.replace(tzinfo=timezone.utc)
            params["filter[changed-filter][condition][path]"] = "changed"
            params["filter[changed-filter][condition][operator]"] = ">="
            params["filter[changed-filter][condition][value]"] = since_utc.isoformat()
            logger.info(f"DrupalLoader: applying since-filter >= {since_utc.isoformat()} for '{content_type}'")

        page_count = 0
        next_url: Optional[str] = endpoint

        logger.info(f"DrupalLoader: loading content type '{content_type}' from {endpoint}")

        while next_url:
            if stop_loading:
                logger.info("Loading process stopped by user.")
                break

            try:
                if page_count == 0:
                    response = session.get(next_url, params=params, timeout=30)
                else:
                    # next_url from JSON:API links already includes all query params
                    response = session.get(next_url, timeout=30)

                response.raise_for_status()
                data = response.json()
            except requests.exceptions.HTTPError as e:
                logger.error(f"DrupalLoader: HTTP error for content type '{content_type}': {e}")
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"DrupalLoader: Request failed for content type '{content_type}': {e}")
                break

            items = data.get("data", [])
            if not items:
                break

            first_item_logged = False
            for item in items:
                if stop_loading:
                    break

                attributes: Dict[str, Any] = item.get("attributes", {})

                # Log available attribute keys once per content type to help with field config
                if not first_item_logged:
                    available_fields = [k for k, v in attributes.items() if v is not None]
                    logger.info(
                        f"DrupalLoader: available non-null attribute fields for '{content_type}': "
                        f"{available_fields}"
                    )
                    first_item_logged = True

                node_id: str = item.get("id", "")
                drupal_id: str = str(attributes.get("drupal_internal__nid", node_id))
                title: str = str(attributes.get("title", ""))
                created: str = str(attributes.get("created", ""))
                changed: str = str(attributes.get("changed", ""))
                langcode: str = str(attributes.get("langcode", language))
                status: bool = bool(attributes.get("status", True))

                page_content = _extract_page_content(attributes, text_fields)
                if not page_content:
                    # Configured text_fields yielded nothing – auto-detect all readable text fields
                    page_content = _autodetect_text_content(attributes)
                    if page_content:
                        logger.debug(
                            f"DrupalLoader: node {drupal_id} – configured text_fields had no content, "
                            f"used auto-detected fields instead."
                        )
                if not page_content:
                    # Last resort fallback: use title only
                    page_content = title

                if not page_content.strip():
                    logger.debug(f"DrupalLoader: skipping empty node {drupal_id} ({title})")
                    continue

                content_hash = hashlib.sha256(page_content.encode("utf-8")).hexdigest()
                source_url = f"{base_url.rstrip('/')}/node/{drupal_id}"

                doc = Document(
                    page_content=page_content,
                    metadata={
                        "source": source_url,
                        "loader_id": loader_id,
                        "content_type": content_type,
                        "node_id": drupal_id,
                        "title": title,
                        "created": created,
                        "changed": changed,
                        "langcode": langcode,
                        "status": status,
                        "content_hash": content_hash,
                    },
                )
                all_documents.append(doc)

            # Follow pagination via JSON:API "links.next"
            links = data.get("links", {})
            next_link = links.get("next")
            if next_link and isinstance(next_link, dict):
                next_url = next_link.get("href")
            elif next_link and isinstance(next_link, str):
                next_url = next_link
            else:
                next_url = None

            page_count += 1
            logger.debug(f"DrupalLoader: fetched page {page_count} for '{content_type}', {len(items)} items")

        logger.info(f"DrupalLoader: loaded {len(all_documents)} documents total so far after content type '{content_type}'")

    logger.info(f"DrupalLoader: finished. Total documents loaded: {len(all_documents)}")
    return all_documents


def get_all_drupal_document_ids(
    base_url: str,
    content_types: List[str],
    auth_type: str = "none",
    username: str = "",
    password: str = "",
    token: str = "",
    page_size: int = 100,
    language: str = "",
) -> List[str]:
    """
    Retrieve the source URL for every current node in Drupal without loading content.

    Intended for deletion detection in the 2-pass delta import: compare the returned
    set against the paths stored in Qdrant to find nodes that have been removed.

    Args:
        base_url (str): Base URL of the Drupal site, e.g. ``"https://example.com"``.
        content_types (list): List of content type machine names, e.g. ``["article", "page"]``.
        auth_type (str): Authentication type: ``"none"``, ``"basic"``, or ``"token"``.
        username (str): Username for Basic Auth.
        password (str): Password for Basic Auth.
        token (str): Bearer token for token auth.
        page_size (int): Items per API page request (default 100; larger than load_from_drupal
                         default because only IDs are fetched, saving bandwidth).
        language (str): Optional language filter passed via ``Accept-Language`` header.

    Returns:
        List[str]: Source URLs of all current nodes, e.g. ``["https://example.com/node/42"]``.
    """
    session = _build_session(auth_type, username, password, token)
    if language:
        session.headers.update({"Accept-Language": language})

    endpoint_map = _discover_endpoint_map(base_url, session)
    all_ids: List[str] = []

    for content_type in content_types:
        if stop_loading:
            break

        resource_key = f"node--{content_type}"
        if resource_key in endpoint_map:
            endpoint = endpoint_map[resource_key]
        else:
            endpoint = f"{base_url.rstrip('/')}/jsonapi/node/{content_type}"

        # Request only nid + langcode (no content) to minimise bandwidth
        params: Dict[str, Any] = {
            "fields[node--{}]".format(content_type): "drupal_internal__nid,langcode",
            "page[limit]": page_size,
            "page[offset]": 0,
        }

        next_url: Optional[str] = endpoint
        page_count = 0

        while next_url:
            try:
                if page_count == 0:
                    response = session.get(next_url, params=params, timeout=30)
                else:
                    response = session.get(next_url, timeout=30)
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException as e:
                logger.error(f"get_all_drupal_document_ids: Request failed for '{content_type}': {e}")
                break

            items = data.get("data", [])
            if not items:
                break

            for item in items:
                attributes: Dict[str, Any] = item.get("attributes", {})
                node_id: str = item.get("id", "")
                drupal_id: str = str(attributes.get("drupal_internal__nid", node_id))
                source_url = f"{base_url.rstrip('/')}/node/{drupal_id}"
                all_ids.append(source_url)

            links = data.get("links", {})
            next_link = links.get("next")
            if next_link and isinstance(next_link, dict):
                next_url = next_link.get("href")
            elif next_link and isinstance(next_link, str):
                next_url = next_link
            else:
                next_url = None

            page_count += 1

        logger.info(f"get_all_drupal_document_ids: found {len(all_ids)} IDs so far after content type '{content_type}'")

    logger.info(f"get_all_drupal_document_ids: total {len(all_ids)} document IDs")
    return all_ids
