"""
mediawiki_loader.py

Description:
This module handles loading documents from MediaWiki via the API.
Supports authentication (none, Basic Auth, Bearer token, MediaWiki API Login),
pagination, and incremental loading based on recent changes.

Author: Kyrill Meyer
Institution: IFDT
Version: 0.0.3
Creation Date: August 12, 2026
Last Modified: August 31, 2026
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, TYPE_CHECKING

import requests
from langchain_core.documents import Document

from ..globals import stop_loading
from ..loaders.errors import LoaderAccessError

if TYPE_CHECKING:
    from ..utils.progress import ImportProgress

logger = logging.getLogger("Learn2RAGImporter")


class MediaWikiSourceUnavailable(LoaderAccessError):
    """Raised when MediaWiki cannot be reached or returns unusable responses."""


def _to_iso_utc(dt: datetime) -> str:
    utc_dt = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_namespaces(namespaces: Optional[List[Any]]) -> List[int]:
    if not namespaces:
        return [0]

    normalized: List[int] = []
    for raw_ns in namespaces:
        try:
            normalized.append(int(raw_ns))
        except (TypeError, ValueError):
            logger.warning("MediaWikiLoader: ignoring invalid namespace value '%s'", raw_ns)
    return normalized or [0]


def _build_session(auth_type: str, username: str, password: str, token: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    if auth_type == "basic":
        session.auth = (username, password)
    elif auth_type == "token":
        session.headers.update({"Authorization": f"Bearer {token}"})

    return session


def _resolve_api_url(base_url: str, session: requests.Session) -> str:
    """
    Automatically discover the correct path to MediaWiki's api.php.
    Tests common endpoint patterns and verifies JSON response from MediaWiki.
    """
    cleaned_url = base_url.rstrip("/")

    # Direct path provided in base_url
    if cleaned_url.endswith("api.php"):
        return cleaned_url

    # Candidates for auto-detection
    candidates = [
        f"{cleaned_url}/api.php",      
        f"{cleaned_url}/w/api.php",    
        f"{cleaned_url}/wiki/api.php", 
    ]

    for candidate in candidates:
        try:
            response = session.get(
                candidate,
                params={"action": "query", "format": "json"},
                timeout=5,
            )
            content_type = response.headers.get("Content-Type", "").lower()

            # Must be HTTP 200 AND return JSON
            if response.status_code == 200 and "application/json" in content_type:
                payload = response.json()
                if isinstance(payload, dict):
                    # Valid MediaWiki API if 'query' is present OR if MediaWiki returned an API error (e.g. readapidenied)
                    if "query" in payload or "error" in payload or "servedby" in payload:
                        logger.info("MediaWikiLoader: auto-detected API endpoint at '%s'", candidate)
                        return candidate
        except (requests.exceptions.RequestException, ValueError):
            continue

    raise MediaWikiSourceUnavailable(
        f"Could not auto-detect a valid MediaWiki API at '{base_url}'. "
        f"Please specify the exact path to api.php in base_url."
    )


def _authenticate_mediawiki(session: requests.Session, api_url: str, username: str, password: str) -> None:
    """
    Authenticates the session via MediaWiki API login token.
    Required for wikis that require read permissions (prevents readapidenied).
    """
    if not username or not password:
        return

    try:
        # Step 1: Request Login Token
        token_payload = _api_get(session, api_url, {
            "action": "query",
            "meta": "tokens",
            "type": "login",
            "format": "json",
            "formatversion": 2,
        })
        login_token = token_payload.get("query", {}).get("tokens", {}).get("logintoken")

        if not login_token:
            logger.warning("MediaWikiLoader: Could not obtain login token from API")
            return

        # Step 2: Perform API Client Login
        login_resp = session.post(
            api_url,
            data={
                "action": "login",
                "lgname": username,
                "lgpassword": password,
                "lgtoken": login_token,
                "format": "json",
                "formatversion": 2,
            },
            timeout=15,
        ).json()

        result = login_resp.get("login", {}).get("result")
        if result == "Success":
            logger.info("MediaWikiLoader: Successfully authenticated as '%s' via API", username)
        else:
            reason = login_resp.get("login", {}).get("reason", "Unknown error")
            logger.warning("MediaWikiLoader: API login failed for '%s': %s", username, reason)

    except Exception as exc:
        logger.warning("MediaWikiLoader: Authentication attempt raised exception: %s", exc)


def _api_get(session: requests.Session, api_url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response = session.get(api_url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except (requests.exceptions.RequestException, ValueError) as exc:
        raise MediaWikiSourceUnavailable(f"MediaWiki request failed: {exc}") from exc

    if not isinstance(payload, dict):
        raise MediaWikiSourceUnavailable("MediaWiki response is not a JSON object")

    if payload.get("error"):
        raise MediaWikiSourceUnavailable(f"MediaWiki API error: {payload.get('error')}")

    return payload


def _iter_all_page_ids(
    api_url: str,
    session: requests.Session,
    namespaces: List[int],
    page_size: int,
) -> Set[int]:
    page_ids: Set[int] = set()

    for namespace in namespaces:
        ap_continue = ""

        while True:
            if stop_loading:
                return page_ids

            params: Dict[str, Any] = {
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "list": "allpages",
                "apnamespace": namespace,
                "aplimit": page_size,
            }
            if ap_continue:
                params["apcontinue"] = ap_continue

            payload = _api_get(session, api_url, params)
            query = payload.get("query", {})
            allpages = query.get("allpages", []) if isinstance(query, dict) else []
            if not isinstance(allpages, list):
                raise MediaWikiSourceUnavailable("MediaWiki allpages payload malformed")

            for item in allpages:
                if not isinstance(item, dict):
                    continue
                page_id = item.get("pageid")
                if isinstance(page_id, int) and page_id > 0:
                    page_ids.add(page_id)

            cont = payload.get("continue", {})
            next_continue = cont.get("apcontinue") if isinstance(cont, dict) else None
            if not next_continue:
                break
            ap_continue = str(next_continue)

    return page_ids


def _iter_recent_changed_page_ids(
    api_url: str,
    session: requests.Session,
    namespaces: List[int],
    since: datetime,
    page_size: int,
) -> Set[int]:
    page_ids: Set[int] = set()
    since_iso = _to_iso_utc(since)

    for namespace in namespaces:
        rc_continue = ""

        while True:
            if stop_loading:
                return page_ids

            params: Dict[str, Any] = {
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "list": "recentchanges",
                "rcnamespace": namespace,
                "rcprop": "ids|title|timestamp",
                "rcshow": "!bot",
                "rctype": "new|edit",
                "rcdir": "newer",
                "rcstart": since_iso,
                "rclimit": page_size,
            }
            if rc_continue:
                params["rccontinue"] = rc_continue

            payload = _api_get(session, api_url, params)
            query = payload.get("query", {})
            recentchanges = query.get("recentchanges", []) if isinstance(query, dict) else []
            if not isinstance(recentchanges, list):
                raise MediaWikiSourceUnavailable("MediaWiki recentchanges payload malformed")

            for item in recentchanges:
                if not isinstance(item, dict):
                    continue
                page_id = item.get("pageid")
                if isinstance(page_id, int) and page_id > 0:
                    page_ids.add(page_id)

            cont = payload.get("continue", {})
            next_continue = cont.get("rccontinue") if isinstance(cont, dict) else None
            if not next_continue:
                break
            rc_continue = str(next_continue)

    return page_ids


def _chunked(values: Iterable[int], chunk_size: int) -> Iterable[List[int]]:
    chunk: List[int] = []
    for value in values:
        chunk.append(value)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _fetch_pages(
    api_url: str,
    base_url: str,
    session: requests.Session,
    page_ids: Set[int],
    loader_id: str,
    progress: Optional["ImportProgress"] = None,  
) -> List[Document]:
    documents: List[Document] = []
    total_count = len(page_ids)
    processed_count = 0

    for id_chunk in _chunked(sorted(page_ids), 50):
        if stop_loading:
            break

        params: Dict[str, Any] = {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "prop": "extracts|revisions",
            "explaintext": 1,
            "exlimit": "max",
            "exsectionformat": "plain",
            "rvprop": "timestamp",
            "rvslots": "main",
            "pageids": "|".join(str(i) for i in id_chunk),
            "redirects": 1,
        }

        payload = _api_get(session, api_url, params)
        query = payload.get("query", {})
        pages = query.get("pages", []) if isinstance(query, dict) else []
        if not isinstance(pages, list):
            raise MediaWikiSourceUnavailable("MediaWiki pages payload malformed")

        for page in pages:
            if not isinstance(page, dict):
                continue

            processed_count += 1
            page_id = page.get("pageid")
            if not isinstance(page_id, int) or page_id <= 0:
                continue

            title = str(page.get("title", "")).strip()
            namespace = page.get("ns")
            extract = str(page.get("extract", "")).strip()

            revisions = page.get("revisions", [])
            updated = ""
            if isinstance(revisions, list) and revisions and isinstance(revisions[0], dict):
                updated = str(revisions[0].get("timestamp", ""))

            page_content = extract if extract else title
            if not page_content.strip():
                continue

            # exact progress to console/logs for each page, useful for large imports
            if progress is not None:
                progress.emit(
                    "Phase 2/4 Load",
                    f"Loading page {processed_count}/{total_count}: {title}",
                    processed=processed_count,
                    total=total_count,
                    source=base_url,
                )

            source = f"{base_url.rstrip('/')}/?curid={page_id}"
            content_hash = hashlib.sha256(page_content.encode("utf-8")).hexdigest()

            documents.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "source": source,
                        "loader_id": loader_id,
                        "loader": "MediaWikiLoader",
                        "content_hash": content_hash,
                        "page_id": page_id,
                        "title": title,
                        "namespace": namespace,
                        "updated": updated,
                    },
                )
            )

    return documents


def load_from_mediawiki(
    base_url: str,
    loader_id: str = "",
    auth_type: str = "none",
    username: str = "",
    password: str = "",
    token: str = "",
    namespaces: Optional[List[Any]] = None,
    page_size: int = 50,
    since: Optional[datetime] = None,
    progress: Optional["ImportProgress"] = None,
) -> List[Document]:
    """Load MediaWiki pages and return one document per page."""
    safe_page_size = max(1, min(int(page_size), 500))
    safe_namespaces = _normalize_namespaces(namespaces)

    session = _build_session(auth_type, username, password, token)

    # Resolve API endpoint dynamically
    api_url = _resolve_api_url(base_url, session)

    # Perform API Login if credentials are provided
    if auth_type == "basic" and username and password:
        _authenticate_mediawiki(session, api_url, username, password)

    logger.info("MediaWikiLoader: collecting page IDs from %s", api_url)
    if since is None:
        page_ids = _iter_all_page_ids(
            api_url=api_url,
            session=session,
            namespaces=safe_namespaces,
            page_size=safe_page_size,
        )
    else:
        page_ids = _iter_recent_changed_page_ids(
            api_url=api_url,
            session=session,
            namespaces=safe_namespaces,
            since=since,
            page_size=safe_page_size,
        )

    if progress is not None:
        progress.emit(
            "Phase 2/4 Load",
            f"MediaWiki IDs collected | {len(page_ids)} pages",
            processed=len(page_ids),
            source=base_url,
        )

    documents = _fetch_pages(
        api_url=api_url,
        base_url=base_url,
        session=session,
        page_ids=page_ids,
        loader_id=loader_id,
        progress=progress,
    )

    logger.info("MediaWikiLoader: loaded %s document(s)", len(documents))
    return documents


def get_all_mediawiki_document_ids(
    base_url: str,
    auth_type: str = "none",
    username: str = "",
    password: str = "",
    token: str = "",
    namespaces: Optional[List[Any]] = None,
    page_size: int = 100,
) -> List[str]:
    """Return source IDs (curid URLs) for all pages in selected namespaces."""
    safe_page_size = max(1, min(int(page_size), 500))
    safe_namespaces = _normalize_namespaces(namespaces)

    session = _build_session(auth_type, username, password, token)
    api_url = _resolve_api_url(base_url, session)

    if auth_type == "basic" and username and password:
        _authenticate_mediawiki(session, api_url, username, password)

    page_ids = _iter_all_page_ids(
        api_url=api_url,
        session=session,
        namespaces=safe_namespaces,
        page_size=safe_page_size,
    )

    sources = [f"{base_url.rstrip('/')}/?curid={page_id}" for page_id in sorted(page_ids)]
    logger.info("get_all_mediawiki_document_ids: total %s document IDs", len(sources))
    return sources