import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Literal, Mapping, Protocol, Sequence
from urllib.parse import unquote, urlsplit

from .models import IndexedDocument, IndexResponse

SCROLL_PAGE_SIZE = 256
MAX_CHUNKS = 10_000
PAYLOAD_FIELDS = [
    "source",
    "content_hash",
    "title",
    "uri",
    "loader_id",
    "document_id",
]

_MISSING_IDENTIFIERS = {"", "n/a", "none", "null", "unknown"}
_WINDOWS_PATH = re.compile(r"^[a-zA-Z]:[\\/]")


class QdrantPoint(Protocol):
    id: Any
    payload: Mapping[str, Any] | None


class QdrantReader(Protocol):
    def collection_exists(self, collection_name: str) -> bool: ...

    def scroll(self, **kwargs: Any) -> tuple[Sequence[QdrantPoint], Any]: ...


@dataclass
class _DocumentGroup:
    id: str
    name: str
    source_type: Literal["local", "url", "unknown"] | None
    chunk_count: int = 0


def inspect_index(
    client: QdrantReader,
    collection_name: str,
    *,
    page_size: int = SCROLL_PAGE_SIZE,
    max_chunks: int = MAX_CHUNKS,
) -> IndexResponse:
    """Read and summarize chunk payloads without exposing their raw metadata."""
    if not client.collection_exists(collection_name):
        return _empty_response(collection_name)

    groups: dict[str, _DocumentGroup] = {}
    offset: Any = None
    chunk_count = 0
    truncated = False

    while chunk_count < max_chunks:
        limit = min(page_size, max_chunks - chunk_count)
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_payload=PAYLOAD_FIELDS,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload if isinstance(point.payload, Mapping) else {}
            grouping_key = _grouping_key(payload, point.id)
            opaque_id = _opaque_id(grouping_key)
            group = groups.get(grouping_key)
            if group is None:
                name, source_type = _display_metadata(payload)
                group = _DocumentGroup(
                    id=opaque_id,
                    name=name,
                    source_type=source_type,
                )
                groups[grouping_key] = group
            group.chunk_count += 1
            chunk_count += 1

        if next_offset is None:
            break
        if not points:
            # Guard against a misbehaving backend returning a non-advancing page.
            truncated = True
            break
        if chunk_count >= max_chunks:
            truncated = True
            break
        offset = next_offset

    documents = sorted(
        (
            IndexedDocument(
                id=group.id,
                name=group.name,
                chunk_count=group.chunk_count,
                source_type=group.source_type,
            )
            for group in groups.values()
        ),
        key=lambda document: (document.name.casefold(), document.id),
    )
    status = "partial" if truncated else ("ready" if chunk_count else "empty")
    return IndexResponse(
        collection=collection_name,
        status=status,
        document_count=len(documents),
        chunk_count=chunk_count,
        documents=documents,
        truncated=truncated,
    )


def _empty_response(collection_name: str) -> IndexResponse:
    return IndexResponse(
        collection=collection_name,
        status="empty",
        document_count=0,
        chunk_count=0,
        documents=[],
    )


def _grouping_key(payload: Mapping[str, Any], point_id: Any) -> str:
    loader_id = _text(payload.get("loader_id"))
    document_id = _meaningful_identifier(payload.get("document_id"))
    source = _text(payload.get("source"))
    uri = _text(payload.get("uri"))
    content_hash = _meaningful_identifier(payload.get("content_hash"))
    title = _text(payload.get("title"))

    if document_id:
        identity: tuple[str, ...] = ("document_id", loader_id, document_id)
    elif source:
        identity = ("source", loader_id, source, content_hash)
    elif uri:
        identity = ("uri", loader_id, uri, content_hash)
    elif content_hash:
        identity = ("content_hash", loader_id, content_hash)
    elif title:
        identity = ("title", loader_id, title)
    else:
        identity = ("point", str(point_id))
    return json.dumps(identity, ensure_ascii=False, separators=(",", ":"))


def _opaque_id(grouping_key: str) -> str:
    return hashlib.sha256(grouping_key.encode("utf-8")).hexdigest()[:24]


def _display_metadata(
    payload: Mapping[str, Any],
) -> tuple[str, Literal["local", "url", "unknown"] | None]:
    source = _text(payload.get("source"))
    uri = _text(payload.get("uri"))
    title = _text(payload.get("title"))

    for value in (source, uri):
        if value:
            label, source_type = _safe_source_label(value)
            if label:
                return label, source_type

    if title:
        return _safe_plain_label(title), None
    return "Indexed document", None


def _safe_source_label(source: str) -> tuple[str, Literal["local", "url"]]:
    parsed = urlsplit(source)
    scheme = parsed.scheme.casefold()

    if scheme in {"http", "https"} and parsed.hostname:
        hostname = parsed.hostname
        leaf = PurePosixPath(unquote(parsed.path)).name
        label = f"{leaf} — {hostname}" if leaf else hostname
        return _trim_label(label), "url"

    if scheme == "file":
        path = unquote(parsed.path)
        return _path_name(path), "local"

    if _WINDOWS_PATH.match(source) or "\\" in source:
        return _trim_label(PureWindowsPath(source).name or "Indexed document"), "local"

    if source.startswith("/") or "/" in source or not scheme:
        return _path_name(source), "local"

    # For an unfamiliar URI scheme, expose only a terminal label, never its
    # authority, credentials, query string, or fragment.
    leaf = PurePosixPath(unquote(parsed.path)).name
    return _trim_label(leaf or "Remote document"), "url"


def _path_name(path: str) -> str:
    name = PurePosixPath(path).name
    return _trim_label(name or "Indexed document")


def _safe_plain_label(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme.casefold() in {"file", "http", "https"}:
        return _safe_source_label(value)[0]
    if _WINDOWS_PATH.match(value) or "\\" in value:
        return _trim_label(PureWindowsPath(value).name or "Indexed document")
    if value.startswith("/") or "/" in value:
        return _path_name(value)
    return _trim_label(value)


def _trim_label(value: str) -> str:
    clean = " ".join(value.split())
    return (clean[:157] + "...") if len(clean) > 160 else clean


def _meaningful_identifier(value: Any) -> str:
    normalized = _text(value)
    return "" if normalized.casefold() in _MISSING_IDENTIFIERS else normalized


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""
