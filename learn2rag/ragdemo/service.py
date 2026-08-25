"""Safe orchestration and projection boundary for the public RAG Demo.

Internal Qdrant and LangChain objects are converted here into explicit public
models so filesystem paths, credentials, and implementation metadata do not
cross into browser-facing responses.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Literal, Mapping, Protocol, Sequence
from urllib.parse import unquote, urlsplit

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from qdrant_client.conversions import common_types as qdrant_types

from learn2rag.pipeline.generate import build_prompt_messages, invoke_prompt_messages

from .models import (
    IndexedDocument,
    IndexResponse,
    QueryPromptMessage,
    QueryPromptResponse,
    QueryResponse,
    QuerySearchResponse,
    QuerySearchResult,
)

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
# Authorization identity is owned by the server; the public request may only
# supply a question and cannot select a user or other pipeline internals.
DEMO_USER = "ragdemo"


class QdrantReader(Protocol):
    """Minimum read-only Qdrant capability needed for public index inspection."""

    def collection_exists(self, collection_name: str) -> bool: ...

    def scroll(
        self,
        collection_name: str,
        *,
        limit: int,
        offset: qdrant_types.PointId | None,
        with_payload: Sequence[str],
        with_vectors: bool,
    ) -> tuple[list[qdrant_types.Record], qdrant_types.PointId | None]: ...


class QuerySearchOperator(Protocol):
    """Authorized retrieval boundary used by the demo query flow."""

    async def __call__(self, inputs: Any, prov: Any = None) -> Any: ...


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
    """Summarize the configured Qdrant collection for the public index view.

    Qdrant is the authoritative live index; the bounded scroll avoids an
    unbounded request and projects only safe document metadata to the browser.
    """
    if not client.collection_exists(collection_name):
        return _empty_response(collection_name)

    groups: dict[str, _DocumentGroup] = {}
    offset: qdrant_types.PointId | None = None
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
    index_status: Literal["ready", "empty", "partial"] = (
        "partial" if truncated else ("ready" if chunk_count else "empty")
    )
    return IndexResponse(
        collection=collection_name,
        status=index_status,
        document_count=len(documents),
        chunk_count=chunk_count,
        documents=documents,
        truncated=truncated,
    )


async def execute_query(
    search_operator: QuerySearchOperator,
    question: str,
    runtime_config: Mapping[str, Any],
) -> QueryResponse:
    """Run one authorized retrieval and one generation for a public query.

    Search, prompt inspection, and generation share the same ordered points.
    Sources are sanitized while building the messages, before model invocation,
    so the exact application-level prompt returned to the browser is also safe.
    """
    search_output = await search_operator(
        inputs={"question": question, "user": DEMO_USER}
    )
    if not isinstance(search_output, Mapping):
        raise TypeError("Search output must be a mapping")

    documents = search_output.get("documents")
    points = getattr(documents, "points", documents)
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        raise TypeError("Search documents must be a sequence")

    mode, label, technical_label, score_label = _search_metadata(runtime_config)
    results = [
        _public_search_result(point, rank)
        for rank, point in enumerate(points, start=1)
    ]
    messages = build_prompt_messages(
        question,
        list(points),
        dict(runtime_config),
        source_transform=_safe_prompt_source,
    )
    public_prompt = _public_prompt(messages)
    answer = invoke_prompt_messages(messages)
    if not isinstance(answer, str):
        raise TypeError("Model answer must be text")

    public_answer = _sanitize_answer(answer, points)
    return QueryResponse(
        question=question,
        answer=public_answer,
        search=QuerySearchResponse(
            mode=mode,
            label=label,
            technical_label=technical_label,
            score_label=score_label,
            results=results,
        ),
        prompt=public_prompt,
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
    """Build a conservative identity without assuming incomplete records match.

    A meaningful document ID is preferred. Source/URI fallbacks include the
    content hash so different indexed versions are not silently merged.
    """
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
    """Derive a stable public ID without revealing Qdrant or source identifiers."""
    return hashlib.sha256(grouping_key.encode("utf-8")).hexdigest()[:24]


def _public_search_result(point: Any, rank: int) -> QuerySearchResult:
    """Allowlist one retrieved point into the browser-facing Search schema.

    Loader IDs, hashes, and raw sources remain internal. The numeric value is
    exposed as a retrieval/similarity score, never as probability or confidence.
    """
    payload_value = getattr(point, "payload", None)
    payload = payload_value if isinstance(payload_value, Mapping) else {}
    content = payload.get("content")
    score = getattr(point, "score", None)
    if not isinstance(content, str):
        raise TypeError("Retrieved chunk content must be text")
    if not isinstance(score, (int, float)):
        raise TypeError("Retrieved chunk score must be numeric")

    source, _ = _display_metadata(payload)
    return QuerySearchResult(
        rank=rank,
        id=_chunk_display_id(point, payload),
        source=source,
        score=float(score),
        content=content,
    )


def _safe_prompt_source(source: Any) -> str:
    """Sanitize a source before it enters the demo's model-bound messages."""
    if not isinstance(source, str):
        raise TypeError("Retrieved chunk source must be text")
    return _safe_source_label(source)[0]


def _public_prompt(messages: Sequence[BaseMessage]) -> QueryPromptResponse:
    """Serialize exact application-level messages using only role and content.

    Provider wire formats, internal chat templates, tokenization, hidden
    instructions, and arbitrary LangChain metadata are intentionally excluded.
    """
    public_messages: list[QueryPromptMessage] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            role: Literal["system", "human"] = "system"
        elif isinstance(message, HumanMessage):
            role = "human"
        else:
            raise TypeError("Unsupported application-level prompt message type")
        if not isinstance(message.content, str):
            raise TypeError("Application-level prompt message content must be text")
        public_messages.append(QueryPromptMessage(role=role, content=message.content))

    return QueryPromptResponse(
        label="Prompt sent to the model",
        technical_label="Application-level chat messages",
        note=(
            "These are the exact application-level chat messages passed by "
            "Learn2RAG to the configured model. Provider-specific serialization "
            "or chat templates are not shown."
        ),
        messages=public_messages,
    )


def _sanitize_answer(answer: str, points: Sequence[Any]) -> str:
    """Remove reproduced raw source values as defense-in-depth.

    Demo prompts normally contain only safe labels before invocation, but a
    model must not expose an unexpected raw source if one is ever reproduced.
    """
    replacements: dict[str, str] = {}
    for point in points:
        payload_value = getattr(point, "payload", None)
        if not isinstance(payload_value, Mapping):
            continue

        safe_label, _ = _display_metadata(payload_value)
        for field in ("source", "uri"):
            raw_value = payload_value.get(field)
            if isinstance(raw_value, str) and raw_value:
                replacements.setdefault(raw_value, safe_label)

    if not replacements:
        return answer

    raw_values = sorted(replacements, key=lambda value: (-len(value), value))
    pattern = re.compile("|".join(re.escape(value) for value in raw_values))
    return pattern.sub(lambda match: replacements[match.group(0)], answer)


def _chunk_display_id(point: Any, payload: Mapping[str, Any]) -> str:
    identity = (
        "retrieved_chunk",
        _text(payload.get("loader_id")),
        _text(payload.get("document_id")),
        _text(payload.get("source")),
        _text(payload.get("uri")),
        _text(payload.get("content_hash")),
        _text(payload.get("chunk_hash")),
        str(getattr(point, "id", "")),
    )
    return _opaque_id(json.dumps(identity, ensure_ascii=False, separators=(",", ":")))


def _search_metadata(runtime_config: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Describe configured retrieval without presenting scores as confidence."""
    mode = runtime_config.get("search_mode")
    if not isinstance(mode, str) or not mode:
        raise ValueError("A valid search mode is required")

    if mode == "dense":
        label = "Semantic vector search"
        score_label = "Dense similarity"
        technical_mode = "Dense"
    else:
        label = "Configured retrieval"
        score_label = "Retrieval score"
        technical_mode = mode.replace("_", " ").title()

    embedding_model = runtime_config.get("embedding_model")
    embedding_label = "BGE-M3" if embedding_model == "BAAI/bge-m3" else None
    technical_label = (
        f"{technical_mode} / {embedding_label}"
        if embedding_label
        else technical_mode
    )
    return mode, label, technical_label, score_label


def _display_metadata(
    payload: Mapping[str, Any],
) -> tuple[str, Literal["local", "url", "unknown"] | None]:
    """Choose a safe public label without exposing raw source or URI details."""
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
