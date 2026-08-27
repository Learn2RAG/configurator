"""Safe orchestration and projection boundary for the public RAG Demo.

Internal Qdrant and LangChain objects are converted here into explicit public
models so filesystem paths, credentials, and implementation metadata do not
cross into browser-facing responses.

The public routes are intended to read from a dedicated public demo collection,
not an arbitrary private production collection; safe projection remains a
defense-in-depth boundary rather than a substitute for that deployment scope.
"""

import hashlib
import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Literal, Mapping, Protocol, Sequence, cast
from urllib.parse import unquote, urlsplit

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
import numpy as np
from qdrant_client.conversions import common_types as qdrant_types
from qdrant_client.http.models import QueryResponse as QdrantQueryResponse
from qdrant_client.http.models import ScoredPoint

from learn2rag.pipeline.authorization import filter_authorized
from learn2rag.pipeline.embeddings import create_embeddings
from learn2rag.pipeline.generate import build_prompt_messages, invoke_prompt_messages

from .models import (
    IndexedDocument,
    IndexResponse,
    QueryPromptMessage,
    QueryPromptResponse,
    QueryResponse,
    RetrievalMode,
    QuerySearchResponse,
    QuerySearchResult,
    QueryVisualizationPoint,
    QueryVisualizationQueryPoint,
    QueryVisualizationResponse,
)

logger = logging.getLogger(__name__)

SCROLL_PAGE_SIZE = 256
MAX_CHUNKS = 10_000
MAX_VISUALIZATION_CHUNKS = 2_000
MAX_KEYWORD_SEARCH_CHUNKS = 2_000
DEFAULT_KEYWORD_TOP_K = 5
MAX_KEYWORD_RESULTS = 50
BM25_K1 = 1.5
BM25_B = 0.75
MAX_MATCHED_TERMS = 12
MAX_MATCHED_TERM_LENGTH = 64
PAYLOAD_FIELDS = [
    "source",
    "content_hash",
    "title",
    "uri",
    "loader_id",
    "document_id",
]
VISUALIZATION_PAYLOAD_FIELDS = [
    "source",
    "title",
    "uri",
    "loader_id",
    "document_id",
    "content_hash",
    "chunk_hash",
]
KEYWORD_PAYLOAD_FIELDS = [
    "content",
    "source",
    "title",
    "uri",
    "loader_id",
    "document_id",
    "content_hash",
    "chunk_hash",
]

_MISSING_IDENTIFIERS = {"", "n/a", "none", "null", "unknown"}
_WINDOWS_PATH = re.compile(r"^[a-zA-Z]:[\\/]")
_WORD_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)
# Authorization identity is owned by the server; the public request may only
# supply a question and cannot select a user or other pipeline internals.
DEMO_USER = "ragdemo"
NO_EVIDENCE_ANSWER = (
    "No relevant chunks were found with the selected retrieval method, so no "
    "grounded answer can be shown."
)
NO_KEYWORD_EVIDENCE_ANSWER = (
    "No matching keyword chunks were found, so no grounded answer can be shown."
)


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
        with_vectors: bool | Sequence[str],
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


@dataclass(frozen=True)
class _VisualizationChunk:
    id: str
    source: str
    vector: np.ndarray[Any, Any]


@dataclass(frozen=True)
class _QueryRetrieval:
    points: list[Any]
    matched_terms_by_id: Mapping[str, tuple[str, ...]]


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
    *,
    retrieval_mode: RetrievalMode = "semantic",
    qdrant_reader: QdrantReader | None = None,
    collection_name: Any = None,
) -> QueryResponse:
    """Run the selected authorized retrieval and one shared generation path.

    Semantic mode delegates unchanged to SearchOperator. Keyword mode is local
    to the demo because BGE-M3 sparse vectors are not BM25 and the dense-only
    collection should not be migrated merely to demonstrate lexical ranking.
    Both paths converge on the same ordered points before prompt construction.
    """
    if retrieval_mode == "semantic":
        retrieval = await _semantic_retrieval(search_operator, question)
    elif retrieval_mode == "keyword":
        if (
            qdrant_reader is None
            or not isinstance(collection_name, str)
            or not collection_name.strip()
        ):
            raise ValueError("Keyword retrieval requires the configured collection")
        retrieval = await _keyword_retrieval(
            qdrant_reader,
            collection_name,
            question,
            runtime_config,
        )
    else:
        raise ValueError("Unsupported public retrieval mode")

    points = retrieval.points
    mode, label, technical_label, score_label = _search_metadata(
        retrieval_mode,
        runtime_config,
    )
    results = []
    for rank, point in enumerate(points, start=1):
        public_id = _point_display_id(point)
        results.append(
            _public_search_result(
                point,
                rank,
                matched_terms=retrieval.matched_terms_by_id.get(public_id, ()),
            )
        )
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

    public_answer = _public_answer(answer, points, retrieval_mode)
    visualization = (
        _safe_query_visualization(
            qdrant_reader,
            collection_name,
            question,
            runtime_config,
            results,
        )
        if retrieval_mode == "semantic"
        else _visualization_status(
            "unsupported",
            "The embedding explorer is available for semantic vector search.",
        )
    )
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
        visualization=visualization,
        prompt=public_prompt,
    )


def _public_answer(
    model_answer: str,
    points: Sequence[Any],
    retrieval_mode: RetrievalMode,
) -> str:
    """Suppress ungrounded model text at the public demo projection boundary.

    Prompt construction and the single configured model invocation remain
    inspectable even with zero context. The free-form result is deliberately
    discarded in that case so the educational UI never presents it as a
    retrieval-grounded answer.
    """
    if points:
        return _sanitize_answer(model_answer, points)
    if retrieval_mode == "keyword":
        return NO_KEYWORD_EVIDENCE_ANSWER
    return NO_EVIDENCE_ANSWER


async def _semantic_retrieval(
    search_operator: QuerySearchOperator,
    question: str,
) -> _QueryRetrieval:
    """Keep SearchOperator authoritative for the existing semantic path."""
    search_output = await search_operator(
        inputs={"question": question, "user": DEMO_USER}
    )
    if not isinstance(search_output, Mapping):
        raise TypeError("Search output must be a mapping")

    documents = search_output.get("documents")
    points = getattr(documents, "points", documents)
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        raise TypeError("Search documents must be a sequence")
    return _QueryRetrieval(points=list(points), matched_terms_by_id={})


async def _keyword_retrieval(
    client: QdrantReader,
    collection_name: str,
    question: str,
    runtime_config: Mapping[str, Any],
    *,
    page_size: int = SCROLL_PAGE_SIZE,
    max_chunks: int = MAX_KEYWORD_SEARCH_CHUNKS,
) -> _QueryRetrieval:
    """Rank a bounded text snapshot with BM25, then reuse pipeline authorization.

    The scan requests no vectors. Candidates become ordinary ScoredPoint
    objects so the same loader/document authorization policy protects this
    demo-local lexical path before its final configured top-k is selected.
    """
    records, truncated = _scan_keyword_chunks(
        client,
        collection_name,
        page_size=page_size,
        max_chunks=max_chunks,
    )
    candidates, matched_terms = _bm25_candidates(question, records)
    if not candidates:
        return _QueryRetrieval(points=[], matched_terms_by_id={})

    authorized = await filter_authorized(
        DEMO_USER,
        QdrantQueryResponse(points=candidates),
    )
    authorized.sort(key=_keyword_sort_key)
    top_k = _keyword_top_k(runtime_config)
    selected = list(authorized[:top_k])
    selected_ids = {_point_display_id(point) for point in selected}
    if truncated:
        logger.warning(
            "Keyword retrieval used a bounded partial index snapshot of %d chunks",
            max_chunks,
        )
    return _QueryRetrieval(
        points=selected,
        matched_terms_by_id={
            point_id: terms
            for point_id, terms in matched_terms.items()
            if point_id in selected_ids
        },
    )


def _scan_keyword_chunks(
    client: QdrantReader,
    collection_name: str,
    *,
    page_size: int,
    max_chunks: int,
) -> tuple[list[Any], bool]:
    """Read one deterministic bounded text snapshot without requesting vectors."""
    if page_size < 1 or max_chunks < 1:
        raise ValueError("Keyword scan bounds must be positive")
    if not client.collection_exists(collection_name):
        return [], False

    records: list[Any] = []
    offset: qdrant_types.PointId | None = None
    scanned_count = 0
    truncated = False
    while scanned_count < max_chunks:
        limit = min(page_size, max_chunks - scanned_count)
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_payload=KEYWORD_PAYLOAD_FIELDS,
            with_vectors=False,
        )
        bounded_points = points[:limit]
        records.extend(bounded_points)
        scanned_count += len(bounded_points)
        if next_offset is None:
            break
        if not bounded_points or next_offset == offset:
            truncated = True
            break
        if scanned_count >= max_chunks:
            truncated = True
            break
        offset = next_offset
    return records, truncated


def _tokenize_keyword_text(value: str) -> list[str]:
    """Return small Unicode-aware terms without an English-specific NLP stack."""
    tokens = []
    for match in _WORD_TOKEN.finditer(value):
        token = match.group(0).casefold()
        if 2 <= len(token) <= MAX_MATCHED_TERM_LENGTH:
            tokens.append(token)
    return tokens


def _bm25_candidates(
    question: str,
    records: Sequence[Any],
) -> tuple[list[ScoredPoint], dict[str, tuple[str, ...]]]:
    """Apply deterministic BM25 scoring and presentation-term reporting."""
    query_terms = tuple(dict.fromkeys(_tokenize_keyword_text(question)))
    if not query_terms:
        return [], {}

    documents: list[tuple[Any, dict[str, Any], list[str], Counter[str]]] = []
    for record in records:
        payload_value = getattr(record, "payload", None)
        payload = dict(payload_value) if isinstance(payload_value, Mapping) else {}
        content = payload.get("content")
        if not isinstance(content, str) or not isinstance(payload.get("source"), str):
            continue
        tokens = _tokenize_keyword_text(content)
        documents.append((record, payload, tokens, Counter(tokens)))
    if not documents:
        return [], {}

    document_count = len(documents)
    average_length = (
        sum(len(tokens) for _, _, tokens, _ in documents) / document_count
    ) or 1.0
    document_frequency = Counter(
        term
        for term in query_terms
        for _, _, _, frequencies in documents
        if frequencies[term] > 0
    )

    candidates: list[ScoredPoint] = []
    matched_terms_by_id: dict[str, tuple[str, ...]] = {}
    for record, payload, tokens, frequencies in documents:
        score = 0.0
        length_normalization = 1 - BM25_B + BM25_B * len(tokens) / average_length
        for term in query_terms:
            term_frequency = frequencies[term]
            if term_frequency == 0:
                continue
            frequency = document_frequency[term]
            inverse_document_frequency = math.log(
                1 + (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            score += inverse_document_frequency * (
                term_frequency * (BM25_K1 + 1)
                / (term_frequency + BM25_K1 * length_normalization)
            )
        if score <= 0:
            continue

        point_id = getattr(record, "id", None)
        if point_id is None:
            continue
        candidate = ScoredPoint(
            id=point_id,
            version=int(getattr(record, "version", 0) or 0),
            score=float(score),
            payload=payload,
        )
        public_id = _point_display_id(candidate)
        matched_terms_by_id[public_id] = tuple(
            term for term in query_terms if frequencies[term] > 0
        )[:MAX_MATCHED_TERMS]
        candidates.append(candidate)

    candidates.sort(key=_keyword_sort_key)
    return candidates, matched_terms_by_id


def _keyword_sort_key(point: ScoredPoint) -> tuple[float, str]:
    return (-float(point.score), _point_display_id(point))


def _keyword_top_k(runtime_config: Mapping[str, Any]) -> int:
    configured = runtime_config.get("top_k", DEFAULT_KEYWORD_TOP_K)
    if not isinstance(configured, int) or isinstance(configured, bool) or configured < 1:
        raise ValueError("Configured top_k must be a positive integer")
    return min(configured, MAX_KEYWORD_RESULTS)


def _safe_query_visualization(
    client: QdrantReader | None,
    collection_name: Any,
    question: str,
    runtime_config: Mapping[str, Any],
    search_results: Sequence[QuerySearchResult],
) -> QueryVisualizationResponse:
    """Isolate the educational vector view from the authoritative RAG path.

    Search, prompt, and generation have already used one shared retrieval. Any
    scan, embedding, or projection failure here is supplemental and must not
    discard that valid response or expose diagnostic details publicly.
    """
    if runtime_config.get("search_mode") != "dense":
        return _visualization_status(
            "unsupported",
            "This embedding map is available for dense retrieval only.",
        )
    if client is None or not isinstance(collection_name, str) or not collection_name.strip():
        return _visualization_status(
            "unavailable",
            "The embedding map is temporarily unavailable.",
        )

    try:
        return build_query_visualization(
            client,
            collection_name,
            question,
            runtime_config,
            search_results,
        )
    except Exception:
        logger.exception("Unable to build the supplemental dense embedding visualization")
        return _visualization_status(
            "unavailable",
            "The embedding map is temporarily unavailable.",
        )


def build_query_visualization(
    client: QdrantReader,
    collection_name: str,
    question: str,
    runtime_config: Mapping[str, Any],
    search_results: Sequence[QuerySearchResult],
    *,
    page_size: int = SCROLL_PAGE_SIZE,
    max_chunks: int = MAX_VISUALIZATION_CHUNKS,
) -> QueryVisualizationResponse:
    """Project stored dense chunk vectors and the question into three dimensions.

    The bounded, read-only scroll obtains only the named dense vector and the
    payload fields needed for safe labels and the shared public chunk ID. Raw
    vectors remain server-side. The second question embedding exists only
    because SearchOperator does not expose its internal query vector: it never
    performs or changes retrieval.
    """
    if runtime_config.get("search_mode") != "dense":
        return _visualization_status(
            "unsupported",
            "This embedding map is available for dense retrieval only.",
        )
    if page_size < 1 or max_chunks < 1:
        raise ValueError("Visualization scan bounds must be positive")
    if not client.collection_exists(collection_name):
        return _visualization_status(
            "unavailable",
            "No dense embedding points are available to display.",
        )

    chunks, truncated = _scan_visualization_chunks(
        client,
        collection_name,
        page_size=page_size,
        max_chunks=max_chunks,
    )
    if not chunks:
        return _visualization_status(
            "unavailable",
            "No usable dense embedding points are available to display.",
        )

    query_vector = _visualization_query_embedding(question, runtime_config)
    compatible_chunks = [
        chunk for chunk in chunks if chunk.vector.size == query_vector.size
    ]
    if not compatible_chunks:
        return _visualization_status(
            "unavailable",
            "The dense embedding points could not be projected safely.",
        )

    matrix = np.vstack(
        [chunk.vector for chunk in compatible_chunks] + [query_vector]
    )
    coordinates = _pca_3d(matrix)
    retrieved_ranks = {result.id: result.rank for result in search_results}
    public_points = []
    for chunk, coordinate in zip(compatible_chunks, coordinates[:-1], strict=True):
        rank = retrieved_ranks.get(chunk.id)
        public_points.append(
            QueryVisualizationPoint(
                id=chunk.id,
                source=chunk.source,
                x=float(coordinate[0]),
                y=float(coordinate[1]),
                z=float(coordinate[2]),
                retrieved=rank is not None,
                rank=rank,
                preview=None,
            )
        )

    query_coordinate = coordinates[-1]
    status: Literal["ready", "partial"] = "partial" if truncated else "ready"
    note = _visualization_note(truncated)
    return QueryVisualizationResponse(
        status=status,
        label="Explore the embedding space",
        technical_label=_visualization_technical_label(runtime_config),
        note=note,
        points=public_points,
        query=QueryVisualizationQueryPoint(
            x=float(query_coordinate[0]),
            y=float(query_coordinate[1]),
            z=float(query_coordinate[2]),
        ),
        truncated=truncated,
    )


def _scan_visualization_chunks(
    client: QdrantReader,
    collection_name: str,
    *,
    page_size: int,
    max_chunks: int,
) -> tuple[list[_VisualizationChunk], bool]:
    """Collect a deterministic bounded snapshot without exposing its vectors."""
    chunks: list[_VisualizationChunk] = []
    offset: qdrant_types.PointId | None = None
    scanned_count = 0
    truncated = False

    while scanned_count < max_chunks:
        limit = min(page_size, max_chunks - scanned_count)
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_payload=VISUALIZATION_PAYLOAD_FIELDS,
            with_vectors=["dense"],
        )
        bounded_points = points[:limit]
        scanned_count += len(bounded_points)

        for point in bounded_points:
            chunk = _visualization_chunk(point)
            if chunk is not None:
                chunks.append(chunk)

        if next_offset is None:
            break
        if not bounded_points:
            truncated = True
            break
        if scanned_count >= max_chunks:
            truncated = True
            break
        offset = next_offset

    return chunks, truncated


def _visualization_chunk(point: Any) -> _VisualizationChunk | None:
    payload_value = getattr(point, "payload", None)
    payload = payload_value if isinstance(payload_value, Mapping) else {}
    vectors = getattr(point, "vector", None)
    if not isinstance(vectors, Mapping) or "dense" not in vectors:
        return None
    vector = _finite_vector(vectors["dense"])
    if vector is None:
        return None
    source, _ = _display_metadata(payload)
    return _VisualizationChunk(
        id=_chunk_display_id(point, payload),
        source=source,
        vector=vector,
    )


def _visualization_query_embedding(
    question: str,
    runtime_config: Mapping[str, Any],
) -> np.ndarray[Any, Any]:
    """Embed the question again for display, without issuing another search."""
    model_name = runtime_config.get("embedding_model")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("A configured embedding model is required")
    embedding = create_embeddings([question], model_name, embedding_mode="dense")
    dense_values = embedding.get("dense_vecs") if isinstance(embedding, Mapping) else embedding
    dense_array = np.asarray(dense_values, dtype=np.float64)
    if dense_array.ndim != 2 or dense_array.shape[0] != 1:
        raise ValueError("The visualization query embedding has an invalid shape")
    vector = _finite_vector(dense_array[0])
    if vector is None:
        raise ValueError("The visualization query embedding is not usable")
    return vector


def _finite_vector(value: Any) -> np.ndarray[Any, Any] | None:
    """Normalize one dense vector and reject malformed data before PCA."""
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if vector.ndim != 1 or vector.size < 1 or not np.isfinite(vector).all():
        return None
    return vector


def _pca_3d(matrix: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Return a finite 3D explanatory PCA view, not retrieval-space distances.

    Three coordinates give the browser enough depth for a playable camera while
    raw embeddings remain server-side. Missing mathematical components are
    zero-padded, and signs are normalized around their largest coordinate so
    repeated projections do not arbitrarily mirror the display layout.
    """
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("At least two vectors with one dimension are required")
    if not np.isfinite(matrix).all():
        raise ValueError("PCA input must be finite")

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    left_vectors, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    component_count = min(3, singular_values.size)
    projected = left_vectors[:, :component_count] * singular_values[:component_count]
    for column_index in range(component_count):
        column = projected[:, column_index]
        pivot_index = int(np.argmax(np.abs(column)))
        if column[pivot_index] < 0:
            projected[:, column_index] *= -1
    if component_count < 3:
        projected = np.pad(projected, ((0, 0), (0, 3 - component_count)))
    if not np.isfinite(projected).all():
        raise ValueError("PCA output must be finite")
    return cast(np.ndarray[Any, Any], projected)


def _visualization_status(
    status: Literal["unavailable", "unsupported"],
    note: str,
) -> QueryVisualizationResponse:
    return QueryVisualizationResponse(
        status=status,
        label="Explore the embedding space",
        technical_label="3D PCA projection of dense embeddings",
        note=note,
        points=[],
        query=None,
        truncated=False,
    )


def _visualization_technical_label(runtime_config: Mapping[str, Any]) -> str:
    model_name = runtime_config.get("embedding_model")
    model_label = "BGE-M3" if model_name == "BAAI/bge-m3" else None
    base = "3D PCA projection of dense embeddings"
    return f"{base} / {model_label}" if model_label else base


def _visualization_note(truncated: bool) -> str:
    base = (
        "This is a 3D PCA projection. Retrieval itself uses the full-dimensional "
        "dense embedding space."
    )
    if truncated:
        return f"{base} The bounded display shows a partial index snapshot."
    return base


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


def _public_search_result(
    point: Any,
    rank: int,
    *,
    matched_terms: Sequence[str] = (),
) -> QuerySearchResult:
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
        # Terms are bounded presentation metadata; ranking and prompt content
        # remain entirely determined by the selected internal points.
        matched_terms=list(matched_terms),
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


def _point_display_id(point: Any) -> str:
    payload_value = getattr(point, "payload", None)
    payload = payload_value if isinstance(payload_value, Mapping) else {}
    return _chunk_display_id(point, payload)


def _search_metadata(
    retrieval_mode: RetrievalMode,
    runtime_config: Mapping[str, Any],
) -> tuple[RetrievalMode, str, str, str]:
    """Describe the public choice without presenting scores as confidence."""
    if retrieval_mode == "keyword":
        return "keyword", "Keyword search", "BM25 lexical ranking", "BM25 score"

    configured_mode = runtime_config.get("search_mode")
    if not isinstance(configured_mode, str) or not configured_mode:
        raise ValueError("A valid configured search mode is required")
    technical_mode = (
        "Dense" if configured_mode == "dense"
        else configured_mode.replace("_", " ").title()
    )

    embedding_model = runtime_config.get("embedding_model")
    embedding_label = "BGE-M3" if embedding_model == "BAAI/bge-m3" else None
    technical_label = (
        f"{technical_mode} / {embedding_label}"
        if embedding_label
        else technical_mode
    )
    return "semantic", "Semantic vector search", technical_label, "Dense similarity"


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
