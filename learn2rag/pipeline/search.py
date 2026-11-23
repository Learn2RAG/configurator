import asyncio
import warnings
from typing import (
    cast,
    Any,
    Iterable,
    List,
    Mapping,
    Sequence,
)
import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from FlagEmbedding import FlagReranker # type: ignore[import-untyped]
from sentence_transformers import CrossEncoder
from sentence_transformers.base.modality_types import PairInput
from qdrant_client import models
from qdrant_client.http.models import QueryResponse, ScoredPoint

from .authorization import DOC_ID_FIELDS, _get_doc_id, _get_loader_id, filter_authorized_cached, get_doc_id_field
from .chat import Message
from .config import opt_config, user_config
from .embeddings import create_embeddings
from .qdrant import Qdrant
from . import rewrite

logger = logging.getLogger(__name__)
profilingLogger = logging.getLogger('profiling')


@lru_cache(maxsize=4)
def _get_flag_reranker(model_name: str, use_fp16: bool) -> FlagReranker:
    return FlagReranker(model_name, use_fp16=use_fp16)


@lru_cache(maxsize=4)
def _get_cross_encoder(model_name: str) -> CrossEncoder:
    return cast(CrossEncoder, CrossEncoder(model_name))


def _rerank_points_with_flagreranker(
    query: str,
    points: list[ScoredPoint],
    *,
    top_k: int | None = None,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    use_fp16: bool = True,
) -> list[ScoredPoint]:
    reranker = _get_flag_reranker(model_name, use_fp16)

    valid_points = [p for p in points if p.payload and isinstance(p.payload.get("content"), str)]
    if len(valid_points) != len(points):
        logging.warning("%d points have no usable payload", len(points) - len(valid_points))

    if not valid_points:
        return []

    pairs = [[query, cast(str, p.payload["content"])] for p in valid_points if p.payload is not None]
    scores = reranker.compute_score(pairs)

    for p, score in zip(valid_points, scores):
        p.payload["reranking_score"] = score  # type: ignore[index]

    return sorted(
        valid_points,
        key=lambda p: cast(float, p.payload["reranking_score"]),  # type: ignore[index]
        reverse=True,
    )[:top_k]


def _rerank_points_with_sentence_transformers(
        query: str,
        points: list[ScoredPoint],
        *,
        top_k: int | None = None,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
) -> list[ScoredPoint]:

    model = _get_cross_encoder(model_name)

    valid_points = [p for p in points if p.payload and isinstance(p.payload.get("content"), str)]
    if len(valid_points) != len(points):
        logging.warning("%d points have no usable payload", len(points) - len(valid_points))

    if not valid_points:
        return []

    # CrossEncoder expects list of (query, text) pairs
    pairs: list[PairInput] = [(query, cast(str, p.payload["content"])) for p in valid_points if p.payload is not None]
    scores = model.predict(pairs)

    # Attach scores to payload
    for p, score in zip(valid_points, scores):
        p.payload["reranking_score"] = float(score)  # type: ignore[index]

    return sorted(
        valid_points,
        key=lambda p: cast(float, p.payload["reranking_score"]),  # type: ignore[index]
        reverse=True,
    )[:top_k]


def _rerank_points_with_colbert(
    query: str,
    points: list[ScoredPoint],
    *,
    top_k: int | None = None,
    opt_config: dict[str, Any],
    user_config: dict[str, Any],
) -> list[ScoredPoint]:
    if not points:
        return []

    collection_name = user_config["collection_name"]
    qdrant = Qdrant(collection_name=collection_name, opt_config=opt_config)

    emb = create_embeddings([query], opt_config["embedding_model"], embedding_mode="colbert")
    colbert_vecs = emb["colbert_vecs"]
    colbert_query = colbert_vecs[0] if len(colbert_vecs) > 0 else []

    candidate_ids = [p.id for p in points]
    results = qdrant.get_client().query_points(
        collection_name=collection_name,
        query_filter=models.Filter(
            must=[
                models.HasIdCondition(has_id=candidate_ids)
            ]
        ),
        query=colbert_query,  # type: ignore[arg-type]
        using="colbert",
        limit=len(candidate_ids) if top_k is None else top_k,
        timeout=120
    )
    return list(results.points)


# similarity search
def search(query: str, user_config: dict[str, Any], opt_config: dict[str, Any], *, request_id: str | None = None) -> QueryResponse:
    profilingLogger.info('start', extra={'activity': 'search', 'request_id': request_id})
    profilingLogger.info(
        "search_called query=%r search_mode=%s collection_name=%s",
        query,
        opt_config.get("search_mode"),
        user_config.get("collection_name"),
        extra={'activity': '_collect_query_points', 'request_id': request_id},
    )
    collection_name = user_config["collection_name"]
    if opt_config["fusion_mode"] == "RRF":
        fusion_mode = models.Fusion.RRF
    if opt_config["fusion_mode"] == "DBSF":
        fusion_mode = models.Fusion.DBSF

    # Init vector store
    qdrant = Qdrant(
        collection_name=collection_name,
        opt_config=opt_config
        )
    if not qdrant.get_client().collection_exists(collection_name):
        # Nothing has been imported yet
        logger.warning("Collection %s does not exist, returning no documents", collection_name)
        return QueryResponse(points=[])

    if opt_config["embedding_model"] == "BAAI/bge-m3":
        if opt_config["search_mode"] == "dense":
            query_embedding = create_embeddings([query], opt_config["embedding_model"], embedding_mode="dense")["dense_vecs"][0]
        if opt_config["search_mode"] == "sparse":
            query_embedding = (
                lambda x: {
                    "lexical_weights": x["lexical_weights"][0],
                }
            )(create_embeddings([query], opt_config["embedding_model"], embedding_mode="sparse"))
        if opt_config["search_mode"] == "dense_sparse":
            query_embedding = (
                lambda x: {
                    'dense_vecs': x['dense_vecs'][0],
                    'lexical_weights': x['lexical_weights'][0],
                    'colbert_vecs': x['colbert_vecs']
                    }
                )(create_embeddings([query], opt_config["embedding_model"], embedding_mode="dense_sparse"))
        if opt_config["search_mode"] == "dense_sparse_colbert":
            query_embedding = (
                lambda x: {
                    'dense_vecs': x['dense_vecs'][0],
                    'lexical_weights': x['lexical_weights'][0],
                    'colbert_vecs': x['colbert_vecs'][0]
                    }
                )(create_embeddings([query], opt_config["embedding_model"], embedding_mode="dense_sparse_colbert"))
        if opt_config["search_mode"] == "multi_search":
            vecs_to_concat: list[np.ndarray[Any, Any]] = []
            for item in ["content"]+opt_config["multi_search"]:
                vecs_to_concat.append(cast(np.ndarray[Any, Any], create_embeddings([query[item]], opt_config["embedding_model"], opt_config["search_mode"])["dense_vecs"]))
            query_embedding = np.concatenate(vecs_to_concat, axis=0)
    else:
        query_embedding = create_embeddings([query], opt_config["embedding_model"])
    if opt_config["search_mode"] == "dense":
        results = qdrant.get_client().query_points(
            collection_name=collection_name,
            query=query_embedding, # type: ignore[arg-type, unused-ignore]
            using="dense",
            limit=opt_config["top_k"],
            timeout=120
        )
    elif opt_config["search_mode"] == "sparse":
        indices = [int(k) for k in query_embedding["lexical_weights"].keys()]  # type: ignore[union-attr]
        values = [float(v) for v in query_embedding["lexical_weights"].values()]  # type: ignore[union-attr]
        results = qdrant.get_client().query_points(
            collection_name=collection_name,
            query=models.SparseVector(indices=indices, values=values),
            using="sparse",
            limit=opt_config["top_k"],
            timeout=120
        )
    elif opt_config["search_mode"] == "dense_sparse":
        indices = [int(k) for k in query_embedding["lexical_weights"].keys()] # type: ignore[union-attr]
        values = [float(v) for v in query_embedding["lexical_weights"].values()] # type: ignore[union-attr]
        results = qdrant.get_client().query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(indices=indices, values=values),
                    using="sparse",
                    limit=opt_config["prefetch_limit_sparse"],
                ),
                models.Prefetch(
                    query=query_embedding["dense_vecs"], # type: ignore[arg-type]
                    using="dense",
                    limit=opt_config["prefetch_limit_dense"],
                )
            ],
            query=models.FusionQuery(fusion=fusion_mode),
            limit=opt_config["top_k"],
            timeout=120
        )
    
    elif opt_config["search_mode"] == "dense_sparse_colbert":
        indices = [int(k) for k in query_embedding["lexical_weights"].keys()] # type: ignore[union-attr]
        values = [float(v) for v in query_embedding["lexical_weights"].values()] # type: ignore[union-attr]
        results = qdrant.get_client().query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(indices=indices, values=values),
                    using="sparse",
                    limit=opt_config["prefetch_limit_sparse"],
                ),
                models.Prefetch(
                    query=query_embedding["dense_vecs"], # type: ignore[arg-type]
                    using="dense",
                    limit=opt_config["prefetch_limit_dense"],
                ),
                models.Prefetch(
                    query=list(query_embedding["colbert_vecs"]), # type: ignore[arg-type]
                    using="colbert",
                    limit=opt_config["prefetch_limit_colbert"],
                ),
            ],
            query=models.FusionQuery(fusion=fusion_mode),
            limit=opt_config["top_k"],
            timeout=120
        )

    elif opt_config["search_mode"] == "multi_search":
        results = qdrant.get_client().query_points(
            collection_name=collection_name,
            query=query_embedding, # type: ignore[arg-type, unused-ignore]
            using="multi",
            limit=opt_config["top_k"],
            timeout=120
        )
    return results

def search_multi(multi_query: dict[str, str], user_config: dict[str, Any], opt_config: dict[str, Any]) -> QueryResponse:
    collection_name = user_config["collection_name"]

    # Init vector store
    qdrant = Qdrant(
        collection_name=collection_name,
        opt_config=opt_config
        )
    if not qdrant.get_client().collection_exists(collection_name):
        # Nothing has been imported yet
        logger.warning("Collection %s does not exist, returning no documents", collection_name)
        return QueryResponse(points=[])

    query_embedding: np.ndarray[Any, Any]
    if opt_config["embedding_model"] == "BAAI/bge-m3":
        if opt_config["query_mode"] == "multi":
            vecs_to_concat: list[np.ndarray[Any, Any]] = []
            for item in ["content"]+opt_config["multi_search"]:
                vecs_to_concat.append(cast(np.ndarray[Any, Any], create_embeddings([multi_query[item]], opt_config["embedding_model"], opt_config["search_mode"])["dense_vecs"][0]))
            query_embedding = np.concatenate(vecs_to_concat, axis=0)
    else:
        raise NotImplementedError(
            f"Embedding model '{opt_config['embedding_model']}' not supported. "
            "Only 'BAAI/bge-m3' with 'multi' query_mode implemented."
        )

    if opt_config["search_mode"] != "dense":
        warnings.warn(f"Search mode {opt_config["search_mode"]} for multimodal vector is not implemented yet. Using dense.")

    results = qdrant.get_client().query_points(
        collection_name=collection_name,
        query=query_embedding, # type: ignore[arg-type, unused-ignore]
        using="multi",
        limit=opt_config["top_k"],
        timeout=120
    )
    return results


@dataclass(frozen=True)
class _Queries:
    """The queries of one request: effective question, subqueries and keywords."""
    question: str
    subqueries: list[str]
    keywords: list[str]


async def _no_queries() -> list[str]:
    return []


async def _build_queries(
    question: str,
    history: Sequence[Message],
    opt_config: dict[str, Any],
    *,
    request_id: str | None = None,
) -> _Queries:
    """Run the rewrite LLM calls once per request.

    The effective question is the history-contextualized question, or the
    original one without history, without the `history` component or if the
    contextualization returns nothing. Subqueries and keywords are generated
    from the effective question.
    """
    rewrite_components = (opt_config.get("rewrite_mode") or "").split("_") if opt_config.get("rewrite") == "True" else []

    effective_question = question
    if history and "history" in rewrite_components:
        contextualized_question = await asyncio.to_thread(rewrite.contextualize_query, question, history, opt_config)
        if contextualized_question:
            profilingLogger.info(
                "history_rewrite_applied original_query=%r contextualized_query=%r",
                question,
                contextualized_question,
                extra={'activity': 'search_authorized', 'request_id': request_id},
            )
            effective_question = contextualized_question

    subqueries, keywords = await asyncio.gather(
        asyncio.to_thread(rewrite.generate_subqueries, effective_question, n=opt_config["n_subqueries"])
        if "subqueries" in rewrite_components else _no_queries(),
        asyncio.to_thread(rewrite.generate_keywords, effective_question, n=opt_config["n_keywords"])
        if "keywords" in rewrite_components else _no_queries(),
    )
    if rewrite_components:
        profilingLogger.info(
            "queries_built query=%r subqueries=%r keywords=%r",
            effective_question,
            subqueries,
            keywords,
            extra={'activity': 'search_authorized', 'request_id': request_id},
        )
    return _Queries(effective_question, subqueries, keywords)


@dataclass(frozen=True)
class _Embedding:
    """Query vectors of one query; only the vector types needed by the search mode are set."""
    dense: np.ndarray[Any, Any] | None = None
    sparse: models.SparseVector | None = None
    colbert: list[np.ndarray[Any, Any]] | None = None


@dataclass(frozen=True)
class _QueryEmbeddings:
    question: _Embedding
    subqueries: list[_Embedding]
    keywords: list[_Embedding]


_SEARCH_MODE_VECTORS = {
    "dense": ("dense",),
    "sparse": ("sparse",),
    "dense_sparse": ("dense", "sparse"),
    "dense_sparse_colbert": ("dense", "sparse", "colbert"),
}


def _embed_texts(texts: list[str], embedding_mode: str, opt_config: dict[str, Any]) -> list[_Embedding]:
    """Embed all texts with one model call."""
    if not texts:
        return []
    embedding_model = opt_config["embedding_model"]
    result = create_embeddings(texts, embedding_model, embedding_mode=embedding_mode)
    if embedding_model != "BAAI/bge-m3":
        # Other models only provide dense vectors.
        if embedding_mode != "dense":
            raise ValueError(f"Embedding model '{embedding_model}' only supports the dense search mode")
        return [_Embedding(dense=row) for row in cast(np.ndarray[Any, Any], result)]

    vectors = cast(dict[str, Any], result)
    return [
        _Embedding(
            dense=vectors["dense_vecs"][i] if "dense" in embedding_mode else None,
            sparse=models.SparseVector(
                indices=[int(k) for k in vectors["lexical_weights"][i].keys()],
                values=[float(v) for v in vectors["lexical_weights"][i].values()],
            ) if "sparse" in embedding_mode else None,
            colbert=list(vectors["colbert_vecs"][i]) if "colbert" in embedding_mode else None,
        )
        for i in range(len(texts))
    ]


def _embed_queries(queries: _Queries, opt_config: dict[str, Any]) -> _QueryEmbeddings:
    """Create the query embeddings of one request.

    Question and subqueries are embedded together in the `search_mode`, the
    keywords with the sparse model.
    """
    search_mode = opt_config["search_mode"]
    if search_mode == "multi_search":
        raise ValueError("search_mode 'multi_search' is not supported by search_authorized")
    if search_mode not in _SEARCH_MODE_VECTORS:
        raise ValueError(f"Unknown search_mode '{search_mode}'")

    embeddings = _embed_texts([queries.question, *queries.subqueries], search_mode, opt_config)
    keyword_embeddings = _embed_texts(queries.keywords, "sparse", opt_config)
    return _QueryEmbeddings(embeddings[0], embeddings[1:], keyword_embeddings)


def _limit(node: models.Prefetch) -> int:
    return cast(int, node.limit)


def _fuse(children: list[models.Prefetch], query: models.FusionQuery | models.RrfQuery, limit: int | None = None) -> models.Prefetch:
    """Fuse the children into one node; a single child is returned as is.

    Without `limit` the node keeps all candidates of its children.
    """
    if len(children) == 1:
        return children[0]
    return models.Prefetch(
        prefetch=children,
        query=query,
        limit=sum(_limit(child) for child in children) if limit is None else limit,
    )


def _rrf_query(weights: list[float] | None = None, k: int | None = None) -> models.RrfQuery:
    return models.RrfQuery(rrf=models.Rrf(k=k, weights=weights))


def _vector_branch(
    embedding: _Embedding,
    opt_config: dict[str, Any],
    limit: int | None = None,
    exclusion: models.Filter | None = None,
) -> models.Prefetch:
    """Search with all vector types of the search mode and fuse them with `fusion_mode`.

    Without `limit` the leaves use `prefetch_limit_*` and the fusion keeps all their candidates.
    With `limit` (subqueries) a single leaf and the fusion are limited to it.
    `exclusion` is set as filter on every leaf.
    """
    leaves = [
        models.Prefetch(
            query=getattr(embedding, vector_type),
            using=vector_type,
            filter=exclusion,
            limit=opt_config[f"prefetch_limit_{vector_type}"],
        )
        for vector_type in _SEARCH_MODE_VECTORS[opt_config["search_mode"]]
    ]
    if len(leaves) == 1:
        return leaves[0] if limit is None else leaves[0].model_copy(update={"limit": limit})

    fusion_mode = {"RRF": models.Fusion.RRF, "DBSF": models.Fusion.DBSF}.get(opt_config.get("fusion_mode"))  # type: ignore[arg-type]
    if fusion_mode is None:
        raise ValueError(f"fusion_mode must be 'RRF' or 'DBSF' for search_mode '{opt_config['search_mode']}'")
    return _fuse(leaves, models.FusionQuery(fusion=fusion_mode), limit)


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


_DEFAULT_MAX_SEARCH_ROUNDS = 3


def _read_max_search_rounds(opt_config: dict[str, Any]) -> int:
    """Read `max_search_rounds`; a missing or invalid value means the default."""
    value = opt_config.get("max_search_rounds", _DEFAULT_MAX_SEARCH_ROUNDS)
    if not _is_positive_int(value):
        logger.warning("max_search_rounds must be an integer >= 1, got %r; using %d", value, _DEFAULT_MAX_SEARCH_ROUNDS)
        return _DEFAULT_MAX_SEARCH_ROUNDS
    return cast(int, value)


@dataclass(frozen=True)
class _FusionParams:
    weights: tuple[float, float]
    """Weights of branch A (original) and branch B (rewrite) in the root fusion."""
    rrf_k: int | None


def _read_fusion_params(opt_config: dict[str, Any]) -> _FusionParams:
    """Read `fusion_weights` (default 1.0 each) and the optional `rrf_k`.

    Invalid values fall back to their default with a warning.
    """
    weights = opt_config.get("fusion_weights", {})
    if not isinstance(weights, dict):
        logger.warning("fusion_weights must be an object with the keys 'original' and 'rewrite', got %r; using 1.0 for both",
                       weights)
        weights = {}
    values = []
    for key in ("original", "rewrite"):
        value = weights.get(key, 1.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
            logger.warning("fusion_weights.%s must be a number > 0, got %r; using 1.0", key, value)
            value = 1.0
        values.append(float(value))

    rrf_k = opt_config.get("rrf_k")
    if rrf_k is not None and not _is_positive_int(rrf_k):
        logger.warning("rrf_k must be an integer > 0, got %r; using the Qdrant default", rrf_k)
        rrf_k = None
    return _FusionParams((values[0], values[1]), rrf_k)


def _build_prefetch_tree(
    embeddings: _QueryEmbeddings,
    opt_config: dict[str, Any],
    fusion_params: _FusionParams,
    exclusion: models.Filter | None = None,
) -> models.Prefetch:
    """Build the fusion tree of one search round.

    root   weighted RRF over A (original) and B (rewrites)
    A      vector types of the question, fused with `fusion_mode`
    B      RRF over B1 (RRF over the subqueries) and B2 (RRF over the keywords)

    `fusion_params.weights` weights A and B, `fusion_params.rrf_k` sets the RRF constant of
    B1, B2, B and the root. A node with a single child is replaced by the child. All fusion
    nodes above the individual searches keep all candidates of their children.
    `exclusion` is set as filter on every individual search, so that excluded points do
    not take places within their limits.
    """
    rrf_k = fusion_params.rrf_k
    branch_a = _vector_branch(embeddings.question, opt_config, exclusion=exclusion)

    subqueries = [
        _vector_branch(e, opt_config, opt_config["top_k_subqueries"], exclusion) for e in embeddings.subqueries
    ]
    keywords = [
        models.Prefetch(query=e.sparse, using="sparse", filter=exclusion, limit=opt_config["top_k_keywords"])
        for e in embeddings.keywords
    ]
    groups = [_fuse(group, _rrf_query(k=rrf_k)) for group in (subqueries, keywords) if group]
    if not groups:
        return branch_a

    branch_b = _fuse(groups, _rrf_query(k=rrf_k))
    return _fuse([branch_a, branch_b], _rrf_query(list(fusion_params.weights), rrf_k))


def _is_empty(field: str) -> models.Filter:
    """Missing, `null` or `""`, the values `_get_doc_id` skips (apart from other falsy ones like `0`)."""
    return models.Filter(should=[
        models.IsEmptyCondition(is_empty=models.PayloadField(key=field)),
        models.FieldCondition(key=field, match=models.MatchValue(value="")),
    ])


def _document_exclusion(point: ScoredPoint) -> models.Filter | None:
    """Condition matching all chunks of the document of `point`.

    It compares `loader_id` and exactly the field the document ID comes from, with the stored
    value; the fields before it must be empty. Returns `None` if the point ID is the document
    ID (the point itself is excluded by its ID) or a value cannot be matched by Qdrant.
    """
    field, value = get_doc_id_field(point)
    loader_id = (point.payload or {}).get("loader_id")
    if field == "id" or not isinstance(value, (str, int)) or not isinstance(loader_id, (str, int)):
        return None
    preceding = DOC_ID_FIELDS[:DOC_ID_FIELDS.index(field)]
    return models.Filter(must=[
        models.FieldCondition(key="loader_id", match=models.MatchValue(value=loader_id)),
        models.FieldCondition(key=field, match=models.MatchValue(value=value)),
        *(_is_empty(earlier) for earlier in preceding),
    ])


def _exclusion_filter(
    point_ids: Sequence[models.ExtendedPointId],
    documents: Iterable[models.Filter],
) -> models.Filter | None:
    """Filter excluding the given points and all chunks of the given documents; `None` if there is nothing to exclude."""
    must_not: list[models.Condition] = [models.HasIdCondition(has_id=list(point_ids))] if point_ids else []
    must_not += documents
    return models.Filter(must_not=must_not) if must_not else None


def _retrieve_pool(
    root: models.Prefetch,
    user_config: dict[str, Any],
    opt_config: dict[str, Any],
    *,
    request_id: str | None = None,
) -> list[ScoredPoint]:
    """Run the fusion tree in one Qdrant call and return the pool in fusion order."""
    collection_name = user_config["collection_name"]
    qdrant = Qdrant(collection_name=collection_name, opt_config=opt_config)

    if root.prefetch is None:
        # The tree is a single search, its filter becomes the filter of the query.
        response = qdrant.get_client().query_points(
            collection_name=collection_name,
            query=root.query,
            using=root.using,
            query_filter=root.filter,
            limit=_limit(root),
            timeout=120,
        )
    else:
        response = qdrant.get_client().query_points(
            collection_name=collection_name,
            prefetch=root.prefetch,
            query=root.query,
            limit=_limit(root),
            timeout=120,
        )
    profilingLogger.info(
        "pool_retrieved points=%d limit=%d",
        len(response.points),
        _limit(root),
        extra={'activity': 'search_authorized', 'request_id': request_id},
    )
    return list(response.points)


def _rerank_pool(
    question: str,
    pool: list[ScoredPoint],
    user_config: dict[str, Any],
    opt_config: dict[str, Any],
    *,
    request_id: str | None = None,
) -> list[ScoredPoint]:
    """Sort the whole pool with the configured reranker; the pool keeps its fusion order without one.

    `question` is the effective question. Candidates are only sorted, never cut off.
    """
    if opt_config.get("reranking") != "True":
        return pool

    reranking_mode = opt_config["reranking_mode"]
    profilingLogger.info(
        "reranking_enabled query=%r reranking_mode=%s pool=%d",
        question,
        reranking_mode,
        len(pool),
        extra={'activity': 'search_authorized', 'request_id': request_id},
    )
    if reranking_mode == "reranking_with_flagreranker":
        return _rerank_points_with_flagreranker(question, pool)
    if reranking_mode == "reranking_with_sentence_transformers":
        return _rerank_points_with_sentence_transformers(question, pool)
    if reranking_mode == "reranking_with_colbert":
        return _rerank_points_with_colbert(question, pool, opt_config=opt_config, user_config=user_config)
    raise ValueError(f"Unknown reranking_mode '{reranking_mode}'")


async def search_authorized(
    question: str,
    user_auths: Mapping[str, Any],
    *,
    history: Sequence[Message] = (),
    request_id: str | None = None,
    user_config: dict[str, Any] = user_config,
    opt_config: dict[str, Any] = opt_config,
) -> List[ScoredPoint]:
    """Search in rounds and return up to `top_k` authorized points, ordered by round and rank.

    Queries and embeddings are built once. Each round retrieves a pool with the same limits,
    sorts it and checks it lazily in blocks of `top_k` candidates. If fewer than `top_k`
    points are authorized, the next round excludes all points delivered so far and all
    rejected documents. The rounds end at `top_k` authorized points, after
    `max_search_rounds` rounds or with an empty pool. Each document is sent to the
    authorization filter at most once per request.
    """
    top_k = opt_config.get("top_k", 10)
    max_search_rounds = _read_max_search_rounds(opt_config)
    fusion_params = _read_fusion_params(opt_config)
    queries = await _build_queries(question, history, opt_config, request_id=request_id)
    embeddings = await asyncio.to_thread(_embed_queries, queries, opt_config)

    checked: dict[tuple[str, str], bool] = {}
    authorized_points: list[ScoredPoint] = []
    delivered_ids: list[models.ExtendedPointId] = []
    rejected: dict[tuple[str, str], models.Filter | None] = {}
    n_pool = n_checked = rounds = 0
    stop_reason = "max_rounds"
    while rounds < max_search_rounds:
        rounds += 1
        excluded_documents = [condition for condition in rejected.values() if condition is not None]
        exclusion = _exclusion_filter(delivered_ids, excluded_documents)
        root = _build_prefetch_tree(embeddings, opt_config, fusion_params, exclusion)
        pool = await asyncio.to_thread(_retrieve_pool, root, user_config, opt_config, request_id=request_id)
        if pool:
            pool = await asyncio.to_thread(
                _rerank_pool, queries.question, pool, user_config, opt_config, request_id=request_id)

        n_authorized_before = len(authorized_points)
        n_round_checked = 0
        for start in range(0, len(pool), top_k):
            block = pool[start:start + top_k]
            authorized_points += await filter_authorized_cached(user_auths, block, checked)
            n_round_checked += len(block)
            if len(authorized_points) >= top_k:
                break
        n_pool += len(pool)
        n_checked += n_round_checked

        profilingLogger.info(
            "search_round round=%d pool=%d checked=%d authorized=%d excluded_points=%d excluded_documents=%d",
            rounds,
            len(pool),
            n_round_checked,
            len(authorized_points) - n_authorized_before,
            len(delivered_ids),
            len(excluded_documents),
            extra={'activity': 'search_authorized', 'request_id': request_id},
        )
        if not pool:
            stop_reason = "exhausted"
            break
        if len(authorized_points) >= top_k:
            stop_reason = "top_k"
            break

        # Not enough authorized points: the whole pool is checked.
        delivered_ids += [point.id for point in pool]
        for point in pool:
            if not point.payload:
                continue
            key = (await _get_loader_id(point), await _get_doc_id(point))
            if checked.get(key) is False and key not in rejected:
                rejected[key] = _document_exclusion(point)

    profilingLogger.info(
        "authorization_checked pool=%d checked=%d documents=%d authorized=%d top_k=%d rounds=%d stop_reason=%s",
        n_pool,
        n_checked,
        len(checked),
        len(authorized_points),
        top_k,
        rounds,
        stop_reason,
        extra={'activity': 'search_authorized', 'request_id': request_id},
    )
    return authorized_points[:top_k]
