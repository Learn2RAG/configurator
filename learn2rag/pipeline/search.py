import warnings
from typing import List, Any, cast
import logging
import copy

import numpy as np
from FlagEmbedding import FlagReranker # type: ignore[import-untyped]
from qdrant_client import models
from qdrant_client.http.models import QueryResponse, ScoredPoint

from .authorization import filter_authorized
from .config import opt_config, user_config
from .embeddings import create_embeddings
from .qdrant import Qdrant
from . import rewrite


profilingLogger = logging.getLogger('profiling')


def _sort_and_deduplicate(points: list[ScoredPoint]) -> list[ScoredPoint]:
    best_by_path: dict[str, ScoredPoint] = {}
    fallback_points: list[ScoredPoint] = []

    for p in points:
        payload = p.payload or {}
        path = payload.get("path")

        # if path is missing/non-string, keep as fallback
        if not isinstance(path, str):
            fallback_points.append(p)
            continue

        current = best_by_path.get(path)
        if current is None or p.score > current.score:
            best_by_path[path] = p

    merged = list(best_by_path.values()) + fallback_points
    return sorted(merged, key=lambda p: p.score, reverse=True)


def _collect_query_points(
    query: str,
    user_config: dict[str, Any],
    opt_config: dict[str, Any],
    *,
    request_id: str | None = None,
) -> list[ScoredPoint]:


    profilingLogger.info(
        "collect_query_points_start query=%r rewrite=%s rewrite_mode=%s request_id=%s",
        query,
        opt_config.get("rewrite"),
        opt_config.get("rewrite_mode"),
        request_id,
    )

    points_all: list[ScoredPoint] = []

    # Base query
    profilingLogger.info(
        "base_search_start query=%r search_mode=%s top_k=%s request_id=%s",
        query,
        opt_config.get("search_mode"),
        opt_config.get("top_k"),
        request_id,
    )
    base_results = search(query, user_config, opt_config, request_id=request_id)
    points_all.extend(base_results.points)

    if opt_config.get("rewrite") == "True":
        rewrite_mode = opt_config.get("rewrite_mode")

        profilingLogger.info(
            "rewrite_enabled query=%r rewrite_mode=%s request_id=%s",
            query,
            rewrite_mode,
            request_id,
        )

        if rewrite_mode in ["subqueries", "subqueries_keywords"]:
            opt_config_subqueries = copy.deepcopy(opt_config)
            opt_config_subqueries["top_k"] = opt_config["top_k_subqueries"]

            subqueries = rewrite.generate_subqueries(query, n=opt_config["n_subqueries"])
            profilingLogger.info(
                "subqueries_generated query=%r subqueries=%r n_subqueries=%d top_k_subqueries=%s request_id=%s",
                query,
                subqueries,
                len(subqueries),
                opt_config_subqueries["top_k"],
                request_id,
            )

            for sq in subqueries:
                sq_results = search(sq, user_config, opt_config_subqueries, request_id=request_id)
                points_all.extend(sq_results.points)

        if rewrite_mode in ["keywords", "subqueries_keywords"]:
            opt_config_keywords = copy.deepcopy(opt_config)
            opt_config_keywords["top_k"] = opt_config["top_k_keywords"]
            opt_config_keywords["search_mode"] = "sparse"

            keywords = rewrite.generate_keywords(query, n=opt_config["n_keywords"])
            profilingLogger.info(
                "keywords_generated query=%r keywords=%r n_keywords=%d top_k_keywords=%s search_mode=%s request_id=%s",
                query,
                keywords,
                len(keywords),
                opt_config_keywords["top_k"],
                opt_config_keywords["search_mode"],
                request_id,
            )

            for kw in keywords:
                kw_results = search(kw, user_config, opt_config_keywords, request_id=request_id)
                points_all.extend(kw_results.points)

    return _sort_and_deduplicate(points_all)


# similarity search
def search(query: str, user_config: dict[str, Any], opt_config: dict[str, Any], *, request_id: str | None=None) -> QueryResponse:
    profilingLogger.info('start', extra={'activity': 'search', 'request_id': request_id})
    profilingLogger.info(
        "search_called query=%r search_mode=%s collection_name=%s request_id=%s",
        query,
        opt_config.get("search_mode"),
        user_config.get("collection_name"),
        request_id,
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
        if opt_config["search_mode"] == "dense_sparse_colbert" or opt_config["search_mode"] == "reranking_with_colbert" or opt_config["search_mode"] == "reranking_with_flagreranker":
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
        results = qdrant.client.query_points(
            collection_name=collection_name,
            query=query_embedding, # type: ignore[arg-type, unused-ignore]
            using="dense",
            limit=opt_config["top_k"],
        )
    elif opt_config["search_mode"] == "sparse":
        indices = [int(k) for k in query_embedding["lexical_weights"].keys()]  # type: ignore[union-attr]
        values = [float(v) for v in query_embedding["lexical_weights"].values()]  # type: ignore[union-attr]
        results = qdrant.client.query_points(
            collection_name=collection_name,
            query=models.SparseVector(indices=indices, values=values),
            using="sparse",
            limit=opt_config["top_k"],
        )
    elif opt_config["search_mode"] == "dense_sparse":
        indices = [int(k) for k in query_embedding["lexical_weights"].keys()] # type: ignore[union-attr]
        values = [float(v) for v in query_embedding["lexical_weights"].values()] # type: ignore[union-attr]
        results = qdrant.client.query_points(
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
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=opt_config["top_k"],
        )
    
    elif opt_config["search_mode"] == "dense_sparse_colbert":
        indices = [int(k) for k in query_embedding["lexical_weights"].keys()] # type: ignore[union-attr]
        values = [float(v) for v in query_embedding["lexical_weights"].values()] # type: ignore[union-attr]
        results = qdrant.client.query_points(
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
        )

    elif opt_config["search_mode"] == "multi_search":
        results = qdrant.client.query_points(
            collection_name=collection_name,
            query=query_embedding, # type: ignore[arg-type, unused-ignore]
            using="multi",
            limit=opt_config["top_k"],
        )

    elif opt_config["search_mode"] == "reranking_with_colbert":
        indices = [int(k) for k in query_embedding["lexical_weights"].keys()] # type: ignore[union-attr]
        values = [float(v) for v in query_embedding["lexical_weights"].values()] # type: ignore[union-attr]
        results = qdrant.client.query_points(
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
            query=list(query_embedding["colbert_vecs"]), # type: ignore[arg-type]
            using="colbert",
            limit=opt_config["top_k"],
        )

    elif opt_config["search_mode"] == "reranking_with_flagreranker":
        reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        indices = [int(k) for k in query_embedding["lexical_weights"].keys()] # type: ignore[union-attr]
        values = [float(v) for v in query_embedding["lexical_weights"].values()] # type: ignore[union-attr]
        results = qdrant.client.query_points(
          collection_name=collection_name,
          prefetch=[
              models.Prefetch(
                  query=models.SparseVector(indices=indices, values=values),
                  using="sparse",
                  limit=opt_config["prefetch_limit_sparse"],
              ),
              models.Prefetch(
                  query=query_embedding["dense_vecs"],  # type: ignore[arg-type]
                  using="dense",
                  limit=opt_config["prefetch_limit_dense"],
              )
          ],
          query=models.FusionQuery(fusion=models.Fusion.RRF),
          limit=opt_config["prefetch_limit_sparse"]+opt_config["prefetch_limit_dense"],
          )
        reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
  
        points = [point for point in results.points if point.payload]
        if len(points) != len(results.points):
            logging.warning(f'{len(results.points) - len(points)} has no payload')
  
        scores = reranker.compute_score([[query, res.payload['content']] for res in points]) # type: ignore[index]
  
        for i in range(len(points)):
          points[i].payload['reranking_score'] = scores[i] # type: ignore[index]
      
        points = sorted(
          points, 
          key=lambda x: x.payload["reranking_score"], # type: ignore[index]
          reverse=True
        )[:opt_config["top_k"]]
  
        results.points = points
    return results
 
def search_multi(multi_query: dict[str, str], user_config: dict[str, Any], opt_config: dict[str, Any], request_id: str | None=None) -> QueryResponse:
    collection_name = user_config["collection_name"]

    # Init vector store
    qdrant = Qdrant(
        collection_name=collection_name,
        opt_config=opt_config
        )

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

    results = qdrant.client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        using="multi",
        limit=opt_config["top_k"],
    )
    profilingLogger.info('end', extra={'activity': 'search', 'request_id': request_id})
    return results

async def search_authorized(question: str, user: str, *, request_id: str | None = None) -> List[ScoredPoint]:
    points = _collect_query_points(question, user_config, opt_config, request_id=request_id)
    query_response = QueryResponse(points=points)
    authorized_points = await filter_authorized(user, query_response)
    # keep deterministic order after auth filter
    return _sort_and_deduplicate(list(authorized_points))
