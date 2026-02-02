from .qdrant import Qdrant
from .embeddings import create_embeddings
import warnings
from qdrant_client import models
import numpy as np

# similarity search
def search(query, user_config, opt_config) -> list:
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
        if opt_config["search_mode"] == "dense_sparse":
            query_embedding = create_embeddings(query, opt_config["embedding_model"], embedding_mode="dense_sparse")
        if opt_config["search_mode"] == "dense_sparse_colbert":
            query_embedding = create_embeddings(query, opt_config["embedding_model"], embedding_mode="dense_sparse_colbert")
        if opt_config["search_mode"] == "dense":
            query_embedding = create_embeddings(query, opt_config["embedding_model"], embedding_mode="dense")["dense_vecs"]
        if opt_config["search_mode"] == "multi_search":
            vecs_to_concat = []
            for item in ["content"]+opt_config["multi_search"]:
                vecs_to_concat.append(create_embeddings(query[item], opt_config["embedding_model"], opt_config["search_mode"])["dense_vecs"])
            query_embedding = np.concatenate(vecs_to_concat, axis=0)
    else:
        query_embedding = create_embeddings(query, opt_config["embedding_model"])

    if opt_config["search_mode"] == "dense":
        results = qdrant.client.query_points(
            collection_name=collection_name,
            query = query_embedding,
            using = "dense",
            limit=opt_config["top_k"],
        )

    elif opt_config["search_mode"] == "dense_sparse":
        indices = [int(k) for k in query_embedding["lexical_weights"].keys()]
        values = [float(v) for v in query_embedding["lexical_weights"].values()]
        results = qdrant.client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(indices=indices, values=values),
                    using="sparse",
                    limit=opt_config["prefetch_limit_sparse"],
                ),
                models.Prefetch(
                    query=query_embedding["dense_vecs"],
                    using="dense",
                    limit=opt_config["prefetch_limit_dense"],
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=opt_config["top_k"],
        )
    
    elif opt_config["search_mode"] == "dense_sparse_colbert":
        indices = [int(k) for k in query_embedding["lexical_weights"].keys()]
        values = [float(v) for v in query_embedding["lexical_weights"].values()]
        results = qdrant.client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(indices=indices, values=values),
                    using="sparse",
                    limit=opt_config["prefetch_limit_sparse"],
                ),
                models.Prefetch(
                    query=query_embedding["dense_vecs"],
                    using="dense",
                    limit=opt_config["prefetch_limit_dense"],
                ),
                models.Prefetch(
                    query=list(query_embedding["colbert_vecs"]),
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
            query = query_embedding,
            using = "multi",
            limit=opt_config["top_k"],
        )

    return results