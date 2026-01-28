from .qdrant import Qdrant
from .embeddings import create_embeddings
import warnings
from qdrant_client import models

# similarity search
def search(query, user_config, opt_config) -> list:
    collection_name = user_config["collection_name"]

    # Init vector store
    qdrant = Qdrant(
        collection_name=collection_name,
        vector_size=opt_config["vector_size"][opt_config["embedding_model"]],
        search_mode=opt_config["search_mode"]
    )

    if opt_config["embedding_model"] == "BAAI/bge-m3":
        if opt_config["search_mode"] == "hybrid":
            query_embedding = create_embeddings(query, opt_config["embedding_model"], embedding_mode="hybrid")
        if opt_config["search_mode"] == "dense":
            query_embedding = create_embeddings(query, opt_config["embedding_model"], embedding_mode="dense")["dense_vecs"]
    else:
        query_embedding = create_embeddings(query, opt_config["embedding_model"])

    if opt_config["search_mode"] == "dense":
        results = qdrant.client.query_points(
            collection_name=collection_name,
            query = query_embedding,
            using = "dense",
            limit=opt_config["top_k"],
        )

    elif opt_config["search_mode"] == "hybrid":
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
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=opt_config["top_k"],
        )

    else:
        warnings.warn(f"Search mode unknown or not provided. Using default mode: dense")
        results = qdrant.client.query_points(
            collection_name=collection_name,
            query = query_embedding,
            using = "dense",
            limit=opt_config["top_k"],
        )
    return results
