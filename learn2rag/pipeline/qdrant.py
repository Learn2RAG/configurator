import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, MultiVectorComparator, MultiVectorConfig

from .config import user_config


class Qdrant:
    client = QdrantClient(
        location=os.environ.get('QDRANT_LOCATION', 'http://localhost:6336'),
        api_key=os.environ.get('QDRANT__SERVICE__API_KEY'),
    )

    def __init__(self, collection_name: str, opt_config: dict[str, Any]) -> None:
        self.collection_name = collection_name
        self.vector_size = opt_config["vector_size"][opt_config["embedding_model"]]
        self.search_mode = opt_config["search_mode"]
        self.query_mode = opt_config["query_mode"]
        self.multi_search = opt_config["multi_search"]

    @classmethod
    def ensure_collection(cls, collection_name: str, opt_config: dict[str, Any]) -> None:
        if cls.client.collection_exists(collection_name):
            return

        vector_size = opt_config["vector_size"][opt_config["embedding_model"]]
        search_mode = opt_config["search_mode"]
        query_mode = opt_config["query_mode"]
        multi_search = opt_config["multi_search"]

        if search_mode == "dense_sparse":
            cls.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(size=vector_size, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    ),
                },
            )
        elif search_mode == "dense_sparse_colbert":
            cls.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(size=vector_size, distance=Distance.COSINE),
                    "colbert": VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                        multivector_config=MultiVectorConfig(
                            comparator=MultiVectorComparator.MAX_SIM,
                        )
                    ),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    ),
                },
            )
        elif query_mode == "multi":
            multi_vector_size = (len(multi_search) + 1) * vector_size
            cls.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "multi": VectorParams(size=multi_vector_size, distance=Distance.COSINE)
                }
            )
        else:
            cls.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(size=vector_size, distance=Distance.COSINE)
                }
            )
