import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, MultiVectorComparator, MultiVectorConfig

from .config import user_config


class Qdrant:
    client = QdrantClient(
        host="localhost",
        port=int(os.environ.get('QDRANT__SERVICE__HTTP_PORT', 6336)),
        api_key=os.environ.get('QDRANT__SERVICE__API_KEY'),
        https=False,
    )

    def __init__(self, collection_name: str, opt_config: dict[str, Any]) -> None:
        self.collection_name = collection_name
        self.vector_size = opt_config["vector_size"][opt_config["embedding_model"]]
        self.search_mode = opt_config["search_mode"]
        self.query_mode = opt_config["query_mode"]
        self.multi_search = opt_config["multi_search"]

        if self.search_mode == "dense_sparse":
            if not Qdrant.client.collection_exists(self.collection_name):
                Qdrant.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(size=self.vector_size, distance=Distance.COSINE)
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(
                            index=SparseIndexParams(on_disk=False)
                        ),
                    },
                )
        elif self.search_mode == "dense_sparse_colbert":
            if not Qdrant.client.collection_exists(self.collection_name):
                Qdrant.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(size=self.vector_size, distance=Distance.COSINE),
                        "colbert": VectorParams(
                            size=self.vector_size,
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
        elif self.query_mode == "multi":
            if not Qdrant.client.collection_exists(self.collection_name):
                vector_size = (len(self.multi_search)+1)*self.vector_size
                Qdrant.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "multi": VectorParams(size=vector_size, distance=Distance.COSINE)
                    }
                )


        else:
            if not Qdrant.client.collection_exists(self.collection_name):
                Qdrant.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(size=self.vector_size, distance=Distance.COSINE)
                    }
                )
