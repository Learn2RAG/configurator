import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, MultiVectorComparator, MultiVectorConfig


class Qdrant:
    client = QdrantClient(host="localhost", port=os.environ.get('QDRANT__SERVICE__HTTP_PORT', 6336))

    def __init__(self, collection_name, vector_size, search_mode):
        self.collection_name = collection_name
        self.vector_size = vector_size

        if search_mode == "dense_sparse":
            if not Qdrant.client.collection_exists(self.collection_name):
                Qdrant.client.create_collection(
                    collection_name=self.collection_name,
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
            if not Qdrant.client.collection_exists(self.collection_name):
                Qdrant.client.create_collection(
                    collection_name=self.collection_name,
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

        else:
            if not Qdrant.client.collection_exists(self.collection_name):
                Qdrant.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(size=vector_size, distance=Distance.COSINE)
                    }
                )
