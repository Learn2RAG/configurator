from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore


class Qdrant:
    qdrant = QdrantClient(host="localhost", port=6336)

    def __init__(self, collection_name, encoder, vector_size):
        self.collection_name = collection_name
        self.encoder = encoder
        self.vector_size = vector_size

        if not Qdrant.qdrant.collection_exists(self.collection_name):
            Qdrant.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )

        self.vector_store = QdrantVectorStore(
            client=Qdrant.qdrant,
            collection_name=self.collection_name,
            embedding=self.encoder,
        )
