import os
from typing import Any
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, MultiVectorComparator, MultiVectorConfig
from qdrant_client.http.exceptions import UnexpectedResponse

from .config import user_config

# FIXME: when running a Windows package,
# this import causes segmentation fault if done after creating QdrantClient.
# Importing it here prevents this situation...
__import__('FlagEmbedding')


class Qdrant:

    _client = None
    def __init__(self, collection_name: str, opt_config: dict[str, Any]) -> None:
        self.collection_name = collection_name
        self.vector_size = opt_config["vector_size"][opt_config["embedding_model"]]
        self.search_mode = opt_config["search_mode"]
        self.query_mode = opt_config["query_mode"]
        self.multi_search = opt_config["multi_search"]

    @classmethod
    def get_client(cls) -> QdrantClient:
        """Lazy initialization of the QdrantClient."""
        logging.debug("Lazy initialization of QdrantClient")
        if cls._client is None:
            logging.debug("make new one")
            api_key = os.environ.get('QDRANT__SERVICE__API_KEY')
            path = os.environ.get('QDRANT_PATH') or None
            location = None if path else os.environ.get('QDRANT_LOCATION', 'http://localhost:6336')

            cls._client = QdrantClient(
                location=location,
                api_key=api_key,
                path=path,
            )
        return cls._client

    @classmethod
    def ensure_collection(cls, collection_name: str, opt_config: dict[str, Any]) -> None:

        client = cls.get_client()
        logging.info("lets make sure the collection exists")
        logging.debug(f" collection_name : {collection_name} opt_config:{opt_config.keys()}")
        if client.collection_exists(collection_name):
            logging.info("collection already exists")
            return
        logging.debug("creating collection")
        vector_size = opt_config["vector_size"][opt_config["embedding_model"]]
        search_mode = opt_config["search_mode"]
        query_mode = opt_config["query_mode"]
        multi_search = opt_config["multi_search"]
        try:
            if search_mode == "dense_sparse":
                cls.get_client().create_collection(
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
                cls.get_client().create_collection(
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
                cls.get_client().create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "multi": VectorParams(size=multi_vector_size, distance=Distance.COSINE)
                    }
                )
            else:
                cls.get_client().create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "dense": VectorParams(size=vector_size, distance=Distance.COSINE)
                    }
                )
        except UnexpectedResponse as e:
            # Handle the race condition: Another worker created it between our check and our attempt
            if e.status_code == 409 or "already exists" in str(e):
                logging.debug("Collection %s was created by another process.", collection_name)
            else:
                raise e
        except ValueError as e:
            # Some older Qdrant client versions throw ValueError instead of UnexpectedResponse
            if "already exists" in str(e):
                logging.debug("Collection %s was created by another process.", collection_name)
            else:
                raise e