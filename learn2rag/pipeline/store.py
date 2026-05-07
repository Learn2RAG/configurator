import logging
from typing import Any

from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector
from langchain_core.documents.base import Document

from learn2rag.pipeline.ingestion import index
from learn2rag.pipeline.qdrant import Qdrant

def delete_collection(loader_id: str|None, user_config: dict[str, Any], opt_config: dict[str, Any]) -> None:
    """Delete a collection from the vector store or a subset of points based on loader_id."""
    qdrant = Qdrant(user_config["collection_name"], opt_config)
    if qdrant.client.collection_exists(user_config["collection_name"]):
        if loader_id is None:
            logging.info('Deleting entire collection: %s', user_config["collection_name"])
            qdrant.client.delete_collection(collection_name=user_config["collection_name"])
            return
        else:
            # Delete points with the specified loader_id
            logging.info('Deleting points with loader_id: %s from collection: %s', loader_id, user_config["collection_name"])
            qdrant.client.delete(
                collection_name=user_config["collection_name"],
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[FieldCondition(
                                key="loader_id",
                                match=MatchValue(value=loader_id),
                            ),
                        ],
                    )
                ),
            )

def delete_documents(loader_id: str, paths: list[str], user_config: dict[str, Any], opt_config: dict[str, Any]) -> None:
    """Delete documents from the vector store based on loader_id and paths."""
    qdrant = Qdrant(user_config["collection_name"], opt_config)
    if qdrant.client.collection_exists(user_config["collection_name"]):
        logging.info('Deleting documents with loader_id: %s and paths: %s', loader_id, paths)
        # Delete points with the specified loader_id and paths
        for path in paths:
            qdrant.client.delete(
                collection_name=user_config["collection_name"],
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="loader_id",
                                match=MatchValue(value=loader_id),
                            ),
                            FieldCondition(
                                key="source",
                                match=MatchValue(value=path)
                            ),
                        ],
                    )
                ),
            )

def get_documents(loader_id: str, user_config: dict[str, Any], opt_config: dict[str, Any]) -> dict[str, str]:
    """Retrieve documents from the vector store and return a {source: content_hash} mapping."""
    qdrant = Qdrant(user_config["collection_name"], opt_config)
    path_hash_dict: dict[str, str] = {}
    if qdrant.client.collection_exists(user_config["collection_name"]):
        logging.info('Scrolling through collection to retrieve documents with loader_id: %s', loader_id)
        filter = Filter(
            must=[
                FieldCondition(
                    key="loader_id",
                    match=MatchValue(value=loader_id)
                )
            ]
        )
        points = []
        offset = None

        while True:
            result = qdrant.client.scroll(
                collection_name=user_config["collection_name"],
                scroll_filter=filter,
                limit=100,
                offset=offset,
                with_payload=["source", "content_hash"],
                with_vectors=False,
            )

            points.extend(result[0])
            offset = result[1]

            if offset is None:
                break

        for point in points:
            if point.payload:
                source = point.payload.get("source", "")
                content_hash = point.payload.get("content_hash", "")
                if source and source not in path_hash_dict:
                    path_hash_dict[source] = content_hash

    return path_hash_dict



def update_documents(loader_id: str, documents: list[Document], user_config: dict[str, Any], opt_config: dict[str, Any]) -> None:
    qdrant = Qdrant(user_config["collection_name"], opt_config)
    if qdrant.client.collection_exists(user_config["collection_name"]):
        logging.info('Updating documents with loader_id: %s', loader_id)
        delete_documents(loader_id, paths=[doc.metadata["source"] for doc in documents], user_config=user_config, opt_config=opt_config)
        index(documents, user_config, opt_config)