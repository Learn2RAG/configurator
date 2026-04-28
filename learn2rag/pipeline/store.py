import logging
from typing import Any

from langchain_core import documents
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, FilterSelector
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
                                key="path",
                                match=MatchValue(value=path)
                            ),
                        ],
                    )
                ),
            )

def get_documents(loader_id: str, user_config: dict[str, Any], opt_config: dict[str, Any]) -> list[dict[str, Any]]|None:
    """Retrieve documents from the vector store based on loader_id."""
    qdrant = Qdrant(user_config["collection_name"], opt_config)
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
                with_payload=True,
                with_vectors=False,
            )

            points.extend(result[0])
            offset = result[1]

            if offset is None:
                break
        return [point.payload for point in points if point.payload is not None]
    return None


def get_document_hashes(loader_id: str, user_config: dict[str, Any], opt_config: dict[str, Any]) -> dict[str, str]:
    """
    Retrieve a path-to-content_hash mapping for all documents belonging to a loader.

    Scrolls the Qdrant collection and deduplicates by source path, keeping the last
    seen content hash per path. Used by the delta-import orchestration to determine
    which documents are new, changed, or deleted.

    Args:
        loader_id (str): Unique loader identifier to filter by.
        user_config (dict[str, Any]): User configuration dict (must contain
                                      ``collection_name``).
        opt_config (dict[str, Any]): Optimisation configuration dict.

    Returns:
        dict[str, str]: Mapping of ``{source_path: content_hash}`` for every document
                        belonging to this loader. Returns an empty dict when the
                        collection does not exist.
    """
    qdrant = Qdrant(user_config["collection_name"], opt_config)
    collection_name = user_config["collection_name"]
    if not qdrant.client.collection_exists(collection_name):
        return {}
    logging.info('Retrieving document hashes for loader_id: %s', loader_id)
    scroll_filter = Filter(
        must=[FieldCondition(key="loader_id", match=MatchValue(value=loader_id))]
    )
    result: dict[str, str] = {}
    offset = None
    while True:
        points, offset = qdrant.client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            if point.payload:
                path = point.payload.get("path", "")
                content_hash = point.payload.get("content_hash", "")
                if path:
                    result[path] = content_hash
        if offset is None:
            break
    return result


def update_documents(loader_id: str, documents: list[Document], user_config: dict[str, Any], opt_config: dict[str, Any]) -> None:
    qdrant = Qdrant(user_config["collection_name"], opt_config)
    if qdrant.client.collection_exists(user_config["collection_name"]):
        logging.info('Updating documents with loader_id: %s', loader_id)
        delete_documents(loader_id, paths=[doc.metadata["source"] for doc in documents], user_config=user_config, opt_config=opt_config)
        index(documents, user_config, opt_config)