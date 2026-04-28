import argparse
import logging
from uuid import uuid4
import hashlib
from typing import Any, cast
import numpy as np
import warnings
from collections.abc import Iterator

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from .qdrant import Qdrant
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue, SparseVector, VectorParams, MultiVectorConfig, MultiVectorComparator, Distance

from .embeddings import create_embeddings


def get_chunks_metadata(chunks: list[Document], item: str) -> Iterator[str]:
    missing = 0
    for chunk in chunks:
        if item in chunk.metadata:
            yield chunk.metadata[item]
        else:
            missing += 1
            yield ''
    if missing != 0:
        logging.warning('%d out of %d chunks are missing "%s" in metadata; using empty string', missing, len(chunks), item)


def point_exists(qdrant: Qdrant, collection_name: str, loader_id: str, path: str, content_hash: str, chunk_hash: str) -> bool:
    filter = Filter(
        must=[
            FieldCondition(key="loader_id", match=MatchValue(value=loader_id)),
            FieldCondition(key="path", match=MatchValue(value=path)),
            FieldCondition(key="content_hash", match=MatchValue(value=content_hash)),
            FieldCondition(key="chunk_hash", match=MatchValue(value=chunk_hash)),
        ]
    )
    result, _ = qdrant.client.scroll(
        collection_name=collection_name, scroll_filter=filter, limit=1
    )
    return len(result) > 0


def insert(qdrant: Qdrant, collection_name: str, sample: dict[str, Any]) -> None:
    point = PointStruct(
        id=uuid4().hex,
        vector={
            "dense": sample["dense_vec"],
        },
        payload=payload(sample),
    )
    qdrant.client.upsert(collection_name=collection_name, wait=True, points=[point])


def insert_dense_sparse(qdrant: Qdrant, collection_name: str, sample: dict[str, Any]) -> None:
    point = PointStruct(
        id=uuid4().hex,
        vector={
            "dense": sample["dense_vec"],
            "sparse": SparseVector(
                indices=[int(x) for x in sample["lexical_weights"].keys()],
                values=sample["lexical_weights"].values(),
            ),
        },
        payload=payload(sample),
    )
    qdrant.client.upsert(collection_name=collection_name, wait=True, points=[point])

def insert_dense_sparse_colbert(qdrant: Qdrant, collection_name: str, sample: dict[str, Any]) -> None:
    point = PointStruct(
        id=uuid4().hex,
        vector={
            "dense": sample["dense_vec"],
            "sparse": SparseVector(
                indices=[int(x) for x in sample["lexical_weights"].keys()],
                values=sample["lexical_weights"].values(),
            ),
            "colbert": sample["colbert_vecs"],
        },
        payload=payload(sample),
    )
    qdrant.client.upsert(collection_name=collection_name, wait=True, points=[point])

def insert_multi(qdrant: Qdrant, collection_name: str, sample: dict[str, Any]) -> None:
    point = PointStruct(
        id=uuid4().hex,
        vector={
            "multi": sample["dense_vec"],
        },
        payload=payload(sample),
    )
    qdrant.client.upsert(collection_name=collection_name, wait=True, points=[point])

def payload(sample: dict[str, Any]) -> dict[str, str]:
    return {
        "content": sample["page_content"],
        "path": sample["metadata"]["source"],
        "content_hash": sample["metadata"]["content_hash"],
        "chunk_hash": sample["chunk_hash"],
        "title": sample["metadata"].get("title",""),
        "uri": sample["metadata"].get("uri",""),
        "loader_id": sample["metadata"]["loader_id"],
        "document_id": sample["metadata"].get("document_id", "")
    }


def ingest_batch(docs: list[Document], qdrant: Qdrant, user_config: dict[str, Any], opt_config: dict[str, Any]) -> None:
    """
    Chunk, embed, and bulk-insert a list of documents into Qdrant.

    Mirrors the behaviour of the original ``index()`` function but accepts an
    already-constructed ``Qdrant`` instance instead of creating one internally.
    Intended for use by ``process_delta_imports`` and other callers that manage
    their own Qdrant connection.

    Points that already exist (identical ``loader_id``, ``path``, ``content_hash``,
    and ``chunk_hash``) are skipped via ``point_exists()``.

    Args:
        docs (list[Document]): Documents to ingest. May be a full initial load or
                               a filtered subset of changed documents.
        qdrant (Qdrant): Authenticated Qdrant wrapper instance.
        user_config (dict[str, Any]): User configuration dict (must contain
                                      ``collection_name``).
        opt_config (dict[str, Any]): Optimisation configuration dict (must contain
                                     ``chunk_size``, ``chunk_overlap``,
                                     ``embedding_model``, and ``search_mode``).
    """
    collection_name = user_config["collection_name"]
    all_documents = docs

    logging.info('Splitting documents into chunks')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=opt_config["chunk_size"], chunk_overlap=opt_config["chunk_overlap"]
    )
    chunks = text_splitter.split_documents(all_documents)

    chunks_content = [chunk.page_content for chunk in chunks]
    if len(opt_config["multi_search"]) > 0 and opt_config["query_mode"] == "multi":
        chunks_metadata: dict[str, list[str]] = {}
        embeddings_metadata: dict[str, Any] = {}
        for item in opt_config["multi_search"]:
            chunks_metadata[item] = list(get_chunks_metadata(chunks, item))
            embeddings_metadata[item] = create_embeddings(chunks_metadata[item], opt_config["embedding_model"], opt_config["search_mode"])
            dense_vecs = embeddings_metadata[item]["dense_vecs"]
            if isinstance(dense_vecs, np.ndarray):
                assert dense_vecs.ndim == 2, dense_vecs.shape
            else:
                raise TypeError(f"dense_vecs must be np.ndarray, got {type(dense_vecs)}")

    chunk_hash = [hashlib.md5(chunk.page_content.encode()).hexdigest() for chunk in chunks]

    logging.info('Creating embeddings...')
    embeddings = create_embeddings(chunks_content, opt_config["embedding_model"], opt_config["search_mode"])
    if len(opt_config["multi_search"]) > 0 and opt_config["query_mode"] == "multi":
        mmembeddings: list[np.ndarray[Any, Any]] = []
        for i in range(len(embeddings['dense_vecs'])):
            vecs_to_concat: list[np.ndarray[Any, Any]] = [cast(np.ndarray[Any, Any], embeddings['dense_vecs'][i])]
            for item in embeddings_metadata.keys():
                vecs_to_concat.append(cast(np.ndarray[Any, Any], embeddings_metadata[item]['dense_vecs'][i]))
            mmembeddings.append(np.concatenate(vecs_to_concat, axis=0))
        embeddings['dense_vecs'] = mmembeddings

    if isinstance(embeddings, dict) and "dense_vecs" in embeddings:
        if opt_config["search_mode"] == "dense":
            chunks_with_embeddings = [
                dict(chunk) | {"dense_vec": dense, "chunk_hash": c_hash}
                for chunk, dense, c_hash in zip(chunks, embeddings["dense_vecs"], chunk_hash)
            ]
        if opt_config["search_mode"] == "dense_sparse":
            chunks_with_embeddings = [
                dict(chunk)
                | {"dense_vec": dense, "lexical_weights": sparse, "chunk_hash": c_hash}
                for chunk, dense, sparse, c_hash in zip(
                    chunks,
                    list(embeddings["dense_vecs"]),
                    list(embeddings["lexical_weights"]),
                    chunk_hash
                )
            ]
        if opt_config["search_mode"] == "dense_sparse_colbert":
            chunks_with_embeddings = [
                dict(chunk)
                | {"dense_vec": dense, "lexical_weights": sparse, "colbert_vecs": colbert, "chunk_hash": c_hash}
                for chunk, dense, sparse, colbert, c_hash in zip(
                    chunks,
                    list(embeddings["dense_vecs"]),
                    list(embeddings["lexical_weights"]),
                    list(embeddings['colbert_vecs']),
                    chunk_hash
                )
            ]
    else:
        chunks_with_embeddings = [
            dict(chunk) | {"dense_vec": dense, "chunk_hash": c_hash}
            for chunk, dense, c_hash in zip(chunks, embeddings, chunk_hash)
        ]

    for sample in chunks_with_embeddings:
        if not point_exists(qdrant, collection_name, sample['metadata']['loader_id'], sample['metadata']['source'], sample['metadata']['content_hash'], sample['chunk_hash']):
            if opt_config["search_mode"] == "dense_sparse":
                insert_dense_sparse(qdrant, collection_name, sample)
            elif opt_config["search_mode"] == "dense_sparse_colbert":
                insert_dense_sparse_colbert(qdrant, collection_name, sample)
            elif opt_config["query_mode"] == "multi":
                insert_multi(qdrant, collection_name, sample)
            else:
                insert(qdrant, collection_name, sample)


def index(documents: list[Document], user_config: dict[str, Any], opt_config: dict[str, Any]) -> None:
    """
    Ingest a list of documents — entry point for standalone pipeline operation.

    Creates a ``Qdrant`` instance internally and delegates to ``ingest_batch()``.
    This function also serves as the replacement for the originally planned
    ``ingest_document()`` helper: a single-document delta upsert is expressed as
    ``index([doc], user_config, opt_config)`` without requiring a separate function.

    Called by ``pipeline/main.py`` and the ``/ingest`` HTTP endpoint. For the
    delta-import path (which manages its own Qdrant connection), use
    ``ingest_batch()`` directly.

    Args:
        documents (list[Document]): One or more documents to ingest.
        user_config (dict[str, Any]): User configuration dict (must contain
                                      ``collection_name``).
        opt_config (dict[str, Any]): Optimisation configuration dict.
    """
    collection_name = user_config["collection_name"]
    qdrant = Qdrant(collection_name=collection_name, opt_config=opt_config)
    ingest_batch(documents, qdrant, user_config, opt_config)