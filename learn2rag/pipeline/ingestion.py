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


from . import loaders
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


def point_exists(qdrant: Qdrant, collection_name: str, path: str, chunk_hash:str) -> bool:
    filter = Filter(
        must=[
            FieldCondition(key="path", match=MatchValue(value=path)),
            FieldCondition(key="content_hash", match=MatchValue(value=chunk_hash)),
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
        "content_hash": sample["chunk_hash"],
        "title": sample["metadata"].get("title",""),
        "uri": sample["metadata"].get("uri",""),
        "loader_id": sample["metadata"]["loader_id"],
        "document_id": sample["metadata"].get("document_id", "")
    }

def index(user_config: dict[str, Any], opt_config: dict[str, Any]) -> None:
    # TODO: enable list of file paths in loader and adapt user_config
    # Load the documents from pdf
    # all_documents = loaders.sync_pdf_loader(user_config["file_path"])
    # TODO: use ifdt loader to load pdf in json, then:
    logging.info('Loading documents')
    all_documents = loaders.json_loader(user_config['imported_documents_file_path'])

    # Split documents into chunks
    logging.info('Splitting documents into chunks')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=opt_config["chunk_size"], chunk_overlap=opt_config["chunk_overlap"]
    )
    chunks = text_splitter.split_documents(all_documents)

    collection_name = user_config["collection_name"]

    # Init vector store
    qdrant = Qdrant(
        collection_name=collection_name,
        opt_config=opt_config
    )


    chunks_content = [chunk.page_content for chunk in chunks]
    if len(opt_config["multi_search"]) > 0 and opt_config["query_mode"] == "multi":
        chunks_metadata =  {}
        embeddings_metadata = {}
        for item in opt_config["multi_search"]:
            chunks_metadata[item] = list(get_chunks_metadata(chunks, item))
            embeddings_metadata[item] = create_embeddings(chunks_metadata[item], opt_config["embedding_model"], opt_config["search_mode"])
            dense_vecs = embeddings_metadata[item]["dense_vecs"]
            if isinstance(dense_vecs, np.ndarray):
                assert dense_vecs.ndim == 2, dense_vecs.shape
            else:
                raise TypeError(f"dense_vecs must be np.ndarray, got {type(dense_vecs)}")
                
    # TODO: hash if you want to monitore changes in metadata
    chunk_hash = [hashlib.md5(chunk.page_content.encode()).hexdigest() for chunk in chunks]
    # Todo: handle different vector lengths for batch encoding when using sparse vectors

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
                    chunk_hash,
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
                    chunk_hash,
                )
            ]
    else:
        chunks_with_embeddings = [
            dict(chunk) | {"dense_vec": dense, "chunk_hash": c_hash}
            for chunk, dense, c_hash in zip(chunks, embeddings, chunk_hash)
        ]

    for sample in chunks_with_embeddings:
        if not point_exists(qdrant, collection_name, sample['metadata']['source'], sample['chunk_hash']):
            if opt_config["search_mode"] == "dense_sparse":
                insert_dense_sparse(qdrant, collection_name, sample)
            elif opt_config["search_mode"] == "dense_sparse_colbert":
                insert_dense_sparse_colbert(qdrant, collection_name, sample)
            elif opt_config["query_mode"] == "multi":
                insert_multi(qdrant, collection_name, sample)
            else:
                insert(qdrant, collection_name, sample)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    from .config import user_config, opt_config
    index(user_config, opt_config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
