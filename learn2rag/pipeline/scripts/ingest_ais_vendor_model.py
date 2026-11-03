"""
Preprocess an AIS table (one document per row) and ingest it with the regular pipeline
in query_mode 'named'.

Named vectors: dense/sparse (content), dense_vendor/sparse_vendor, dense_model/sparse_model.

Datasets:
  hardware  AIS_seed_2026-10-23_Hardware Models.xlsx           -> AIS_vendor_model
  catalog   catalog_asset_identity_with_platforms.csv          -> AIS_catalog_asset_identity_with_platforms
            (exact duplicate rows are removed before ingestion; content vectors are named
            dense_model-plattform/sparse_model-plattform; sparse vectors from lowercased text)

Run from the repo root:
    QDRANT_LOCATION=http://usdev001:6336 python -m learn2rag.pipeline.scripts.ingest_ais_vendor_model [hardware|catalog]
"""
import argparse
import copy
import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.documents import Document

from .. import ingestion
from ..config import opt_config

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "AIS"

DATASETS: dict[str, dict[str, Any]] = {
    "hardware": {
        "file": DATA_DIR / "AIS_seed_2026-10-23_Hardware Models.xlsx",
        "collection_name": "AIS_vendor_model",
        "loader_id": "AIS_hardware_models",
        "drop_duplicates": False,
        # metadata key -> source column
        "columns": {
            "vendor": "Vendor normalized",
            "model": "Model normalized",
        },
        "content_fields": ["vendor", "model"],
    },
    "catalog": {
        "file": DATA_DIR / "catalog_asset_identity_with_platforms.csv",
        "collection_name": "AIS_catalog_asset_identity_with_platforms",
        "loader_id": "AIS_software_catalog",
        "drop_duplicates": True,
        "columns": {
            "vendor": "Manufacturer",
            "model": "Product name",
            "product_family": "Product family",
            "version": "Version",
            "edition": "Edition",
            "language": "Language",
            "platform": "Platform",
        },
        # product name plus the columns that distinguish otherwise identical products
        "content_fields": ["model", "version", "language", "platform"],
        # dataset specific opt_config overrides
        "opt_config": {
            "named_content_vector": "model-plattform",
            "sparse_lowercase": True,
        },
    },
}

OPT_CONFIG = copy.deepcopy(opt_config) | {
    "embedding_model": "BAAI/bge-m3",
    "search_mode": "dense_sparse",
    "query_mode": "named",
    "multi_search": ["vendor", "model"],
    "ingestion_batch_size": 1024,
}


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".xlsx":
        return pd.read_excel(path, dtype=str).fillna("")
    return pd.read_csv(path, sep=";", dtype=str, keep_default_na=False)


def load_documents(dataset: dict[str, Any]) -> list[Document]:
    df = read_table(dataset["file"])
    if dataset["drop_duplicates"]:
        n_rows = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        logging.info("Removed %d exact duplicate rows, %d rows left", n_rows - len(df), len(df))

    documents = []
    for i, row in df.iterrows():
        meta = {key: row[column].strip() for key, column in dataset["columns"].items()}
        content = " ".join(meta[key] for key in dataset["content_fields"] if meta[key])
        documents.append(Document(
            page_content=content,
            metadata={
                "source": f"{dataset['file'].name}#row{i}",
                "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "loader_id": dataset["loader_id"],
                "document_id": str(i),
                "title": content,
            } | meta,
        ))
    return documents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", choices=DATASETS, default="hardware")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    dataset = DATASETS[args.dataset]
    documents = load_documents(dataset)
    logging.info("Loaded %d documents from %s", len(documents), dataset["file"].name)
    ingestion.index(documents, {"collection_name": dataset["collection_name"]}, OPT_CONFIG | dataset.get("opt_config", {}))
