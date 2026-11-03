"""
Grid search over all combinations of named Qdrant vectors for license clearing.

Collection vectors (pre-computed):
  dense / sparse
  dense_article_number / sparse_article_number
  dense_manufacturer   / sparse_manufacturer
  dense_description    / sparse_description

Run:
    QDRANT_LOCATION=http://usdev001:30333 python -m learn2rag.optimization.multi_search_optimization.optimize_multi_search
"""
import json
import itertools
from copy import deepcopy
from pathlib import Path

import pandas as pd

from learn2rag.pipeline.search import search_named_hybrid_preembedded, _rerank_points_with_flagreranker
from learn2rag.pipeline.embeddings import create_embeddings
from learn2rag.pipeline.config import user_config

EVAL_CSV   = Path(__file__).parent / "eval_data" / "multi_search_eval.csv"
OUTPUT_CSV = Path(__file__).parent / "results.csv"

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

BASE_OPT_CONFIG = {
    "embedding_model": "BAAI/bge-m3",
    "vector_size": {
        "sentence-transformers/all-mpnet-base-v2": 768,
        "BAAI/bge-m3": 1024,
    },
    "prefetch_limit_dense":  25,
    "prefetch_limit_sparse": 25,
    "fusion_mode": "RRF",
    "top_k": 10,
    "top_k_reranker": 5,
    "query_mode": "single",
    "multi_search": [],
    "reranking": "False",
    "reranking_mode": "",
    "rewrite": "False",
}


TOP_K_VALUES          = [1] #[5, 10]
TOP_K_RERANKER_VALUES = [1] #[3, 5, 10]

FUSION_MODES          = ["RRF", "DBSF"]

FIELD_VECTOR_NAMES: dict[str, tuple[str, str]] = {
    "all":            ("dense",                "sparse"),
    "article_number": ("dense_article_number", "sparse_article_number"),
    "manufacturer":   ("dense_manufacturer",   "sparse_manufacturer"),
    "description":    ("dense_description",    "sparse_description"),
}
FIELDS      = list(FIELD_VECTOR_NAMES.keys())
FIELD_MODES = ["none", "dense", "sparse", "both"]


def _safe(value) -> str:
    return str(value).strip() if pd.notna(value) else ""


def load_eval_data() -> tuple[list[dict[str, str]], list[str]]:
    df = pd.read_csv(EVAL_CSV, sep=";")
    df = df[df.iloc[:, 0].str.strip() == "eval"].reset_index(drop=True)

    queries, labels = [], []
    for _, row in df.iterrows():
        manufacturer   = _safe(row.get("Manufacturer (Article)"))
        article_number = _safe(row.get("Manufacturer article number"))
        description    = _safe(row.get("Article description"))
        queries.append({
            "all":            f"{manufacturer} {article_number} {description}".strip(),
            "article_number": article_number,
            "manufacturer":   manufacturer,
            "description":    description,
        })
        labels.append(_safe(row.get("Manufacturer article number (identified)")))

    return queries, labels


def _article_number_from_point(point) -> str:
    try:
        return json.loads(point.payload.get("content", "{}")).get("Article Number", "")
    except Exception:
        return ""


def recall_at_k(hits_list: list[list], labels: list[str], top_k: int) -> float:
    if not labels:
        return 0.0
    return sum(
        1 for hits, label in zip(hits_list, labels)
        if label and label in [_article_number_from_point(p) for p in hits[:top_k]]
    ) / len(labels)


def build_field_vectors(config: dict[str, str]) -> dict[str, list[str]]:
    field_vectors: dict[str, list[str]] = {}
    for field, mode in config.items():
        dense_name, sparse_name = FIELD_VECTOR_NAMES[field]
        if mode == "dense":
            field_vectors[field] = [dense_name]
        elif mode == "sparse":
            field_vectors[field] = [sparse_name]
        elif mode == "both":
            field_vectors[field] = [dense_name, sparse_name]
        # "none" -> field omitted
    return field_vectors


def main() -> None:
    print("Loading eval data...")
    queries, labels = load_eval_data()
    print(f"Eval rows: {len(queries)}  (labels found: {sum(bool(l) for l in labels)})")

    all_configs = [
        dict(zip(FIELDS, combo))
        for combo in itertools.product(FIELD_MODES, repeat=len(FIELDS))
        if any(m != "none" for m in combo)
    ]

    n_results = len(all_configs) * len(FUSION_MODES) * (len(TOP_K_VALUES) + len(TOP_K_RERANKER_VALUES))
    print(
        f"Configs: {len(all_configs)}"
        f"  ×  {len(FUSION_MODES)} fusion modes"
        f"  ×  ({len(TOP_K_VALUES)} top_k plain  +  {len(TOP_K_RERANKER_VALUES)} top_k_reranker with reranker)"
        f"  =  {n_results} result rows\n"
    )

    print("Embedding queries (once)...")
    # Pre-compute all field embeddings once – reused across all configs
    per_row_embeddings: list[dict[str, dict]] = []
    for query in queries:
        enc = create_embeddings(
            list(query.values()),
            BASE_OPT_CONFIG["embedding_model"],
            "dense_sparse",
        )
        per_row_embeddings.append({
            field: {
                "dense_vecs":      enc["dense_vecs"][i],
                "lexical_weights": dict(enc["lexical_weights"][i]),
            }
            for i, field in enumerate(query.keys())
        })
    print("Embeddings ready.\n")

    records: list[dict] = []

    total_configs = len(all_configs) * len(FUSION_MODES)
    run = 0
    for config in all_configs:
        field_vectors = build_field_vectors(config)
        n_active_vectors = sum(len(vnames) for vnames in field_vectors.values())
        reranker_fetch = n_active_vectors * BASE_OPT_CONFIG["prefetch_limit_dense"]

        for fusion_mode in FUSION_MODES:
            run += 1

            opt_config_plain = deepcopy(BASE_OPT_CONFIG)
            opt_config_plain["fusion_mode"] = fusion_mode
            opt_config_plain["top_k"] = max(TOP_K_VALUES)

            hits_plain = [
                search_named_hybrid_preembedded(emb, field_vectors, user_config, opt_config_plain)
                for i, emb in enumerate(per_row_embeddings, start=1)
                if not print(f"  [config {run}/{total_configs}] plain   {fusion_mode} sample {i}/{len(queries)}", end="\r", flush=True)
            ]

            for top_k in TOP_K_VALUES:
                records.append({
                    **config,
                    "fusion_mode":    fusion_mode,
                    "reranker":       False,
                    "top_k":          top_k,
                    "top_k_reranker": None,
                    "recall":         recall_at_k(hits_plain, labels, top_k),
                })

            opt_config_reranker = deepcopy(BASE_OPT_CONFIG)
            opt_config_reranker["fusion_mode"] = fusion_mode
            opt_config_reranker["top_k"] = reranker_fetch

            hits_for_reranking = [
                search_named_hybrid_preembedded(emb, field_vectors, user_config, opt_config_reranker)
                for i, emb in enumerate(per_row_embeddings, start=1)
                if not print(f"  [config {run}/{total_configs}] reranker {fusion_mode} sample {i}/{len(queries)}", end="\r", flush=True)
            ]

            for top_k_reranker in TOP_K_RERANKER_VALUES:
                opt_config_reranker["top_k_reranker"] = top_k_reranker
                hits_reranked = [
                    _rerank_points_with_flagreranker(
                        query["all"],
                        hits,
                        top_k=opt_config_reranker["top_k_reranker"],
                        model_name=RERANKER_MODEL,
                    )
                    for query, hits in zip(queries, hits_for_reranking)
                ]
                records.append({
                    **config,
                    "fusion_mode":    fusion_mode,
                    "reranker":       True,
                    "top_k":          None,
                    "top_k_reranker": top_k_reranker,
                    "recall":         recall_at_k(hits_reranked, labels, top_k_reranker),
                })

        if run % 50 == 0 or run == total_configs:
            best = max(r["recall"] for r in records)
            print(f"  [{run:>3}/{len(all_configs)}] best recall so far: {best:.3f}")

    results_df = (
        pd.DataFrame(records)
        .sort_values("recall", ascending=False)
        .reset_index(drop=True)
    )
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")

    print("\nTop 10 configurations:")
    print(results_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()