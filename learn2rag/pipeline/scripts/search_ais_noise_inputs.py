"""
Search every input of an AIS input csv against a named-vector collection and write
the top_k hits as columns hit_<rank>_model / hit_<rank>_vendor / hit_<rank>_score.

Without --output the hits of the (single) config are written back into the csv.
With --output <file>.xlsx every config gets its own sheet (named after its vectors, suffix
+dbsf for DBSF fusion, +lower for sparse_lowercase and +rerank for reranking),
plus a column expected_rank (rank of expected_lookup_name in the hits, if present).
An existing xlsx keeps its other sheets; sheets with the same name are replaced.

If opt_config.reranking is "True" (reranking_mode reranking_with_flagreranker), the top_k
search results are reranked and the top_k_reranker hits are written with the reranker score.

Run from the repo root:
    QDRANT_LOCATION=http://usdev001:6336 python -m learn2rag.pipeline.scripts.search_ais_noise_inputs \
        learn2rag/data/AIS/noise-inputs.csv learn2rag/data/AIS/noise-inputs_search_config_2026-10-18.json
    QDRANT_LOCATION=http://usdev001:6336 python -m learn2rag.pipeline.scripts.search_ais_noise_inputs \
        learn2rag/data/AIS/noise-inputs.csv <config> [<config> ...] --output <file>.xlsx
"""
import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..search import embed_named_queries, search_named_hybrid_preembedded, _rerank_points_with_flagreranker


def is_reranking(opt_config: dict[str, Any]) -> bool:
    if opt_config.get("reranking") != "True":
        return False
    if opt_config["reranking_mode"] != "reranking_with_flagreranker":
        raise NotImplementedError(f"reranking_mode {opt_config['reranking_mode']} not supported")
    return True


def search_hits(queries: list[str], embeddings: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    opt_config = config["opt_config"]
    reranking = is_reranking(opt_config)
    n_hits = opt_config["top_k_reranker"] if reranking else opt_config["top_k"]
    hits: dict[str, list[object]] = {}
    for i in range(len(queries)):
        field_embeddings = {config["query_field"]: embeddings[i]}
        points = search_named_hybrid_preembedded(
            field_embeddings, config["field_vectors"], config["user_config"], opt_config
        )
        if reranking:
            points = _rerank_points_with_flagreranker(queries[i], points, top_k=n_hits)
        for rank in range(1, n_hits + 1):
            point = points[rank - 1] if rank <= len(points) else None
            payload = (point.payload or {}) if point else {}
            score = payload["reranking_score"] if reranking and point else (point.score if point else None)
            hits.setdefault(f"hit_{rank}_model", []).append(payload.get("model", ""))
            hits.setdefault(f"hit_{rank}_vendor", []).append(payload.get("vendor", ""))
            hits.setdefault(f"hit_{rank}_score", []).append(round(score, 4) if score is not None else "")
    return pd.DataFrame(hits)


def expected_rank(df: pd.DataFrame) -> list[object]:
    n_hits = sum(1 for c in df.columns if c.startswith("hit_") and c.endswith("_model"))
    ranks: list[object] = []
    for _, row in df.iterrows():
        expected = row["expected_lookup_name"].lower()
        found = [r for r in range(1, n_hits + 1) if str(row[f"hit_{r}_model"]).lower() == expected]
        ranks.append(found[0] if found else "")
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("configs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, help="xlsx file with one sheet per config")
    args = parser.parse_args()
    if args.output is None and len(args.configs) > 1:
        parser.error("multiple configs require --output <file>.xlsx")

    df = pd.read_csv(args.input_csv, sep=";", dtype=str, keep_default_na=False)
    df = df.drop(columns=[c for c in df.columns if c.startswith("hit_")])
    configs = [json.loads(path.read_text(encoding="utf-8")) for path in args.configs]
    queries = df[configs[0]["input_column"]].tolist()

    # embed the queries once per embedding setup, reused across configs
    embeddings_cache: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    results = []
    for config in configs:
        opt_config = config["opt_config"]
        key = (opt_config["embedding_model"], opt_config.get("sparse_lowercase", False))
        if key not in embeddings_cache:
            embeddings_cache[key] = embed_named_queries(queries, opt_config)
        results.append(pd.concat([df, search_hits(queries, embeddings_cache[key], config)], axis=1))

    if args.output is None:
        results[0].to_csv(args.input_csv, sep=";", index=False, lineterminator="\r\n")
        print(f"Wrote hits for {len(df)} inputs to {args.input_csv}")
        return

    append = {"mode": "a", "if_sheet_exists": "replace"} if args.output.exists() else {}
    with pd.ExcelWriter(args.output, engine="openpyxl", **append) as writer:
        for config, result in zip(configs, results):
            if "expected_lookup_name" in result.columns:
                result["expected_rank"] = expected_rank(result)
            vectors = [v for field_vectors in config["field_vectors"].values() for v in field_vectors]
            sheet = "+".join(vectors)
            if len(vectors) > 1 and config["opt_config"]["fusion_mode"] != "RRF":
                sheet += f"+{config['opt_config']['fusion_mode'].lower()}"
            if config["opt_config"].get("sparse_lowercase", False):
                sheet += "+lower"
            if is_reranking(config["opt_config"]):
                sheet += "+rerank"
            result.to_excel(writer, sheet_name=sheet[:31], index=False)
    print(f"Wrote {len(results)} sheets for {len(df)} inputs to {args.output}")


if __name__ == "__main__":
    main()
