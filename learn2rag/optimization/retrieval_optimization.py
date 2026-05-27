"""
RAG Retrieval Optimization.
"""

import argparse
import json
import logging
import pathlib
import time
import copy
import os
import asyncio
from typing import Dict, Any, List, Union, Tuple

import numpy as np
from bert_score import score as bert_score
from ConfigSpace import ConfigurationSpace, Integer, Categorical, ForbiddenGreaterThanRelation, Configuration, \
    ForbiddenAndConjunction, ForbiddenEqualsClause
from smac import HyperparameterOptimizationFacade, Scenario

from learn2rag.evaluation.tools import read_dataset_qa
from learn2rag.pipeline.config import opt_config
import learn2rag.pipeline.search
import learn2rag.pipeline.generate




def load_registry(path: str = "registry.json") -> dict:
    p = pathlib.Path(path)
    if not p.is_file():
       logging.error("registry file not found")
    with p.open() as f:
        return json.load(f)

def run_search(question: str, user_config: Dict[str, Any], working_config: Dict[str, Any]) -> Tuple[List[Any], float]:
    t0 = time.time()
    # docs = learn2rag.pipeline.search.search(question, user_config, working_config)
    # TODO user config/opt config
    docs = asyncio.run(learn2rag.pipeline.search.search_authorized(question, user="anonymous", request_id=None, user_config=user_config, opt_config=working_config))
    search_time = time.time() - t0
    source_list = [point.payload['source'] for point in docs]
    return source_list, search_time


def recall(search_results, labels):
    count = 0
    top_k = opt_config["top_k"]
    for q in range(len(search_results)):
        label = str(labels[q])
        hits = [str(h) for h in search_results[q]]
        print('label ',label, ' hits: ', hits)
        if label in hits[:top_k]:
            count += 1
    return count / len(labels) if labels else 0.0


#I removed seed because there are no use for it
# removed dataset_name because it just use in user config and now we inject it
def objective(config: Configuration,
    questions: List[Dict[str, Any]],
    dataset_name: str,
    state: Dict[str, Any],
    answers_dir: pathlib.Path
    ,prompt_map
) -> float:
    state["trial_count"] += 1
    tid = state["trial_count"]
    cfg = dict(config)
    logging.info(f"Trial {tid}: {cfg}")

    if cfg["rewrite"] == "False" and cfg["rewrite_mode"] in {"keywords", "subqueries", "subqueries_keywords"}:
        logging.warning(f"Skip invalid cfg: {cfg}")
        return 1.0

    if cfg["reranking"] == "False" and cfg["reranking_mode"] in {"reranking_with_flagreranker", "reranking_with_sentence_transformers", "reranking_with_colbert"}:
        logging.warning(f"Skip invalid cfg: {cfg}")
        return 1.0

    if cfg["search_mode"] in {"dense", "sparse"} and cfg["fusion_mode"] in {"DBSF", "RRF"}:
        logging.warning(f"Skip invalid cfg: {cfg}")
        return 1.0


# TODO
    working_cfg = copy.deepcopy(opt_config)
    working_cfg.update({
        "chunk_size": cfg["chunk_size"],
        "chunk_overlap": cfg["chunk_overlap"],
        "search_mode": cfg["search_mode"],
        "reranking_mode": cfg["reranking_mode"],
        "rewrite_mode": cfg["rewrite_mode"],
        "fusion_mode": cfg["fusion_mode"],
        "reranking": cfg["reranking"],
        "rewrite": cfg["rewrite"]
    })

# TODO
    ucfg = {
        "file_path": None,
        "collection_name": "CSC-CS_2000-CO_50", # TODO get collection name depending on hyperparameters
        "imported_documents_file_path": None,
        "llm": None,
    }
    env_user_cfg = os.environ.get("PIPELINE_USER_CONFIG")
    if env_user_cfg and pathlib.Path(env_user_cfg).exists():
        ucfg.update(json.loads(pathlib.Path(env_user_cfg).read_text()))

    predictions, goldens = [], []
    qa_pairs = []
    t_start = time.time()
    t_search = 0.0

    for q in questions:
        # Preserve the original “skip empty question” guard
        if not q.get("question"):
            continue
        try:
            source_list, t_s = run_search(q["question"], ucfg, working_cfg)
            t_search += t_s
            predictions.append(source_list)
            goldens.append(q["ground_truth"])
            qa_pairs.append({**q, "retrieved_sources": source_list})
        except Exception as e:
            # Same behaviour as the old version: record a blank answer. TODO : check if we need this
            logging.warning(f"Trial {tid}, q{q.get('id','?')} failed: {e}")
            predictions.append([])
            goldens.append(q["ground_truth"])
            qa_pairs.append({**q, "generated_answer": "", "retrieved_context": ""})

    if not predictions:
        return 1.0

    t_score = time.time()
    recall_score = recall(predictions, goldens)
    scoring_time = time.time() - t_score

    # objective function
    cost = 1.0 - recall_score
    total_time = time.time() - t_start

    trial_answers = {
        "trial_id": tid,
        "config": cfg,
        "cost": float(cost),
        "avg_recall": float(recall_score),
        "qa_pairs": qa_pairs,
    }
    answers_file = answers_dir / f"trial_{tid}_answers.json"
    with open(answers_file, "w") as f:
        json.dump(trial_answers, f, indent=2, default=str)

    state["best_cost"] = min(state["best_cost"], cost)
    state["convergence"].append({"trial": tid, "cost": float(cost), "best_cost": float(state["best_cost"])})
    state["history"].append({
        "trial_id": tid, "config": cfg,
        "avg_recall": float(recall_score),
        "cost": float(cost), "time_s": round(total_time, 2),
        "search_s": round(t_search, 2),
        "scoring_s": round(scoring_time, 2),
    })

    logging.info(
        f"Trial {tid}: recall={recall_score:.4f} cost={cost:.4f} "
        f"time={total_time:.1f}s (search={t_search:.1f} score={scoring_time:.1f})"
    )
    return float(cost)


def param_importance(smac: HyperparameterOptimizationFacade, output_path: pathlib.Path) -> Dict[str, Any]:
    params = list(smac.scenario.configspace.keys())
    configs, costs = [], []
    for key, val in smac.runhistory.items():
        configs.append(dict(smac.runhistory.get_config(key.config_id)))
        costs.append(val.cost)
    if len(configs) < 3:
        return {}

    raw = {}
    for p in params:
        groups = {}
        for c, cost in zip(configs, np.array(costs)):
            groups.setdefault(str(c[p]), []).append(cost)
        means = [np.mean(g) for g in groups.values()]
        raw[p] = float(np.var(means)) if len(means) > 1 else 0.0

    total = sum(raw.values())
    imp = {p: round(v / total, 4) for p, v in raw.items()} if total > 0 else raw
    ranking = sorted(imp, key=imp.get, reverse=True)
    result = {"method": "variance_based", "ranking": ranking, "individual": imp}
    with open(output_path / "parameter_importance.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def run(dataset_name: str, max_questions: int, n_trials: int, output_dir: Union[str, pathlib.Path], registry_path:str) -> Tuple[
        Dict[str, Any], List[Any], Dict[str, Any]]:
    registry = load_registry(registry_path)
    datasets = registry["datasets"]
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(datasets.keys())}")
    dcfg = datasets[dataset_name]
    fields = dcfg["fields"]

    out = pathlib.Path(output_dir) / dataset_name
    out.mkdir(parents=True, exist_ok=True)
    answers_dir = out / "trial_answers"
    answers_dir.mkdir(parents=True, exist_ok=True)

    qa = read_dataset_qa(dataset_name, dcfg["subdirectory"], dcfg["split"])
    if max_questions:
        qa = qa.select(range(min(max_questions, len(qa))))

    questions = [
        {
            "question": r[fields["q"]],
            "ground_truth": r[fields["a"]],
            "id": r.get(fields["id"], str(i)),
        }
        for i, r in enumerate(qa)
    ]
    prompt_map = registry["prompts"]

    cs = ConfigurationSpace(seed=42)
    cs.add([
        # Categorical("chunk_size", [250, 1000, 2000], default=1000),
        # Categorical("chunk_overlap", [50, 200], default=50),
        Categorical("chunk_size", [2000], default=2000),
        Categorical("chunk_overlap", [50], default=50),
        Categorical("search_mode", ["dense", "sparse", "dense_sparse", "dense_sparse_colbert"], default="dense"),
        Categorical("reranking_mode", ["none", "reranking_with_flagreranker", "reranking_with_sentence_transformers", "reranking_with_colbert"], default="none"),
        Categorical("rewrite_mode", ["none", "subqueries", "keywords", "subqueries_keywords"], default="none"),
        Categorical("fusion_mode", ["none", "DBSF", "RRF"], default="none"),
        Categorical("reranking", ["True", "False"], default="False"),
        Categorical("rewrite", ["True", "False"], default="False"),
    ])
    cs.add(ForbiddenGreaterThanRelation(cs["chunk_overlap"], cs["chunk_size"]))

    for sm in ["dense", "sparse"]:
        for fm in ["DBSF", "RRF"]:
            cs.add(ForbiddenAndConjunction(
                ForbiddenEqualsClause(cs["search_mode"], sm),
                ForbiddenEqualsClause(cs["fusion_mode"], fm),
            ))

    for sm in ["dense_sparse", "dense_sparse_colbert"]:
        cs.add(ForbiddenAndConjunction(
            ForbiddenEqualsClause(cs["search_mode"], sm),
            ForbiddenEqualsClause(cs["fusion_mode"], "none"),
        ))


    for rrm in ["reranking_with_flagreranker", "reranking_with_sentence_transformers", "reranking_with_colbert"]:
        cs.add(ForbiddenAndConjunction(
            ForbiddenEqualsClause(cs["reranking"], "False"),
            ForbiddenEqualsClause(cs["reranking_mode"], rrm),
        ))

    cs.add(ForbiddenAndConjunction(
        ForbiddenEqualsClause(cs["reranking"], "True"),
        ForbiddenEqualsClause(cs["reranking_mode"], "none"),
    ))

    for rwm in ["keywords", "subqueries", "subqueries_keywords"]:
        cs.add(ForbiddenAndConjunction(
            ForbiddenEqualsClause(cs["rewrite"], "False"),
            ForbiddenEqualsClause(cs["rewrite_mode"], rwm),
        ))

    cs.add(ForbiddenAndConjunction(
        ForbiddenEqualsClause(cs["rewrite"], "True"),
        ForbiddenEqualsClause(cs["rewrite_mode"], "none"),
    ))


    scenario = Scenario(
        cs,
        deterministic=True,
        n_trials=n_trials,
        walltime_limit=7200,
        seed=42,
        output_directory=out / "smac_output",
    )
    state: Dict[str, Any] = {"trial_count": 0, "best_cost": 1.0, "convergence": [], "history": []}

    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=lambda config, seed=0: objective(config, questions, dataset_name, state, answers_dir,prompt_map)
    )
    t0 = time.time()
    incumbent = smac.optimize()
    importance = param_importance(smac, out)
    total_time = time.time() - t0
    best_cfg = incumbent.get_dictionary()
    results_path = out / "optimization_results.json"
    results_path.write_text(json.dumps({
        "best_config": best_cfg,
        "run_history": state["history"],
        "convergence": state["convergence"],
        "parameter_importance": importance,
        "total_time_s": round(total_time, 2),
        "dataset": dataset_name,
        "metric": "recall",
        "answers_dir":str(answers_dir),
    }, indent=2, default = str))

    return best_cfg, state["history"], importance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs='?', default="learn2rag.optimization")
    parser.add_argument("--dataset", type=str, default="WikiEval")
    parser.add_argument("--max_questions", type=int, default=50)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--logging-config", type=str)
    parser.add_argument("--registry", type=str, default="registry.json")
    parser.add_argument("--output_dir", type=str, default="optimization_results_baseline")
    args, _ = parser.parse_known_args()

    final_output_dir = pathlib.Path(args.output_dir)

    env_out = os.environ.get("PIPELINE_OPT_CONFIG")
    if not final_output_dir.exists() and env_out:
        final_output_dir = pathlib.Path(env_out).parent

    incumbent, history, importance = run(args.dataset, args.max_questions, args.n_trials, final_output_dir, args.registry)

    # incumbent, history, importance = run(
    #     args.dataset, args.max_questions, args.n_trials, args.output_dir,
    # )

    best = min(history, key=lambda x: x["cost"])
    print(f"\nBest config: {dict(incumbent)}")
    # print(f"BERTScore (golden): {best['avg_bertscore_golden']:.4f}")
    if importance:
        print(f"\nParameter importance:")
        for i, p in enumerate(importance["ranking"], 1):
            print(f"  {i}. {p}: {importance['individual'][p]:.4f}")