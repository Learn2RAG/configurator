"""
RAG Retrieval Optimization.
"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #Select GPU number 0

import argparse
import json
import logging
import pathlib
import time
import copy
import os
import asyncio
from typing import Dict, Any, List, Union, Tuple, cast

import numpy as np
# from bert_score import score as bert_score  # type: ignore[import-not-found]
from ConfigSpace import (# type: ignore[import-not-found]
    ConfigurationSpace,
    Integer,
    Categorical,
    ForbiddenGreaterThanRelation,
    Configuration,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
)
from smac import HyperparameterOptimizationFacade, Scenario# type: ignore[import-not-found]

from learn2rag.evaluation.tools import read_dataset_qa
from learn2rag.pipeline.config import opt_config
import learn2rag.pipeline.search
# import learn2rag.pipeline.generate




def load_registry(path: str = "registry.json") -> dict[str, Any]:
    p = pathlib.Path(path)
    if not p.is_file():
       logging.error("registry file not found")
    with p.open() as f:
        return cast(dict[str, Any], json.load(f))


def _load_existing_trial_answers(answers_dir: pathlib.Path) -> list[dict[str, Any]]:
    if not answers_dir.exists():
        return []
    trials: list[dict[str, Any]] = []
    for p in sorted(answers_dir.glob("trial_*_answers.json")):
        try:
            data = json.loads(p.read_text())
            if isinstance(data, dict) and "trial_id" in data:
                trials.append(data)
        except Exception as e:
            logging.warning(f"Could not read {p}: {e}")
    return trials


def _restore_state_from_existing(out: pathlib.Path, answers_dir: pathlib.Path) -> Dict[str, Any]:
    state: Dict[str, Any] = {"trial_count": 0, "best_cost": 1.0, "convergence": [], "history": []}

    results_path = out / "optimization_results.json"
    if results_path.exists():
        try:
            results = json.loads(results_path.read_text())
            history = results.get("run_history", [])
            convergence = results.get("convergence", [])
            if isinstance(history, list) and history:
                state["history"] = history
                state["convergence"] = convergence if isinstance(convergence, list) else []
                state["trial_count"] = max(int(h.get("trial_id", 0)) for h in history)
                state["best_cost"] = min(float(h.get("cost", 1.0)) for h in history)
                return state
        except Exception as e:
            logging.warning(f"Could not read {results_path}: {e}")

    trials = _load_existing_trial_answers(answers_dir)
    if not trials:
        return state

    best_cost = 1.0
    history: list[dict[str, Any]] = []
    convergence: list[dict[str, Any]] = []
    for trial in sorted(trials, key=lambda t: int(t.get("trial_id", 0))):
        tid = int(trial.get("trial_id", 0))
        cost = float(trial.get("cost", 1.0))
        best_cost = min(best_cost, cost)
        history.append({
            "trial_id": tid,
            "config": trial.get("config", {}),
            "recall": float(trial.get("recall", 0.0)),
            "avg_t_search": float(trial.get("avg_t_search", 0.0)),
            "cost": cost,
            "time_s": None,
            "search_s": None,
            "scoring_s": None,
        })
        convergence.append({"trial": tid, "cost": cost, "best_cost": best_cost})

    state["history"] = history
    state["convergence"] = convergence
    state["trial_count"] = max(int(t.get("trial_id", 0)) for t in trials)
    state["best_cost"] = best_cost
    return state


def _load_existing_importance(out: pathlib.Path) -> Dict[str, Any]:
    path = out / "parameter_importance.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logging.warning(f"Could not read {path}: {e}")
        return {}

def run_search(question: str, user_config: Dict[str, Any], working_config: Dict[str, Any]) -> Tuple[List[Any], float]:
    t0 = time.time()
    docs = asyncio.run(learn2rag.pipeline.search.search_authorized(question, user="anonymous", request_id=None, user_config=user_config, opt_config=working_config))
    search_time = time.time() - t0
    source_list = [point.payload['source'] for point in docs if point.payload is not None and "source" in point.payload]
    return source_list, search_time


def recall(search_results: list[list[Any]], labels: list[Any]) -> float:
    count = 0
    top_k = opt_config["top_k"]
    for q in range(len(search_results)):
        label = str(labels[q])
        hits = [str(h) for h in search_results[q]]
        # print('label ', label, ' hits: ', hits)
        if label in hits[:top_k]:
            count += 1
    return count / len(labels) if labels else 0.0


#I removed seed because there are no use for it
# removed dataset_name because it just use in user config and now we inject it
def objective(config: Configuration,
    questions: List[Dict[str, Any]],
    state: Dict[str, Any],
    answers_dir: pathlib.Path
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

    ucfg = {
        "file_path": None,
        "collection_name": f"CSC-CS_{cfg['chunk_size']}-CO_{cfg['chunk_overlap']}",
        "imported_documents_file_path": None,
        "llm": None,
    }
    env_user_cfg = os.environ.get("PIPELINE_USER_CONFIG")
    if env_user_cfg and pathlib.Path(env_user_cfg).exists():
        ucfg.update(json.loads(pathlib.Path(env_user_cfg).read_text()))

    predictions: list[list[Any]] = []
    goldens: list[Any] = []
    qa_pairs: list[dict[str, Any]] = []

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
            qa_pairs.append({**q, "retrieved_sources": ""})

    if not predictions:
        return 1.0

    t_score = time.time()
    recall_score = recall(predictions, goldens)
    scoring_time = time.time() - t_score
    total_time = time.time() - t_start

    # objective function
    w_recall = 0.5
    w_time = 0.5
    t_search_per_sample_upper = 50
    max_time_s = t_search_per_sample_upper*len(predictions)
    time_cost = max(0.0, 1.0 - (t_search / max_time_s))
    cost = 1.0 - w_recall*recall_score - w_time*time_cost
    avg_t_search = t_search/len(predictions)

    trial_answers = {
        "trial_id": tid,
        "config": cfg,
        "cost": float(cost),
        "recall": float(recall_score),
        "avg_t_search": float(avg_t_search),
        "w_recall": w_recall,
        "w_time": w_time,
        "top_k": opt_config["top_k"],
        "qa_pairs": qa_pairs
    }
    answers_file = answers_dir / f"trial_{tid}_answers.json"
    while answers_file.exists():
        # Keep IDs monotonic when resuming from a partially persisted run.
        tid += 1
        state["trial_count"] = tid
        trial_answers["trial_id"] = tid
        answers_file = answers_dir / f"trial_{tid}_answers.json"
    with open(answers_file, "w") as f:
        json.dump(trial_answers, f, indent=2, default=str)

    state["best_cost"] = min(state["best_cost"], cost)
    state["convergence"].append({"trial": tid, "cost": float(cost), "best_cost": float(state["best_cost"])})
    state["history"].append({
        "trial_id": tid, "config": cfg,
        "recall": float(recall_score),
        "avg_t_search": float(avg_t_search),
        "cost": float(cost),
        "time_s": round(total_time, 2),
        "search_s": round(t_search, 2),
        "scoring_s": round(scoring_time, 2),
    })

    logging.info(
        f"Trial {tid}: recall={recall_score:.4f} avg_t_search={avg_t_search: .4f} cost={cost:.4f} "
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

    raw: dict[str, float] = {}
    for p in params:
        groups: dict[str, list[float]] = {}
        for c, cost in zip(configs, np.array(costs)):
            groups.setdefault(str(c[p]), []).append(float(cost))
        means = [np.mean(g) for g in groups.values()]
        raw[p] = float(np.var(means)) if len(means) > 1 else 0.0

    total = sum(raw.values())
    imp = {p: round(v / total, 4) for p, v in raw.items()} if total > 0 else raw
    ranking = sorted(imp, key=imp.get, reverse=True)  # type: ignore[arg-type]
    result = {"method": "variance_based", "ranking": ranking, "individual": imp}
    with open(output_path / "parameter_importance.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def run(
    dataset_name: str,
    max_questions: int,
    n_trials: int,
    output_dir: Union[str, pathlib.Path],
    registry_path: str,
    resume: bool = False,
    n_trials_is_total: bool = True,
) -> Tuple[
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

    cs = ConfigurationSpace(seed=42)
    cs.add([
        Categorical("chunk_size", [250, 1000, 2000], default=1000),
        Categorical("chunk_overlap", [50, 200], default=50),
        Categorical("search_mode", ["dense", "sparse", "dense_sparse", "dense_sparse_colbert"], default="dense"),
        Categorical("reranking_mode", ["none", "reranking_with_flagreranker", "reranking_with_sentence_transformers", "reranking_with_colbert"], default="none"),
        Categorical("rewrite_mode", ["none", "subqueries", "keywords", "subqueries_keywords"], default="none"),
        Categorical("fusion_mode", ["none", "DBSF", "RRF"], default="none"),
        Categorical("reranking", ["True", "False"], default="False"),
        Categorical("rewrite", ["True", "False"], default="False"),
    ])
    cs.add(ForbiddenGreaterThanRelation(cs["chunk_overlap"], cs["chunk_size"]))

    cs.add(ForbiddenAndConjunction(
        ForbiddenEqualsClause(cs["chunk_size"], 250),
        ForbiddenEqualsClause(cs["chunk_overlap"], 200),
    ))

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


    state: Dict[str, Any]
    if resume:
        state = _restore_state_from_existing(out, answers_dir)
    else:
        state = {"trial_count": 0, "best_cost": 1.0, "convergence": [], "history": []}

    already_done = int(state["trial_count"])
    remaining_trials = max(0, n_trials - already_done) if n_trials_is_total else n_trials

    best_cfg: Dict[str, Any] = {}
    importance: Dict[str, Any] = {}
    total_time = 0.0

    if remaining_trials > 0:
        scenario = Scenario(
            cs,
            deterministic=True,
            n_trials=remaining_trials,
            walltime_limit=172800, #7200,
            seed=42,
            output_directory=out / "smac_output",
        )

        smac = HyperparameterOptimizationFacade(
            scenario=scenario,
            target_function=lambda config, seed=0: objective(config, questions, state, answers_dir)
        )
        t0 = time.time()
        incumbent = smac.optimize()
        if isinstance(incumbent, list):
            incumbent = incumbent[0]
        importance = param_importance(smac, out)
        total_time = time.time() - t0
        best_cfg = incumbent.get_dictionary()
    else:
        logging.info("No remaining trials to run. Returning existing results.")
        if state["history"]:
            best_cfg = min(state["history"], key=lambda h: h["cost"]).get("config", {})
        importance = _load_existing_importance(out)

    if not best_cfg and state["history"]:
        best_cfg = min(state["history"], key=lambda h: h["cost"]).get("config", {})

    best_trial = min(
        (h for h in state["history"] if h.get("config") == best_cfg),
        key=lambda h: h["cost"],
        default=None,
    )
    best_trial_id = best_trial["trial_id"] if best_trial else None
    results_path = out / "optimization_results.json"
    results_path.write_text(json.dumps({
        "best_config": best_cfg,
        "best_trial_id": best_trial_id,
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--n_trials_is_total",
        action="store_true",
        help="Interpret --n_trials as the total desired trial count instead of additional trials.",
    )
    args, _ = parser.parse_known_args()

    final_output_dir = pathlib.Path(args.output_dir)

    env_out = os.environ.get("PIPELINE_OPT_CONFIG")
    if not final_output_dir.exists() and env_out:
        final_output_dir = pathlib.Path(env_out).parent

    incumbent, history, importance = run(
        args.dataset,
        args.max_questions,
        args.n_trials,
        final_output_dir,
        args.registry,
        resume=args.resume,
        n_trials_is_total=(args.n_trials_is_total or args.resume),
    )

    # incumbent, history, importance = run(
    #     args.dataset, args.max_questions, args.n_trials, args.output_dir,
    # )

    best = min(history, key=lambda x: x["cost"]) if history else None
    print(f"\nBest config: {dict(incumbent)}")
    # print(f"BERTScore (golden): {best['avg_bertscore_golden']:.4f}")
    if importance:
        print(f"\nParameter importance:")
        for i, p in enumerate(importance["ranking"], 1):
            print(f"  {i}. {p}: {importance['individual'][p]:.4f}")