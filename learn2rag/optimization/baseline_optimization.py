"""
RAG Pipeline Optimization with BERTScore evaluation.
"""

import argparse
import json
import logging
import pathlib
import time
import copy
import os
from typing import Dict, Any, List, Union, Tuple, cast
from qdrant_client.models import ScoredPoint
import numpy as np
from bert_score import score as bert_score # type: ignore
from ConfigSpace import ConfigurationSpace, Integer, Categorical, ForbiddenGreaterThanRelation, Configuration
from smac import HyperparameterOptimizationFacade, Scenario

from learn2rag.evaluation.tools import read_dataset_qa
from learn2rag.pipeline.config import opt_config
import learn2rag.pipeline.search
import learn2rag.pipeline.generate

def load_registry(path: str = "registry.json") -> dict[str, Any]:
    p = pathlib.Path(path)
    if not p.is_file():
       logging.error("registry file not found")
    with p.open() as f:
        return cast(Dict[str, Any], json.load(f))

def run_pipeline(question: str, user_config: Dict[str, Any], working_config: Dict[str, Any]) -> Tuple[str, str, float, float]:
    t0 = time.time()
    docs = learn2rag.pipeline.search.search(question, user_config, working_config)
    search_time = time.time() - t0

    t0 = time.time()
    answer = learn2rag.pipeline.generate.generate(question, docs.points, working_config)
    gen_time = time.time() - t0

    doc_list = docs.points if hasattr(docs, "points") else docs
    context = ""
    if doc_list:
        context_parts = []
        for d in doc_list:
            payload = getattr(d, "payload", {}) or {}
            path = payload.get("path", "unknown") if isinstance(payload, dict) else "unknown"
            content = payload.get("content", "") if isinstance(payload, dict) else ""
            context_parts.append(f"Source: {path}\nContent: {content}")
        context = "\n\n".join(context_parts)
    return answer, context[:3000], search_time, gen_time

#I removed seed because there are no use for it
# removed dataset_name becuase it just use in yser config and now we inject it
def objective(config: Configuration,
    questions: List[Dict[str, Any]],
    dataset_name: str,
    state: Dict[str, Any],
    answers_dir: pathlib.Path
    ,prompt_map: Dict[str, Any]
) -> float:
    state["trial_count"] += 1
    tid = state["trial_count"]
    cfg = dict(config)
    logging.info(f"Trial {tid}: {cfg}")

    working_cfg = copy.deepcopy(opt_config)
    working_cfg.update({
        "top_k": cfg["top_k"],
       # "chunk_size": cfg["chunk_size"],
       # "chunk_overlap": cfg["chunk_overlap"],
        "prompt": prompt_map[cfg["prompt_template"]],
    })

    ucfg = {
        "file_path": None,
        "collection_name": dataset_name,
        "imported_documents_file_path": None,
        "llm": None,
    }
    env_user_cfg = os.environ.get("PIPELINE_USER_CONFIG")
    if env_user_cfg and pathlib.Path(env_user_cfg).exists():
        ucfg.update(json.loads(pathlib.Path(env_user_cfg).read_text()))

    predictions, goldens = [], []
    qa_pairs = []
    t_start = time.time()
    t_search, t_gen = 0.0, 0.0

    for q in questions:
        # Preserve the original “skip empty question” guard
        if not q.get("question"):
            continue
        try:
            answer, context, t_s, t_g = run_pipeline(q["question"], ucfg, working_cfg)
            t_search += t_s
            t_gen += t_g
            predictions.append(answer)
            goldens.append(q["ground_truth"])
            qa_pairs.append({**q, "generated_answer": answer, "retrieved_context": context})
        except Exception as e:
            # Same behaviour as the old version: record a blank answer. TODO : check if we need this
            logging.warning(f"Trial {tid}, q{q.get('id','?')} failed: {e}")
            predictions.append("")
            goldens.append(q["ground_truth"])
            qa_pairs.append({**q, "generated_answer": "", "retrieved_context": ""})

    if not predictions:
        return 1.0

    t_score = time.time()
    _, _, F1_gold = bert_score(predictions, goldens, lang="en", verbose=False, rescale_with_baseline=True)
    scoring_time = time.time() - t_score

    # objective function
    bert_gold = [max(0.0, f.item()) for f in F1_gold]
    avg_bert_gold = np.mean(bert_gold)
    cost = 1.0 - avg_bert_gold
    total_time = time.time() - t_start

    trial_answers = {
        "trial_id": tid,
        "config": cfg,
        "cost": float(cost),
        "avg_bertscore_golden": float(avg_bert_gold),
        "qa_pairs": qa_pairs,
    }
    answers_file = answers_dir / f"trial_{tid}_answers.json"
    with open(answers_file, "w") as f:
        json.dump(trial_answers, f, indent=2, default=str)

    state["best_cost"] = min(state["best_cost"], cost)
    state["convergence"].append({"trial": tid, "cost": float(cost), "best_cost": float(state["best_cost"])})
    state["history"].append({
        "trial_id": tid, "config": cfg,
        "avg_bertscore_golden": float(avg_bert_gold),
        "cost": float(cost), "time_s": round(total_time, 2),
        "search_s": round(t_search, 2), "gen_s": round(t_gen, 2),
        "scoring_s": round(scoring_time, 2),
    })

    logging.info(
        f"Trial {tid}: bertscore_golden={avg_bert_gold:.4f} cost={cost:.4f} "
        f"time={total_time:.1f}s (search={t_search:.1f} gen={t_gen:.1f} score={scoring_time:.1f})"
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
        groups : Dict[str, List[float]] = {}
        for c, cost in zip(configs, np.array(costs)):
            groups.setdefault(str(c[p]), []).append(cost)
        means = [np.mean(g) for g in groups.values()]
        raw[p] = float(np.var(means)) if len(means) > 1 else 0.0

    total = sum(raw.values())
    imp = {p: round(v / total, 4) for p, v in raw.items()} if total > 0 else raw
    ranking = sorted(imp, key=lambda k: imp[k], reverse=True)
    result = {"method": "variance_based", "ranking": ranking, "individual": imp}
    with open(output_path / "parameter_importance.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def run(dataset_name: str, max_questions: int, n_trials: int, output_dir: Union[str, pathlib.Path],registry_path:str) -> Tuple[
        Dict[str, Any], List[Any], Dict[str, Any]]:
    logging.info(f"registry_path is : {registry_path}")
    registry = load_registry(registry_path)
    datasets = registry["datasets"]
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(datasets.keys())}")
    dcfg = datasets[dataset_name]
    fields = dcfg["fields"]
    base_path = pathlib.Path(dcfg["path"])

    out = pathlib.Path(output_dir) / dataset_name
    out.mkdir(parents=True, exist_ok=True)
    answers_dir = out / "trial_answers"
    answers_dir.mkdir(parents=True, exist_ok=True)
    if base_path.suffix.lower() == '.csv':
        target_path = base_path
    else:
        target_path = base_path / 'source' / dcfg.get("subdirectory", "")

    logging.debug(f"target path for dataset is {target_path} ")

    qa = read_dataset_qa(target_path, split=dcfg["split"])
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
        Integer("top_k", (1, 20), default=4),
        #Integer("chunk_size", (200, 4000), default=2000),
        #Integer("chunk_overlap", (0, 500), default=200),
        Categorical("prompt_template", list(prompt_map.keys()), default="default"),
    ])
    #cs.add(ForbiddenGreaterThanRelation(cs["chunk_overlap"], cs["chunk_size"]))
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
    if isinstance(incumbent, list):
        best_cfg = incumbent[0].get_dictionary()
    else:
        best_cfg = incumbent.get_dictionary()
    importance = param_importance(smac, out)
    total_time = time.time() - t0
    #best_cfg = incumbent.get_dictionary()
    results_path = out / "optimization_results.json"
    results_path.write_text(json.dumps({
        "best_config": best_cfg,
        "run_history": state["history"],
        "convergence": state["convergence"],
        "parameter_importance": importance,
        "total_time_s": round(total_time, 2),
        "dataset": dataset_name,
        "metric": "bertscore_golden",
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
    print(f"BERTScore (golden): {best['avg_bertscore_golden']:.4f}")
    if importance:
        print(f"\nParameter importance:")
        for i, p in enumerate(importance["ranking"], 1):
            print(f"  {i}. {p}: {importance['individual'][p]:.4f}")