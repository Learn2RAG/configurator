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

import numpy as np
from bert_score import score as bert_score
from ConfigSpace import ConfigurationSpace, Integer, Categorical, ForbiddenGreaterThanRelation
from smac import HyperparameterOptimizationFacade, Scenario

from learn2rag.evaluation.tools import read_dataset_qa
from learn2rag.pipeline.config import opt_config
import learn2rag.pipeline.search
import learn2rag.pipeline.generate

def load_registry(registry_path: pathlib.Path):
    if not registry_path.exists():
        logging.warning("no registry found !")
        return {"datasets": {}, "prompts": {"default": "Context: {context}\nQ: {question}"}}
    return json.loads(registry_path.read_text())

def get_base_user_config():
    """Loads the base config provided by the orchestrator env var."""
    path = os.environ.get("PIPELINE_USER_CONFIG")
    if path and pathlib.Path(path).exists():
        return json.loads(pathlib.Path(path).read_text())
    logging.warning("no user config found !")
    return {}

def run_pipeline(question, user_config, working_config):
    t0 = time.time()
    docs = learn2rag.pipeline.search.search(question, user_config, working_config)
    search_time = time.time() - t0

    t0 = time.time()
    answer = learn2rag.pipeline.generate.generate(question, docs, working_config)
    gen_time = time.time() - t0

    doc_list = docs.points if hasattr(docs, "points") else docs
    context = ""
    if doc_list:
        context = "\n\n".join([
            f"Source: {d.payload.get('path', 'unknown')}\nContent: {d.payload.get('content', '')}"
            for d in doc_list
        ])
    return answer, context[:3000], search_time, gen_time

#I removed seed because there are no use for it
# removed dataset_name becuase it just use in yser config and now we inject it
def objective(config, questions, state, answers_dir, prompts_repo, base_user_cfg):
    state["trial_count"] += 1
    tid = state["trial_count"]
    cfg = dict(config)
    logging.info(f"Trial {tid}: {cfg}")

    working_cfg = copy.deepcopy(opt_config)
    working_cfg.update({
        "top_k": cfg["top_k"],
        "chunk_size": cfg["chunk_size"],
        "chunk_overlap": cfg["chunk_overlap"],
        "prompt": prompts_repo.get(cfg["prompt_template"], prompts_repo["default"])
    })

    predictions, goldens = [], []
    qa_pairs = []
    t_start = time.time()
    t_search, t_gen = 0.0, 0.0

    for idx, q in enumerate(questions):
        if not q["question"]:
            continue
        try:
            answer, context, st, gt = run_pipeline(q["question"], base_user_cfg, working_cfg)
            t_search += st
            t_gen += gt
            predictions.append(answer)
            goldens.append(q["ground_truth"])
            qa_pairs.append({
                "id": q["id"],
                "question": q["question"],
                "golden_answer": q["ground_truth"],
                "generated_answer": answer,
                "retrieved_context": context,
            })
        except Exception as e:
            logging.warning(f"Trial {tid}, q{idx} failed: {e}")
            # do we need to add this :/
            predictions.append("")
            goldens.append(q["ground_truth"])
            qa_pairs.append({
                "id": q["id"],
                "question": q["question"],
                "golden_answer": q["ground_truth"],
                "generated_answer": "",
                "retrieved_context": "",
            })

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


def param_importance(smac, output_path):
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


def run(dataset_name: str, max_questions: int, n_trials, output_dir: int):
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_CONFIG.keys())}"
        )

    dcfg = DATASET_CONFIG[dataset_name]
    out = pathlib.Path(output_dir) / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    answers_dir = out / "trial_answers"
    answers_dir.mkdir(parents=True, exist_ok=True)

    qa = read_dataset_qa(dataset_name, dcfg["subdirectory"], dcfg["split"])
    if max_questions:
        qa = qa.select(range(min(max_questions, len(qa))))

    questions = []
    for i, r in enumerate(qa):
        questions.append({
            "question": r.get(dcfg["question_field"], ""),
            "ground_truth": r.get(dcfg["answer_field"], ""),
            "id": r.get(dcfg["id_field"], str(i)),
        })
    logging.info(f"Loaded {len(questions)} questions from {dataset_name}")

    cs = ConfigurationSpace(seed=42)
    cs.add([Integer("top_k", (1, 20), default=4),
            Integer("chunk_size", (200, 4000), default=2000),
            Integer("chunk_overlap", (0, 500), default=200),
            Categorical("prompt_template", ["default", "concise", "detailed"], default="default")])
    cs.add(ForbiddenGreaterThanRelation(cs["chunk_overlap"], cs["chunk_size"]))

    scenario = Scenario(configspace=cs, deterministic=True, n_trials=n_trials,
                        walltime_limit=7200, seed=42, output_directory=out / "smac_output")

    state = {"trial_count": 0, "best_cost": 1.0, "convergence": [], "history": []}
    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=lambda config, seed=0: objective(
            config, seed, questions, dataset_name, state, answers_dir
        ),
    )

    t0 = time.time()
    incumbent = smac.optimize()
    total_time = time.time() - t0

    importance = param_importance(smac, out)

    with open(out / "optimization_results.json", "w") as f:
        json.dump({
            "best_config": dict(incumbent),
            "run_history": state["history"],
            "convergence": state["convergence"],
            "parameter_importance": importance,
            "total_time_s": round(total_time, 2),
            "dataset": dataset_name,
            "metric": "bertscore_golden",
            "answers_dir": str(answers_dir),
        }, f, indent=2, default=str)

    logging.info(f"Done in {total_time:.0f}s")
    logging.info(f"Trial answers saved to {answers_dir}")
    return incumbent, state["history"], importance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="WikiEval",
                        choices=list(DATASET_CONFIG.keys()))
    parser.add_argument("--max_questions", type=int, default=50)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="optimization_results_baseline")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    incumbent, history, importance = run(
        args.dataset, args.max_questions, args.n_trials, args.output_dir,
    )

    best = min(history, key=lambda x: x["cost"])
    print(f"\nBest config: {dict(incumbent)}")
    print(f"BERTScore (golden): {best['avg_bertscore_golden']:.4f}")
    if importance:
        print(f"\nParameter importance:")
        for i, p in enumerate(importance["ranking"], 1):
            print(f"  {i}. {p}: {importance['individual'][p]:.4f}")