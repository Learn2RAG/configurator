"""
RAG Pipeline Optimization with contrastive evaluation.
"""

import argparse
import json
import logging
import pathlib
import time
import copy

import numpy as np
from bert_score import score as bert_score
from ConfigSpace import ConfigurationSpace, Integer, Categorical, ForbiddenGreaterThanRelation
from smac import HyperparameterOptimizationFacade, Scenario

from learn2rag.evaluation.tools import read_dataset_qa
from learn2rag.pipeline.config import opt_config
import learn2rag.pipeline.search
import learn2rag.pipeline.generate


DATASET_CONFIG = {
    "WikiEval": {
        "subdirectory": "",
        "split": "train",
        "question_field": "question",
        "answer_field": "answer",
        "id_field": "id",
    },
    "rag-mini-bioasq": {
        "subdirectory": "question-answer-passages",
        "split": "test",
        "question_field": "question",
        "answer_field": "answer",
        "id_field": "id",
    },
    "hotpot_qa": {      # Not being used
        "subdirectory": "distractor",
        "split": "validation",
        "question_field": "question",
        "answer_field": "answer",
        "id_field": "id",
    },
    "repliqa": {  # Not being used
        "subdirectory": "repliqa_4",
        "split": None,
        "question_field": "question",
        "answer_field": "long_answer",
        "id_field": "question_id",
    },
}

PROMPT_MAP = {
    "default": (
        "# Role and Objective\nYou will act as a smart AI chatbot that answers "
        "questions only by using the content from the provided information list.\n\n"
        "# Instructions\n- Respond in the language of the question.\n"
        "- Answer clear and concise.\n- Only use the provided information.\n"
        "- NEVER use your general knowledge.\n\n"
        "# Information:\n{context}"
    ),
    "concise": (
        "Answer the question using ONLY the provided information. "
        "Be concise and direct. If the information does not contain the answer, say so.\n\n"
        "Information:\n{context}"
    ),
    "detailed": (
        "You are a knowledgeable assistant. Using ONLY the provided information below, "
        "answer the question thoroughly. Cite your sources. "
        "If the information is insufficient, state that clearly.\n\n"
        "Information:\n{context}"
    ),
}


def load_false_answers(contrastive_file):
    with open(contrastive_file) as f:
        data = json.load(f)
    return {item["question"]: item["false_answer"] for item in data}


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


def objective(config, seed, questions, dataset_name, false_map, state, answers_dir):
    state["trial_count"] += 1
    tid = state["trial_count"]
    cfg = dict(config)
    logging.info(f"Trial {tid}: {cfg}")

    wcfg = copy.deepcopy(opt_config)
    wcfg["top_k"] = cfg["top_k"]
    wcfg["chunk_size"] = cfg["chunk_size"]
    wcfg["chunk_overlap"] = cfg["chunk_overlap"]
    wcfg["prompt"] = PROMPT_MAP[cfg["prompt_template"]]
    ucfg = {"file_path": None, "collection_name": dataset_name,
            "imported_documents_file_path": None, "llm": None}

    predictions, goldens, falses = [], [], []
    qa_pairs = []
    t_start = time.time()
    t_search, t_gen = 0.0, 0.0

    for idx, q in enumerate(questions):
        if not q["question"]:
            continue
        false_ans = false_map.get(q["question"], "")
        if not false_ans:
            continue
        try:
            answer, context, st, gt = run_pipeline(q["question"], ucfg, wcfg)
            t_search += st
            t_gen += gt
            predictions.append(answer)
            goldens.append(q["ground_truth"])
            falses.append(false_ans)
            qa_pairs.append({
                "id": q["id"],
                "question": q["question"],
                "golden_answer": q["ground_truth"],
                "generated_answer": answer,
                "retrieved_context": context,
            })
        except Exception as e:
            logging.warning(f"Trial {tid}, q{idx} failed: {e}")
            predictions.append("")
            goldens.append(q["ground_truth"])
            falses.append(false_ans)
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
    _, _, F1_false = bert_score(predictions, falses, lang="en", verbose=False, rescale_with_baseline=True)
    scoring_time = time.time() - t_score

    bert_gold = [max(0.0, f.item()) for f in F1_gold]
    bert_false = [max(0.0, f.item()) for f in F1_false]

    ratios = []
    # objective function
    for bg, bf in zip(bert_gold, bert_false):
        denom = bg + bf
        ratios.append(bg / denom if denom > 0 else 0.5)

    avg_ratio = np.mean(ratios)
    avg_bert_gold = np.mean(bert_gold)
    avg_bert_false = np.mean(bert_false)
    cost = 1.0 - avg_ratio
    total_time = time.time() - t_start

    trial_answers = {
        "trial_id": tid,
        "config": cfg,
        "cost": float(cost),
        "avg_ratio": float(avg_ratio),
        "qa_pairs": qa_pairs,
    }
    answers_file = answers_dir / f"trial_{tid}_answers.json"
    with open(answers_file, "w") as f:
        json.dump(trial_answers, f, indent=2, default=str)

    state["best_cost"] = min(state["best_cost"], cost)
    state["convergence"].append({"trial": tid, "cost": float(cost), "best_cost": float(state["best_cost"])})
    state["history"].append({
        "trial_id": tid, "config": cfg,
        "avg_ratio": float(avg_ratio),
        "avg_bertscore_golden": float(avg_bert_gold),
        "avg_bertscore_false": float(avg_bert_false),
        "cost": float(cost), "time_s": round(total_time, 2),
        "search_s": round(t_search, 2), "gen_s": round(t_gen, 2),
        "scoring_s": round(scoring_time, 2),
    })

    logging.info(
        f"Trial {tid}: ratio={avg_ratio:.4f} [gold={avg_bert_gold:.4f} false={avg_bert_false:.4f}] "
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


def run(dataset_name, max_questions, n_trials, output_dir, contrastive_dir):
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_CONFIG.keys())}"
        )

    dcfg = DATASET_CONFIG[dataset_name]
    contrastive_file = pathlib.Path(contrastive_dir) / f"contrastive_answers_{dataset_name}.json"
    out = pathlib.Path(output_dir) / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    answers_dir = out / "trial_answers"
    answers_dir.mkdir(parents=True, exist_ok=True)

    false_map = load_false_answers(contrastive_file)
    logging.info(f"Loaded {len(false_map)} false answers from {contrastive_file}")

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
            config, seed, questions, dataset_name, false_map, state, answers_dir
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
            "metric": "bertscore_ratio(golden, false)",
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
    parser.add_argument("--output_dir", type=str, default="optimization_results_contrastive")
    parser.add_argument("--contrastive_dir", type=str, default="contrastive_answers")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    incumbent, history, importance = run(
        args.dataset, args.max_questions, args.n_trials,
        args.output_dir, args.contrastive_dir,
    )

    best = min(history, key=lambda x: x["cost"])
    print(f"\nBest config: {dict(incumbent)}")
    print(f"Ratio score: {best['avg_ratio']:.4f}")
    print(f"  BERTScore vs golden: {best['avg_bertscore_golden']:.4f}")
    print(f"  BERTScore vs false:  {best['avg_bertscore_false']:.4f}")
    if importance:
        print(f"\nParameter importance:")
        for i, p in enumerate(importance["ranking"], 1):
            print(f"  {i}. {p}: {importance['individual'][p]:.4f}")