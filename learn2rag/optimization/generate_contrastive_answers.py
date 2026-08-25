"""
Generate contrastive (false) answers for RAG evaluation.
"""

import argparse
import json
import logging
import pathlib
import time

from learn2rag.pipeline.llm import llm as learn2rag_llm
from learn2rag.evaluation.tools import read_dataset_qa
from langchain_core.messages import HumanMessage


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
    "hotpot_qa": {                                              # Not being used
        "subdirectory": "distractor",
        "split": "validation",
        "question_field": "question",
        "answer_field": "answer",
        "id_field": "id",
    },
    "repliqa": {                                                # Not being used
        "subdirectory": "repliqa_4",
        "split": None,
        "question_field": "question",
        "answer_field": "long_answer",
        "id_field": "question_id",
    },
}


def generate(prompt):
    response = learn2rag_llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def generate_false_answer(question):
    return generate(
        f"Answer the given question in an incorrect manner.\n\n"
        f"question: {question}"
    )

def run(dataset_name, max_questions=50, output_dir="contrastive_answers"):
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_CONFIG.keys())}"
        )

    cfg = DATASET_CONFIG[dataset_name]
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    qa = read_dataset_qa(dataset_name, cfg["subdirectory"], cfg["split"])
    if max_questions:
        qa = qa.select(range(min(max_questions, len(qa))))

    logging.info(f"Dataset: {dataset_name}, questions: {len(qa)}")

    results = []
    total_start = time.time()

    for idx, item in enumerate(qa):
        question = item.get(cfg["question_field"], "")
        golden = item.get(cfg["answer_field"], "")
        qid = item.get(cfg["id_field"], str(idx))

        if not question:
            continue

        print(f"[{idx+1}/{len(qa)}] {question[:60]}...")

        t0 = time.time()
        false_answer = generate_false_answer(question)
        elapsed = time.time() - t0

        results.append({
            "id": qid,
            "question": question,
            "golden_answer": golden,
            "false_answer": false_answer,
            "generation_time_s": round(elapsed, 2),
        })

    total_time = time.time() - total_start

    out_file = output_path / f"contrastive_answers_{dataset_name}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nGenerated false answers for {len(results)} questions in {total_time:.0f}s")
    print(f"Saved to {out_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="WikiEval",
                        choices=list(DATASET_CONFIG.keys()))
    parser.add_argument("--max_questions", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="contrastive_answers")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    run(args.dataset, args.max_questions, args.output_dir)