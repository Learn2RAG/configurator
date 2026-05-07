import logging
import pathlib
import os
import json
import collections.abc
import copy
import yaml
import argparse

from . import baseline_optimization
#TODO : now we need to copy the dataset to here manually it should consider in installation maybe !
# {storage_path}/datasets/WikiEval/
def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG baseline optimization.")
    parser.add_argument("task", help="Module task name (e.g., learn2rag.optimization)")
    parser.add_argument("--logging-config", type=str, help="Path to logging config yml")
    parser.add_argument("--registry-path", type=str, help="Path to registry.json")
    parser.add_argument("--dataset", type=str, default="WikiEval")
    parser.add_argument("--questions", type=int, default=10)
    parser.add_argument("--trials", type=int, default=10)
    args, unknown = parser.parse_known_args()

    if args.logging_config and pathlib.Path(args.logging_config).exists():
        with open(args.logging_config, 'r') as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config)

    logger = logging.getLogger(__name__)
    logger.info(f"Running task: {args.task}")


    logging.info(f"Optimization started for {args.dataset}")
    #TODO : read or get dataset_name and maxquestions and n_trails

    output_dir = pathlib.Path("./optimization/output/")
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_optimization.run(args.dataset,args.questions,args.trials,output_dir,args.registry_path)
    logging.info("optimization is done")
    results_path = output_dir /args.dataset/ "optimization_results.json"
    logging.info(f"save optimized results here : {results_path}")
    if not results_path.exists():
        logging.warning(f"Optimization results not found at {results_path}")
        return

    with open(results_path, "r") as f:
        full_results = json.load(f)

    best_config = full_results.get("best_config", {})

    target_config_path = os.environ.get("PIPELINE_OPT_CONFIG", "learn2rag/pipeline/opt_config.json")

    existing_config = {}
    if os.path.exists(target_config_path):
        try:
            with open(target_config_path, "r", encoding="utf-8") as f:
                existing_config = json.load(f)
        except json.JSONDecodeError:
            logging.error(f"Existing config at {target_config_path} is corrupted. It will be overwritten.")

    updated_config = deep_update(copy.deepcopy(existing_config), best_config)

    pathlib.Path(target_config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(target_config_path, "w", encoding="utf-8") as f:
        json.dump(updated_config, f, indent=4)

    logging.info(f"Successfully updated opt_config at: {target_config_path}")

def deep_update(source, overrides):
    """Recursively updates a dictionary."""
    for key, value in overrides.items():
        if isinstance(value, collections.abc.Mapping) and key in source:
            deep_update(source.get(key, {}), value)
        else:
            source[key] = value
    return source

if __name__ == "__main__":
    main()