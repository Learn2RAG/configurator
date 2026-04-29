import logging
import pathlib
import os
import json
import collections.abc
import copy
from os import mkdir

from . import baseline_optimization
#TODO : now we need to copy the dataset to here manually it should consider in installation maybe !
# {storage_path}/datasets/WikiEval/
def main() -> None:
    logging.error("optimization is started")
    dataset_name = "WikiEval"

    output_dir = pathlib.Path("./optimization/output/")
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_optimization.run(dataset_name,10,10,output_dir)
    logging.error("optimization is done")
    results_path = output_dir /dataset_name/ "optimization_results.json"
    logging.info(f"save optimized results here : {results_path}")
    if not results_path.exists():
        logging.error(f"Optimization results not found at {results_path}")
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

    updated_config = deep_update(existing_config, best_config)

    pathlib.Path(target_config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(target_config_path, "w", encoding="utf-8") as f:
        json.dump(updated_config, f, indent=4)

    logging.info(f"Successfully updated opt_config at: {target_config_path}")

def deep_update(base_dict, overrides):
    """Recursively merges overrides into a copy of base_dict, returning the new dict."""
    result = copy.deepcopy(base_dict)
    for key, value in overrides.items():
        if isinstance(value, collections.abc.Mapping):
            if key in result and not isinstance(result[key], collections.abc.Mapping):
                logging.error(f"Type Mismatch at '{key}': expected dict, got {type(result[key])}")
                result[key] = {}
            result[key] = deep_update(result.get(key, {}), value)
        else:
            logging.info(f"Overriding '{key}': {result.get(key)} -> {value}")
            result[key] = value
    return result
