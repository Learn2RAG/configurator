import json
import os
import logging
import collections.abc
from typing import Any



opt_config: dict[str, Any] = {}

with open(os.environ.get("PIPELINE_USER_CONFIG", "learn2rag/pipeline/user_config.json"), "r") as file:
    user_config = json.load(file)

with open(os.environ.get("IMPORTER_CONFIG", "learn2rag/importer/config/config.json"), "r") as file:
    importer_config = json.load(file)

def refresh_configs():
    global opt_config
    logging.info(f"Refreshing configs...")
    with open(os.environ.get("PIPELINE_OPT_CONFIG", "learn2rag/pipeline/opt_config.json"), "r") as file:
        base_data = json.load(file)

    final_data = base_data

    patch_path = os.environ.get("PIPELINE_OPT_PATCH_CONFIG", "learn2rag/pipeline/opt_patch_config.json")
    if os.path.exists(patch_path):
        logging.info(f"Patch found! Layering {patch_path} onto defaults.")
        try:
            with open(patch_path, "r") as f:
                patch_data = json.load(f)
                # Overlay the patch onto our final_data
                final_data = deep_update(base_data, patch_data)
        except json.JSONDecodeError:
            logging.error(f"Patch file at {patch_path} is corrupted. Skipping overlay.")
    else:
        # This is the "Fallback" behavior
        logging.warning(f"No patch file found at {patch_path}. System will run on default settings.")
    logging.info(f"Refreshing configs done {final_data}")
    opt_config.clear()
    opt_config.update(final_data)

    logging.info(f"FINAL OPT_CONFIG STRUCTURE: {json.dumps(opt_config, indent=2)}")


def deep_update(base_dict, overrides):
    """Recursively merges overrides into base_dict."""
    for key, value in overrides.items():
        if isinstance(value, collections.abc.Mapping):
            if key in base_dict and not isinstance(base_dict[key], collections.abc.Mapping):
                logging.error(f"Type Mismatch at '{key}': expected dict, got {type(base_dict[key])}")
                base_dict[key] = {}
            base_dict[key] = deep_update(base_dict.get(key, {}), value)
        else:
            logging.info(f"Overriding '{key}': {base_dict.get(key)} -> {value}")
            base_dict[key] = value
    return base_dict


# Initial load on module import
refresh_configs()