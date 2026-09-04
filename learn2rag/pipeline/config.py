import json
import os
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

default_user_config = BASE_DIR / "user_config.json"
default_importer_config = BASE_DIR.parent / "importer" / "config" / "config.json"
default_opt_config = BASE_DIR / "opt_config.json"

with open(os.environ.get("PIPELINE_USER_CONFIG", default_user_config), "r") as file:
    user_config = json.load(file)

with open(os.environ.get("IMPORTER_CONFIG", default_importer_config), "r") as file:
    importer_config = json.load(file)

with open(os.environ.get("PIPELINE_OPT_CONFIG", default_opt_config), "r") as file:
    opt_config = json.load(file)
    logging.info(f"Loaded opt_config:\n{json.dumps(opt_config, indent=4)}")
