import json
import os

with open(os.environ.get("PIPELINE_USER_CONFIG", "learn2rag/pipeline/user_config.json"), "r") as file:
    user_config = json.load(file)

with open(os.environ.get("IMPORTER_CONFIG", "learn2rag/importer/config/config.json"), "r") as file:
    importer_config = json.load(file)

with open(os.environ.get("PIPELINE_OPT_CONFIG", "learn2rag/pipeline/opt_config.json"), "r") as file:
    opt_config = json.load(file)
