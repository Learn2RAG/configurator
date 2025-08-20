import json
import os

with open(os.environ.get("PIPELINE_USER_CONFIG", "user_config.json"), "r") as file:
    user_config = json.load(file)

with open(os.environ.get("PIPELINE_OPT_CONFIG", "opt_config.json"), "r") as file:
    opt_config = json.load(file)
