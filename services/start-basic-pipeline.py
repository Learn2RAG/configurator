#!/usr/bin/env python3
import os
import json
import sys

from learn2rag.pipeline import run_pipeline  # this should exist if you're using the default RAG logic


def main():
    user_config_path = os.environ.get("PIPELINE_USER_CONFIG")
    opt_config_path = os.environ.get("PIPELINE_OPT_CONFIG")

    if not user_config_path or not os.path.exists(user_config_path):
        print("Missing or invalid PIPELINE_USER_CONFIG")
        sys.exit(1)

    if not opt_config_path or not os.path.exists(opt_config_path):
        print("Missing or invalid PIPELINE_OPT_CONFIG")
        sys.exit(1)

    with open(user_config_path, "r") as f:
        user_config = json.load(f)

    with open(opt_config_path, "r") as f:
        opt_config = json.load(f)

    print("Starting pipeline with:")
    print(f"   - user config: {user_config_path}")
    print(f"   - opt config:  {opt_config_path}")

    run_pipeline(user_config, opt_config)


if __name__ == "__main__":
    main()
