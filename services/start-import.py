#!/usr/bin/env python3
import subprocess
import os
import sys


def run_command(command_list):
    print(f"Running: {' '.join(command_list)}")
    result = subprocess.run(command_list)
    if result.returncode != 0:
        print(f"Command failed: {' '.join(command_list)}")
        sys.exit(result.returncode)


def main():
    learn2rag_path = os.environ.get("LEARN2RAG_PATH")
    storage_path = os.environ.get("STORAGE_PATH")

    if not learn2rag_path or not storage_path:
        print("LEARN2RAG_PATH or STORAGE_PATH not set in environment")
        sys.exit(1)

    importer_python = os.path.join(
        learn2rag_path,
        "services",
        "importer",
        ".venv",
        "Scripts" if os.name == "nt" else "bin",
        "python",
    )
    basic_pipeline_python = os.path.join(
        learn2rag_path,
        "services",
        "basic-pipeline",
        ".venv",
        "Scripts" if os.name == "nt" else "bin",
        "python",
    )

    importer_script = os.path.join(learn2rag_path, "services", "importer", "main.py")
    ingestion_script = os.path.join(learn2rag_path, "services", "basic-pipeline", "ingestion.py")
    config_path = os.path.join(storage_path, "importer_config.json")

    run_command([importer_python, importer_script, "--config", config_path])
    run_command([basic_pipeline_python, ingestion_script])


if __name__ == "__main__":
    main()
