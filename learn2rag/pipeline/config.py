import json
import os
import logging
from pathlib import Path
from typing import Any

def _load_json_config(env_var: str, relative_path: str, fallback_path: Path) -> dict[str, Any]:
    if env_val := os.environ.get(env_var):
        path = Path(env_val)
        if path.is_file():
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)  # type: ignore[no-any-return]

    learn2rag_path = os.environ.get("LEARN2RAG_PATH", ".")
    candidate = Path(learn2rag_path) / relative_path
    if candidate.is_file():
        with open(candidate, "r", encoding="utf-8") as file:
            return json.load(file)  # type: ignore[no-any-return]

    candidate_cwd = Path(relative_path)
    if candidate_cwd.is_file():
        with open(candidate_cwd, "r", encoding="utf-8") as file:
            return json.load(file)  # type: ignore[no-any-return]

    if fallback_path.is_file():
        with open(fallback_path, "r", encoding="utf-8") as file:
            return json.load(file)  # type: ignore[no-any-return]

    return {}

user_config = _load_json_config(
    "PIPELINE_USER_CONFIG",
    "learn2rag/pipeline/user_config.json",
    Path(__file__).parent / "user_config.json"
)

importer_config = _load_json_config(
    "IMPORTER_CONFIG",
    "learn2rag/importer/config/config.json",
    Path(__file__).parent.parent / "importer" / "config" / "config.json"
)

opt_config = _load_json_config(
    "PIPELINE_OPT_CONFIG",
    "learn2rag/pipeline/opt_config.json",
    Path(__file__).parent / "opt_config.json"
)
logging.info(f"Loaded opt_config:\n{json.dumps(opt_config, indent=4)}")
