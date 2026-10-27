'''
Downloads the embedding model into the Hugging Face cache, so that the first import does not have to.
The start scripts run it before the UI starts, see ../start.linux and ../start.windows.bat
'''
import importlib.resources
import json
import logging
import os
from typing import Any

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

# The same files FlagEmbedding downloads for BAAI/bge-m3, so it finds them all in the cache
IGNORE_PATTERNS = ['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5']


def load_opt_config() -> Any:
    if path := os.environ.get('PIPELINE_OPT_CONFIG'):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    with (importlib.resources.files('learn2rag') / 'pipeline' / 'opt_config.json').open('r', encoding='utf-8') as f:
        return json.load(f)


def main() -> None:
    model_name = str(load_opt_config().get('embedding_model', 'BAAI/bge-m3'))
    logger.info('Downloading the embedding model %s, the first time this takes a while', model_name)
    path = snapshot_download(repo_id=model_name, ignore_patterns=IGNORE_PATTERNS)
    logger.info('Embedding model %s is available in %s', model_name, path)
