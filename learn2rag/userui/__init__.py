import os

import uvicorn

from .app import app


def main() -> None:
    # TODO use uvicorn.config.Config
    uvicorn.run(
        app,
        host='0.0.0.0',  # FIXME: use learn2rag config value
        port=int(os.environ.get('LEARN2RAG_PIPELINE_PORT', 9000)),
        log_level='debug',
    )
