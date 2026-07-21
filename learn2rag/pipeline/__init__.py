import os

import uvicorn


def main() -> None:
    from .app import app
    uvicorn.run(
        app,
        host="0.0.0.0",  # FIXME: use learn2rag config value
        port=int(os.environ.get('LEARN2RAG_PIPELINE_PORT', 9000)),
    )
