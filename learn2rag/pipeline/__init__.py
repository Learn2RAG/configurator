import os

import uvicorn


def main():
    from .app import app
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.environ.get('LEARN2RAG_PIPELINE_PORT', 9000)),
    )
