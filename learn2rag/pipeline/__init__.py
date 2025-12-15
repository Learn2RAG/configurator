import os

import uvicorn

from .app import app


def main():
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.environ.get('LEARN2RAG_PIPELINE_PORT', 9000)),
    )
