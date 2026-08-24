from pathlib import Path
import logging
import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)


def build_app() -> FastAPI:
    '''
    Llama.cpp UI
    '''
    app = FastAPI()

    @app.get('/props')
    async def props() -> JSONResponse:
        'Endpoint required for llama.cpp UI'
        return JSONResponse({
            'modalities': {
                'vision': False,
                'video': False,
                'audio': False,
            },
            "endpoint_slots": False,
            "endpoint_props": False,
            "endpoint_metrics": False,
            "ui": True,
            "chat_template": "",
            "chat_template_caps": {},
            "cors_proxy_enabled": False,
        })

    chat_dir = Path(os.environ['LEARN2RAG_PATH']) / 'services' / 'llama.cpp' / 'tools' / 'ui' / 'dist'
    app.mount('/', StaticFiles(directory=chat_dir, html=True), name='llama.cpp-ui')

    return app
