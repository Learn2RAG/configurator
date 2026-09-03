from pathlib import Path
import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from . import auth, chat
from .constants import SESSION_USER_AUTHS
from ..bootstrap import setup_fastapi as learn2rag_bootstrap_setup
from ..pipeline import api
from ..pipeline.config import importer_config
from ..utils.starlette import PrefixRewriteMiddleware


class TestResponse(BaseModel):
    message: str


def build_app() -> FastAPI:
    app = FastAPI()

    # required by oauth library
    app.add_middleware(SessionMiddleware, secret_key="FIXME")

    auth_prefix = '/auth'
    def login_handler(request: Request, name: str, token: str) -> None:
        if SESSION_USER_AUTHS not in request.session:
            request.session[SESSION_USER_AUTHS] = {}
        request.session[SESSION_USER_AUTHS][name] = {'token': token}
    def logout_handler(request: Request, name: str) -> None:
        del request.session[SESSION_USER_AUTHS][name]
    auth_router = auth.build_router(importer_config, login_handler, logout_handler)
    app.include_router(
        auth_router,
        prefix=auth_prefix,
    )

    api_prefix = '/api'
    app.mount(api_prefix, api.build_app())

    chat_prefix = '/chat'
    # llama.cpp UI always uses a relative path
    app.add_middleware(
        PrefixRewriteMiddleware,
        source=chat_prefix,
        target=api_prefix,
        paths=[
            '/v1/models',
            '/v1/chat/completions',
        ],
       )
    app.mount(chat_prefix, chat.build_app())

    templates = Jinja2Templates(directory=learn2rag_bootstrap_setup(app, [
        Path(__file__).parent.parent / 'userui' / 'templates',
    ]))

    @app.get('/')
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, 'index.html', context={
            'auth_prefix': auth_prefix,
            'auth_routers': auth_router.auth_routers,
            'user_auths': request.session.get(SESSION_USER_AUTHS, {}),
        })

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        message = str(exc)
        logging.error(f"validation_exception_handler: {message}")
        content = {'message': message}
        return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

    @app.get("/test")
    async def test() -> TestResponse:
        return TestResponse(message="Hello World")

    return app
