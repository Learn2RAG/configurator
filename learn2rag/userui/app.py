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
from ..pipeline.config import importer_config, user_config, opt_config
from ..pipeline.operators import BasicPipeline
from ..pipeline.operators.base import BaseOperator
from ..pipeline.qdrant import Qdrant
from ..utils.starlette import PrefixRewriteMiddleware

pipeline: BaseOperator = BasicPipeline()


class TestResponse(BaseModel):
    message: str


app = FastAPI()
# required by oauth library
app.add_middleware(SessionMiddleware, secret_key="FIXME")
def token_handler(request: Request, name: str, token: str) -> None:
    if SESSION_USER_AUTHS not in request.session:
        request.session[SESSION_USER_AUTHS] = {}
    request.session[SESSION_USER_AUTHS][name] = {'token': token}
app.include_router(auth.build_router(importer_config, token_handler), prefix='/auth')


api_prefix = '/api'
app.include_router(api.build_router(), prefix=api_prefix)


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
    })


@app.on_event("startup")
async def startup_event() -> None:
    Qdrant.ensure_collection(user_config["collection_name"], opt_config)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    message = str(exc)
    logging.error(f"validation_exception_handler: {message}")
    content = {'message': message}
    return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


@app.get("/test")
async def test() -> TestResponse:
    return TestResponse(message="Hello World")
