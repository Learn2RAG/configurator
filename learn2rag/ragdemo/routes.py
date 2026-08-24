import logging
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Request, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from learn2rag.pipeline.config import user_config
from learn2rag.pipeline.qdrant import Qdrant

from .models import IndexErrorResponse, IndexResponse
from .service import inspect_index

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _BASE_DIR / "static"
_CONFIGURATOR_STATIC_DIR = _BASE_DIR.parent / "ui" / "static"
templates = Jinja2Templates(directory=_BASE_DIR / "templates")

router = APIRouter()


@router.get("/test")
async def test() -> dict[str, int]:
    return {"test": 123}


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"current_year": date.today().year},
    )


@router.get("/static/ragdemo.css", include_in_schema=False)
async def stylesheet() -> FileResponse:
    return FileResponse(_STATIC_DIR / "ragdemo.css", media_type="text/css")


@router.get("/static/ragdemo.js", include_in_schema=False)
async def javascript() -> FileResponse:
    return FileResponse(_STATIC_DIR / "ragdemo.js", media_type="text/javascript")


@router.get("/assets/bootstrap.css", include_in_schema=False)
async def bootstrap_stylesheet() -> FileResponse:
    return FileResponse(_CONFIGURATOR_STATIC_DIR / "bootstrap.css", media_type="text/css")


@router.get("/assets/configurator.css", include_in_schema=False)
async def configurator_stylesheet() -> FileResponse:
    return FileResponse(_CONFIGURATOR_STATIC_DIR / "main.css", media_type="text/css")


@router.get("/assets/learn2rag-logo.png", include_in_schema=False)
async def learn2rag_logo() -> FileResponse:
    return FileResponse(_CONFIGURATOR_STATIC_DIR / "images" / "logo_learn2rag_logo.png", media_type="image/png")


@router.get("/assets/bmwi.svg", include_in_schema=False)
async def bmwi_logo() -> FileResponse:
    return FileResponse(_CONFIGURATOR_STATIC_DIR / "images" / "BMWi.svg", media_type="image/svg+xml")


@router.get(
    "/api/index",
    response_model=IndexResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": IndexErrorResponse}},
)
async def index_api() -> IndexResponse | JSONResponse:
    try:
        collection_name = user_config["collection_name"]
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("A valid collection_name is required")
        return await run_in_threadpool(
            inspect_index,
            Qdrant.client,
            collection_name,
        )
    except Exception:
        logger.exception("Unable to inspect the configured Qdrant collection")
        error = IndexErrorResponse(
            message="The RAG index is temporarily unavailable. Please try again shortly."
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error.model_dump(),
        )
