import logging
from asyncio import to_thread
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Request, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from learn2rag.pipeline.config import opt_config, user_config
from learn2rag.pipeline.operators.search import SearchOperator
from learn2rag.pipeline.qdrant import Qdrant

from .models import (
    IndexedDocumentChunksResponse,
    IndexErrorResponse,
    IndexResponse,
    PublicExampleQuestions,
    QueryErrorResponse,
    QueryRequest,
    QueryResponse,
)
from .service import (
    execute_query,
    inspect_document_chunks,
    inspect_index,
    load_public_example_questions,
)

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _BASE_DIR / "static"
_CONFIGURATOR_STATIC_DIR = _BASE_DIR.parent / "ui" / "static"
# Expose only the explicitly routed shared assets below; mounting the entire
# Configurator static tree would unnecessarily widen the public demo surface.
templates = Jinja2Templates(directory=_BASE_DIR / "templates")

router = APIRouter()
demo_search_operator = SearchOperator()
# The supplemental explorer and demo-local keyword retrieval share this
# read-only boundary; deployment must aim it at the dedicated public demo index.
demo_qdrant_reader = Qdrant.client


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


@router.get(
    "/api/examples",
    response_model=PublicExampleQuestions,
    responses={503: {"model": IndexErrorResponse}},
)
async def examples_api() -> PublicExampleQuestions | JSONResponse:
    try:
        return load_public_example_questions()
    except Exception:
        logger.exception("Unable to load demo examples")
        return JSONResponse(status_code=503, content=IndexErrorResponse(
            message="Example questions are temporarily unavailable."
        ).model_dump())


@router.get(
    "/api/index/{document_id}/chunks",
    response_model=IndexedDocumentChunksResponse,
    responses={404: {"model": IndexErrorResponse}, 503: {"model": IndexErrorResponse}},
)
async def document_chunks_api(document_id: str) -> IndexedDocumentChunksResponse | JSONResponse:
    try:
        collection_name = user_config["collection_name"]
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("A valid collection_name is required")
        result = await to_thread(
            inspect_document_chunks, demo_qdrant_reader, collection_name, document_id
        )
        if result is not None:
            return result
        return JSONResponse(status_code=404, content=IndexErrorResponse(
            message="The indexed document was not found."
        ).model_dump())
    except Exception:
        logger.exception("Unable to inspect demo document chunks")
        return JSONResponse(status_code=503, content=IndexErrorResponse(
            message="Document chunks are temporarily unavailable. Please try again shortly."
        ).model_dump())


@router.post(
    "/api/query",
    response_model=QueryResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": QueryErrorResponse}},
)
async def query_api(query: QueryRequest) -> QueryResponse | JSONResponse:
    try:
        return await execute_query(
            demo_search_operator,
            query.question,
            opt_config,
            retrieval_mode=query.retrieval_mode,
            qdrant_reader=demo_qdrant_reader,
            collection_name=user_config.get("collection_name"),
        )
    except Exception:
        logger.exception("Unable to execute the RAG Demo query pipeline")
        error = QueryErrorResponse(
            message="The RAG query is temporarily unavailable. Please try again shortly."
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error.model_dump(),
        )
