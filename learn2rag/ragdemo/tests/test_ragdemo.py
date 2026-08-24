from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from learn2rag.ragdemo import router
from learn2rag.ragdemo import routes
from learn2rag.ragdemo.service import inspect_index


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/ragdemo")
    return app


def _point(point_id: str, **payload: Any) -> SimpleNamespace:
    return SimpleNamespace(id=point_id, payload=payload)


def _client(*pages: tuple[list[SimpleNamespace], Any]) -> MagicMock:
    client = MagicMock()
    client.collection_exists.return_value = True
    client.scroll.side_effect = pages
    return client


@pytest.mark.anyio
async def test_existing_test_route_still_works() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=_app()),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/ragdemo/test")

    assert response.status_code == 200
    assert response.json() == {"test": 123}


@pytest.mark.anyio
async def test_index_page_returns_html() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=_app()),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/ragdemo/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Interactive RAG Demo" in response.text
    assert "Indexed documents" in response.text
    assert "./assets/bootstrap.css" in response.text
    assert "./assets/configurator.css" in response.text
    assert "./assets/learn2rag-logo.png" in response.text
    assert "./assets/bmwi.svg" in response.text
    assert "cdnjs.cloudflare.com" not in response.text
    assert 'href="/models"' not in response.text


def test_index_grouping_uses_scroll_pagination() -> None:
    client = _client(
        (
            [
                _point("1", loader_id="directory", source="/data/guide.pdf", content_hash="hash-a"),
                _point("2", loader_id="directory", source="/data/guide.pdf", content_hash="hash-a"),
            ],
            "next-page",
        ),
        (
            [_point("3", loader_id="directory", source="/data/notes.txt", content_hash="hash-b")],
            None,
        ),
    )

    result = inspect_index(client, "learn2rag")

    assert result.document_count == 2
    assert result.chunk_count == 3
    assert {document.name: document.chunk_count for document in result.documents} == {
        "guide.pdf": 2,
        "notes.txt": 1,
    }
    assert client.scroll.call_count == 2
    assert all(call.kwargs["with_vectors"] is False for call in client.scroll.call_args_list)


def test_same_source_with_different_content_hashes_creates_distinct_groups() -> None:
    source = "/data/versioned-guide.pdf"
    client = _client(
        (
            [
                _point("1", loader_id="directory", source=source, content_hash="hash-a"),
                _point("2", loader_id="directory", source=source, content_hash="hash-b"),
            ],
            None,
        )
    )

    result = inspect_index(client, "learn2rag")

    assert result.document_count == 2
    assert result.chunk_count == 2
    assert len({document.id for document in result.documents}) == 2


def test_absolute_local_path_is_sanitized() -> None:
    source = "/home/example/Documents/RAG/demo_event.txt"
    client = _client(([_point("1", loader_id="directory", source=source)], None))

    result = inspect_index(client, "learn2rag")
    serialized = result.model_dump_json()

    assert result.documents[0].name == "demo_event.txt"
    assert result.documents[0].source_type == "local"
    assert source not in serialized
    assert "/home/example" not in serialized


def test_document_ids_are_deterministic_and_do_not_expose_source() -> None:
    source = "/private/customer/contracts/contract.pdf"
    first = inspect_index(
        _client(([_point("random-point-1", loader_id="directory", source=source)], None)),
        "learn2rag",
    )
    second = inspect_index(
        _client(([_point("random-point-2", loader_id="directory", source=source)], None)),
        "learn2rag",
    )

    assert first.documents[0].id == second.documents[0].id
    assert source not in first.documents[0].id


def test_document_id_is_not_used_as_display_name() -> None:
    document_id = "private-sharepoint-document-id"
    client = _client(([_point("1", loader_id="sharepoint", document_id=document_id)], None))

    result = inspect_index(client, "learn2rag")

    assert result.documents[0].name == "Indexed document"
    assert document_id not in result.model_dump_json()


def test_empty_index_is_handled_cleanly() -> None:
    missing_collection = MagicMock()
    missing_collection.collection_exists.return_value = False

    result = inspect_index(missing_collection, "learn2rag")

    assert result.status == "empty"
    assert result.document_count == 0
    assert result.chunk_count == 0
    assert result.documents == []
    missing_collection.scroll.assert_not_called()


@pytest.mark.anyio
async def test_qdrant_failure_returns_generic_structured_error(monkeypatch: Any) -> None:
    secret_detail = "connection failed with api_key=do-not-leak"

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(secret_detail)

    async def run_inline(function: Any, *args: Any) -> Any:
        return function(*args)

    monkeypatch.setattr(routes, "inspect_index", fail)
    monkeypatch.setattr(routes, "run_in_threadpool", run_inline)
    async with AsyncClient(
        transport=ASGITransport(app=_app()),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/ragdemo/api/index")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "message": "The RAG index is temporarily unavailable. Please try again shortly.",
    }
    assert secret_detail not in response.text


@pytest.mark.anyio
async def test_missing_collection_name_returns_generic_structured_error(monkeypatch: Any) -> None:
    monkeypatch.setattr(routes, "user_config", {})

    async with AsyncClient(
        transport=ASGITransport(app=_app()),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/ragdemo/api/index")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "message": "The RAG index is temporarily unavailable. Please try again shortly.",
    }
    assert "collection_name" not in response.text
