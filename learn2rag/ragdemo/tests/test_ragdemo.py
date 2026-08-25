from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import BaseMessage, SystemMessage
from qdrant_client.http.models import ScoredPoint

import learn2rag.pipeline.generate as pipeline_generate
from learn2rag.ragdemo import router
from learn2rag.ragdemo import routes, service
from learn2rag.ragdemo.service import DEMO_USER, inspect_index


DEMO_CONFIG = {
    "search_mode": "dense",
    "embedding_model": "BAAI/bge-m3",
    "prompt": "Use only the following information:\n{context}",
}


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


class FakeSearchOperator:
    def __init__(self, documents: Any = None, error: Exception | None = None) -> None:
        self.documents = documents
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, inputs: dict[str, Any], prov: Any = None) -> dict[str, Any]:
        self.calls.append(inputs)
        if self.error is not None:
            raise self.error
        return {"documents": self.documents}


class CapturingModel:
    def __init__(self, answer: Any, error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.calls: list[list[BaseMessage]] = []

    def invoke(self, messages: list[BaseMessage]) -> SimpleNamespace:
        self.calls.append(list(messages))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content=self.answer)


def _install_query_boundaries(
    monkeypatch: Any,
    documents: Any,
    answer: Any,
) -> tuple[FakeSearchOperator, CapturingModel]:
    search_operator = FakeSearchOperator(documents=documents)
    model = CapturingModel(answer=answer)
    monkeypatch.setattr(routes, "demo_search_operator", search_operator)
    monkeypatch.setattr(routes, "opt_config", DEMO_CONFIG)
    monkeypatch.setattr(pipeline_generate, "llm", model)
    return search_operator, model


def _serialized_captured_messages(model: CapturingModel) -> list[dict[str, str]]:
    assert len(model.calls) == 1
    serialized = []
    for message in model.calls[0]:
        assert isinstance(message.content, str)
        serialized.append({"role": message.type, "content": message.content})
    return serialized


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


@pytest.mark.anyio
async def test_ask_rag_page_contains_query_and_result_containers() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=_app()),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/ragdemo/")

    assert response.status_code == 200
    assert 'id="question-input"' in response.text
    assert 'id="ask-button"' in response.text
    assert 'id="answer-container"' in response.text
    assert 'id="search-results"' in response.text
    assert 'id="prompt-messages"' in response.text
    assert "Prompt sent to the model" in response.text
    assert "Application-level chat messages" in response.text
    assert "<details" in response.text
    assert "Prompt inspection will be added in the next phase." not in response.text


def test_prompt_frontend_uses_safe_text_insertion() -> None:
    javascript = (
        Path(__file__).resolve().parents[1] / "static" / "ragdemo.js"
    ).read_text(encoding="utf-8")

    assert "messageContent.textContent = message.content;" in javascript
    assert "promptNote.textContent = prompt.note;" in javascript
    assert "innerHTML" not in javascript


@pytest.mark.anyio
async def test_valid_query_returns_answer_and_same_retrieved_chunks(monkeypatch: Any) -> None:
    raw_source = "/home/example/Documents/RAG/demo_event.txt"
    documents = [
        _point(
            "qdrant-uuid-1",
            content="The demo is intended for Forum Digitale Technologien in Berlin.",
            source=raw_source,
            loader_id="private-loader-id",
            content_hash="private-content-hash",
            chunk_hash="private-chunk-hash",
        ),
        _point(
            "qdrant-uuid-2",
            content="A second retrieved chunk.",
            source="/home/example/Documents/RAG/search_modes.txt",
            loader_id="private-loader-id",
            content_hash="second-private-hash",
        ),
    ]
    documents[0].score = 0.7258955
    documents[1].score = 0.5609804
    search_operator, model = _install_query_boundaries(
        monkeypatch,
        documents,
        f"The supporting source is {raw_source}.",
    )

    async with AsyncClient(
        transport=ASGITransport(app=_app()),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/ragdemo/api/query",
            json={"question": "  Where is the demo intended to be shown?  "},
        )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"question", "answer", "search", "prompt"}
    assert body["question"] == "Where is the demo intended to be shown?"
    assert body["answer"] == "The supporting source is demo_event.txt."
    assert body["search"]["mode"] == "dense"
    assert body["search"]["label"] == "Semantic vector search"
    assert body["search"]["technical_label"] == "Dense / BGE-M3"
    assert body["search"]["score_label"] == "Dense similarity"
    assert [result["content"] for result in body["search"]["results"]] == [
        document.payload["content"] for document in documents
    ]
    assert body["search"]["results"][0]["source"] == "demo_event.txt"
    assert body["search"]["results"][0]["score"] == 0.7258955
    assert isinstance(body["search"]["results"][0]["score"], float)
    assert set(body["search"]["results"][0]) == {
        "rank",
        "id",
        "source",
        "score",
        "content",
    }
    assert body["prompt"]["label"] == "Prompt sent to the model"
    assert body["prompt"]["technical_label"] == "Application-level chat messages"
    assert "Provider-specific serialization or chat templates are not shown" in body["prompt"]["note"]
    assert [message["role"] for message in body["prompt"]["messages"]] == [
        "system",
        "human",
    ]
    assert body["prompt"]["messages"] == _serialized_captured_messages(model)
    assert len(search_operator.calls) == 1
    assert search_operator.calls[0] == {
        "question": "Where is the demo intended to be shown?",
        "user": DEMO_USER,
    }
    assert len(model.calls) == 1

    system_content = model.calls[0][0].content
    assert isinstance(system_content, str)
    first_source_position = system_content.index("Source: demo_event.txt")
    first_chunk_position = system_content.index(documents[0].payload["content"])
    second_source_position = system_content.index("Source: search_modes.txt")
    second_chunk_position = system_content.index(documents[1].payload["content"])
    assert (
        first_source_position
        < first_chunk_position
        < second_source_position
        < second_chunk_position
    )
    assert model.calls[0][1].content == "Where is the demo intended to be shown?"

    serialized = response.text
    assert raw_source not in serialized
    assert raw_source not in system_content
    assert "demo_event.txt" in system_content
    assert "private-loader-id" not in serialized
    assert "private-content-hash" not in serialized
    assert "private-chunk-hash" not in serialized
    assert "qdrant-uuid-1" not in serialized


@pytest.mark.anyio
async def test_query_answer_sanitizes_credential_bearing_url(monkeypatch: Any) -> None:
    raw_url = "https://user:secret@example.com/docs/report.pdf?token=abc"
    document = _point(
        "qdrant-uuid-url",
        content="A report chunk.",
        source=raw_url,
    )
    document.score = 0.8123
    search_operator, model = _install_query_boundaries(
        monkeypatch,
        [document],
        f"See {raw_url} for the supporting report.",
    )

    async with AsyncClient(
        transport=ASGITransport(app=_app()),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/ragdemo/api/query",
            json={"question": "Which report supports the answer?"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "See report.pdf — example.com for the supporting report."
    assert body["search"]["results"][0]["source"] == "report.pdf — example.com"
    assert body["prompt"]["messages"] == _serialized_captured_messages(model)
    system_content = model.calls[0][0].content
    assert isinstance(system_content, str)
    assert "Source: report.pdf — example.com" in system_content
    assert raw_url not in system_content
    assert "user:secret" not in system_content
    assert "token=abc" not in system_content
    assert "user:secret" not in response.text
    assert "token=abc" not in response.text
    assert len(search_operator.calls) == 1
    assert len(model.calls) == 1


@pytest.mark.anyio
async def test_query_answer_without_raw_sources_is_unchanged(monkeypatch: Any) -> None:
    answer = "The answer contains no source reference and stays exactly the same."
    document = _point(
        "qdrant-uuid-plain",
        content="A retrieved chunk.",
        source="/home/example/Documents/RAG/demo_event.txt",
    )
    document.score = 0.7
    search_operator, model = _install_query_boundaries(monkeypatch, [document], answer)

    async with AsyncClient(
        transport=ASGITransport(app=_app()),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/ragdemo/api/query",
            json={"question": "What does the document say?"},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == answer
    assert len(search_operator.calls) == 1
    assert len(model.calls) == 1


@pytest.mark.anyio
async def test_whitespace_query_is_rejected_before_search(monkeypatch: Any) -> None:
    search_operator = FakeSearchOperator(documents=[])
    monkeypatch.setattr(routes, "demo_search_operator", search_operator)

    async with AsyncClient(
        transport=ASGITransport(app=_app()),
        base_url="http://testserver",
    ) as client:
        response = await client.post("/ragdemo/api/query", json={"question": "   \n\t  "})

    assert response.status_code == 422
    assert search_operator.calls == []


@pytest.mark.anyio
async def test_query_retrieval_failure_returns_safe_generic_error(monkeypatch: Any) -> None:
    secret = "LLM failed with token=super-secret"
    search_operator = FakeSearchOperator(error=RuntimeError(secret))
    monkeypatch.setattr(routes, "demo_search_operator", search_operator)

    async with AsyncClient(
        transport=ASGITransport(app=_app()),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/ragdemo/api/query",
            json={"question": "Where is the demo?"},
        )

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "message": "The RAG query is temporarily unavailable. Please try again shortly.",
    }
    assert secret not in response.text
    assert len(search_operator.calls) == 1


@pytest.mark.anyio
async def test_query_model_failure_returns_safe_generic_error(monkeypatch: Any) -> None:
    secret = "model invocation failed with api_key=super-secret"
    document = _point(
        "qdrant-uuid-model-error",
        content="A retrieved chunk.",
        source="/private/demo_event.txt",
    )
    document.score = 0.7
    search_operator = FakeSearchOperator(documents=[document])
    model = CapturingModel(answer=None, error=RuntimeError(secret))
    monkeypatch.setattr(routes, "demo_search_operator", search_operator)
    monkeypatch.setattr(routes, "opt_config", DEMO_CONFIG)
    monkeypatch.setattr(pipeline_generate, "llm", model)

    async with AsyncClient(
        transport=ASGITransport(app=_app()),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/ragdemo/api/query",
            json={"question": "Where is the demo?"},
        )

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "message": "The RAG query is temporarily unavailable. Please try again shortly.",
    }
    assert secret not in response.text
    assert len(search_operator.calls) == 1
    assert len(model.calls) == 1


@pytest.mark.anyio
async def test_non_text_prompt_content_returns_safe_generic_error(monkeypatch: Any) -> None:
    secret = "prompt-content-secret"
    document = _point(
        "qdrant-uuid-prompt-error",
        content="A retrieved chunk.",
        source="/private/demo_event.txt",
    )
    document.score = 0.7
    search_operator = FakeSearchOperator(documents=[document])
    model = CapturingModel(answer="unused")

    def build_non_text_prompt(*args: Any, **kwargs: Any) -> list[BaseMessage]:
        return [SystemMessage(content=[{"type": "text", "text": secret}])]

    monkeypatch.setattr(routes, "demo_search_operator", search_operator)
    monkeypatch.setattr(routes, "opt_config", DEMO_CONFIG)
    monkeypatch.setattr(service, "build_prompt_messages", build_non_text_prompt)
    monkeypatch.setattr(pipeline_generate, "llm", model)

    async with AsyncClient(
        transport=ASGITransport(app=_app()),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/ragdemo/api/query",
            json={"question": "Where is the demo?"},
        )

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "message": "The RAG query is temporarily unavailable. Please try again shortly.",
    }
    assert secret not in response.text
    assert len(search_operator.calls) == 1
    assert model.calls == []


def test_generate_default_prompt_behavior_keeps_raw_source(monkeypatch: Any) -> None:
    raw_source = "/home/example/Documents/RAG/demo_event.txt"
    content = "The original chunk content remains unchanged."
    document = ScoredPoint(
        id="qdrant-uuid-generate",
        version=0,
        score=0.7,
        payload={"source": raw_source, "content": content},
    )
    model = CapturingModel(answer="Generated answer")
    monkeypatch.setattr(pipeline_generate, "llm", model)

    answer = pipeline_generate.generate(
        "What does the document say?",
        [document],
        DEMO_CONFIG,
    )

    expected_context = pipeline_generate.context_template.format(
        source=raw_source,
        content=content,
    )
    assert answer == "Generated answer"
    assert len(model.calls) == 1
    assert [message.type for message in model.calls[0]] == ["system", "human"]
    assert model.calls[0][0].content == DEMO_CONFIG["prompt"].format(
        context=expected_context
    )
    assert model.calls[0][1].content == "What does the document say?"
    assert raw_source in model.calls[0][0].content


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
