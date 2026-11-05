import json
import math
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import ValidationError
from qdrant_client.http.models import ScoredPoint

import learn2rag.pipeline.generate as pipeline_generate
from learn2rag.ragdemo import router
from learn2rag.ragdemo import routes, service
from learn2rag.ragdemo.models import (
    AggregateQueryRequest,
    AggregateQueryResponse,
    DirectModeResult,
    ExampleQuestions,
    MAX_PUBLIC_CHUNK_CHARS,
    QueryRequest,
    QueryComparisonResults,
    QueryModeResult,
    QuerySearchResult,
)
from learn2rag.ragdemo.service import (
    DEMO_USER,
    build_query_visualization,
    inspect_chunk_details,
    inspect_index,
)


DEMO_CONFIG = {
    "search_mode": "dense",
    "embedding_model": "BAAI/bge-m3",
    "prompt": "Use only the following information:\n{context}",
}
KEYWORD_CONFIG = {**DEMO_CONFIG, "top_k": 5}
AGGREGATE_CONFIG = {
    **KEYWORD_CONFIG,
    "search_mode": "dense_sparse",
    "fusion_mode": "DBSF",
}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/ragdemo")
    return app


async def _run_inline(function: Any, *args: Any) -> Any:
    return function(*args)


def _point(point_id: str, **payload: Any) -> SimpleNamespace:
    return SimpleNamespace(id=point_id, payload=payload)


def _vector_point(point_id: str, vector: Any, **payload: Any) -> SimpleNamespace:
    point = _point(point_id, **payload)
    point.vector = {"dense": vector}
    return point


def _client(*pages: tuple[list[SimpleNamespace], Any]) -> MagicMock:
    client = MagicMock()
    client.collection_exists.return_value = True
    client.scroll.side_effect = pages
    return client


def _collection_schema(*, dense: bool = True, sparse: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors={"dense": object()} if dense else {},
                sparse_vectors={"sparse": object()} if sparse else {},
            )
        )
    )


class FakeSearchOperator:
    def __init__(self, documents: Any = None, error: Exception | None = None) -> None:
        self.documents = documents
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.config_calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    async def __call__(self, inputs: dict[str, Any], prov: Any = None) -> dict[str, Any]:
        self.calls.append(inputs)
        if self.error is not None:
            raise self.error
        return {"documents": self.documents}

    async def run_with_config(
        self,
        inputs: dict[str, Any],
        *,
        user_config: dict[str, Any],
        opt_config: dict[str, Any],
    ) -> dict[str, Any]:
        self.config_calls.append((user_config, opt_config))
        return await self(inputs)


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
    qdrant_reader = _client(([], None))
    monkeypatch.setattr(routes, "demo_qdrant_reader", qdrant_reader)
    monkeypatch.setattr(pipeline_generate, "llm", model)
    return search_operator, model


def _serialized_captured_messages(model: CapturingModel) -> list[dict[str, str]]:
    assert len(model.calls) == 1
    serialized = []
    for message in model.calls[0]:
        assert isinstance(message.content, str)
        serialized.append({"role": message.type, "content": message.content})
    return serialized


def test_query_request_defaults_to_semantic_and_rejects_unknown_modes() -> None:
    request = QueryRequest(question="What is RAG?")

    assert request.retrieval_mode == "semantic"
    with pytest.raises(ValidationError):
        QueryRequest.model_validate(
            {"question": "What is RAG?", "retrieval_mode": "sparse"}
        )

    assert QueryRequest(
        question="What is RAG?", retrieval_mode="hybrid"
    ).retrieval_mode == "hybrid"


def test_aggregate_query_request_accepts_only_one_trimmed_question() -> None:
    request = AggregateQueryRequest(question="  What is RAG?  ")

    assert request.model_dump() == {"question": "What is RAG?"}
    with pytest.raises(ValidationError):
        AggregateQueryRequest.model_validate(
            {"question": "What is RAG?", "search_mode": "dense_sparse"}
        )


@pytest.mark.anyio
async def test_aggregate_query_endpoint_accepts_only_the_question(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    async def execute(*args: Any, **kwargs: Any) -> AggregateQueryResponse:
        captured["question"] = args[1]
        captured["runtime_config"] = args[2]
        captured["kwargs"] = kwargs
        return AggregateQueryResponse(
            question=args[1],
            results=QueryComparisonResults(
                semantic=QueryModeResult(status="unavailable", message="Unavailable."),
                keyword=QueryModeResult(status="unavailable", message="Unavailable."),
                hybrid=QueryModeResult(status="unavailable", message="Unavailable."),
                generative=DirectModeResult(status="unavailable", message="Unavailable."),
            ),
        )

    monkeypatch.setattr(routes, "execute_all_queries", execute)
    async with AsyncClient(
        transport=ASGITransport(app=_app()), base_url="http://test"
    ) as client:
        response = await client.post(
            "/ragdemo/api/query/all",
            json={"question": "What is RAG?"},
        )
        rejected = await client.post(
            "/ragdemo/api/query/all",
            json={"question": "What is RAG?", "top_k": 99},
        )

    assert response.status_code == 200
    assert response.json()["question"] == "What is RAG?"
    assert rejected.status_code == 422
    assert captured["question"] == "What is RAG?"
    assert captured["runtime_config"] is routes.opt_config
    assert captured["kwargs"]["collection_name"] == routes.user_config.get("collection_name")


@pytest.mark.anyio
async def test_aggregate_query_prepares_four_isolated_modes(monkeypatch: Any) -> None:
    semantic_point = _point(
        "semantic-private-id",
        source="/private/semantic.pdf",
        content="Semantic Berlin evidence",
    )
    semantic_point.score = 0.91
    hybrid_point = _point(
        "hybrid-private-id",
        source="/private/hybrid.pdf",
        content="Hybrid Berlin evidence",
    )
    hybrid_point.score = 0.73
    keyword_record = _point(
        "keyword-private-id",
        source="/private/keyword.pdf",
        content="Berlin keyword evidence",
    )

    class ModeSearchOperator(FakeSearchOperator):
        async def run_with_config(
            self,
            inputs: dict[str, Any],
            *,
            user_config: dict[str, Any],
            opt_config: dict[str, Any],
        ) -> dict[str, Any]:
            self.calls.append(inputs)
            self.config_calls.append((user_config, opt_config))
            documents = (
                [semantic_point]
                if opt_config["search_mode"] == "dense"
                else [hybrid_point]
            )
            return {"documents": documents}

    search_operator = ModeSearchOperator()
    model = CapturingModel(answer="Prepared answer")
    client = _client(([keyword_record], None))
    client.get_collection.return_value = _collection_schema(sparse=True)
    monkeypatch.setattr(pipeline_generate, "llm", model)
    monkeypatch.setattr(
        service,
        "filter_authorized",
        lambda user, response: _run_inline(lambda: list(response.points)),
    )
    monkeypatch.setattr(
        service,
        "_safe_query_visualization",
        lambda *args, **kwargs: service._visualization_status(
            "unavailable", "The embedding map is temporarily unavailable."
        ),
    )

    response = await service.execute_all_queries(
        search_operator,
        "What happened in Berlin?",
        AGGREGATE_CONFIG,
        qdrant_reader=client,
        collection_name="configured-demo",
        runtime_user_config={"collection_name": "configured-demo"},
    )

    assert response.question == "What happened in Berlin?"
    assert response.results.semantic.status == "ready"
    assert response.results.keyword.status == "ready"
    assert response.results.hybrid.status == "ready"
    assert response.results.generative.status == "ready"
    assert [config["search_mode"] for _, config in search_operator.config_calls] == [
        "dense",
        "dense_sparse",
    ]
    assert AGGREGATE_CONFIG["search_mode"] == "dense_sparse"
    assert len(model.calls) == 4

    semantic = response.results.semantic.result
    keyword = response.results.keyword.result
    hybrid = response.results.hybrid.result
    direct = response.results.generative.result
    assert semantic is not None and keyword is not None and hybrid is not None
    assert direct is not None
    assert semantic.search.mode == "semantic"
    assert keyword.search.mode == "keyword"
    assert hybrid.search.mode == "hybrid"
    assert semantic.search.results[0].content == "Semantic Berlin evidence"
    assert keyword.search.results[0].content == "Berlin keyword evidence"
    assert hybrid.search.results[0].content == "Hybrid Berlin evidence"
    assert "Semantic Berlin evidence" in semantic.prompt.messages[0].content
    assert "Berlin keyword evidence" in keyword.prompt.messages[0].content
    assert "Hybrid Berlin evidence" in hybrid.prompt.messages[0].content
    assert direct.retrieval_used is False
    assert direct.indexed_documents_used is False
    assert not hasattr(direct, "search")
    assert not hasattr(direct, "visualization")
    assert direct.prompt.messages[0].content == service.DIRECT_GENERATION_SYSTEM_PROMPT
    assert direct.prompt.messages[1].content == "What happened in Berlin?"
    assert "evidence" not in direct.prompt.messages[0].content.casefold()


@pytest.mark.anyio
async def test_aggregate_failure_is_mode_local_and_safe(monkeypatch: Any) -> None:
    point = _point("private-id", source="/private/source.pdf", content="Berlin evidence")
    point.score = 0.8

    class HybridFailingOperator(FakeSearchOperator):
        async def run_with_config(
            self,
            inputs: dict[str, Any],
            *,
            user_config: dict[str, Any],
            opt_config: dict[str, Any],
        ) -> dict[str, Any]:
            self.config_calls.append((user_config, opt_config))
            if opt_config["search_mode"] == "dense_sparse":
                raise RuntimeError("secret hybrid backend failure")
            return {"documents": [point]}

    client = _client(([point], None))
    client.get_collection.return_value = _collection_schema(sparse=True)
    model = CapturingModel(answer="Available answer")
    monkeypatch.setattr(pipeline_generate, "llm", model)
    monkeypatch.setattr(
        service,
        "filter_authorized",
        lambda user, response: _run_inline(lambda: list(response.points)),
    )
    monkeypatch.setattr(
        service,
        "_safe_query_visualization",
        lambda *args, **kwargs: service._visualization_status(
            "unavailable", "The embedding map is temporarily unavailable."
        ),
    )

    response = await service.execute_all_queries(
        HybridFailingOperator(),
        "Berlin?",
        AGGREGATE_CONFIG,
        qdrant_reader=client,
        collection_name="configured-demo",
    )

    assert response.results.semantic.status == "ready"
    assert response.results.keyword.status == "ready"
    assert response.results.hybrid.status == "unavailable"
    assert response.results.generative.status == "ready"
    assert response.results.hybrid.message == "The Hybrid result is temporarily unavailable."
    assert "secret" not in response.model_dump_json()
    assert len(model.calls) == 3


def test_direct_query_invokes_model_without_any_retrieval_context(monkeypatch: Any) -> None:
    model = CapturingModel(answer="Direct answer")
    monkeypatch.setattr(pipeline_generate, "llm", model)

    result = service.execute_direct_query("What is the Berlin Process?")

    assert result.answer == "Direct answer"
    assert len(model.calls) == 1
    assert [message.content for message in model.calls[0]] == [
        service.DIRECT_GENERATION_SYSTEM_PROMPT,
        "What is the Berlin Process?",
    ]
    assert "Information:" not in result.prompt.messages[0].content
    assert result.retrieval_used is False


def test_capability_uses_persisted_dense_and_sparse_schema() -> None:
    dense_client = MagicMock()
    dense_client.collection_exists.return_value = True
    dense_client.get_collection.return_value = _collection_schema(sparse=False)
    hybrid_client = MagicMock()
    hybrid_client.collection_exists.return_value = True
    hybrid_client.get_collection.return_value = _collection_schema(sparse=True)

    assert service.retrieval_capabilities(
        dense_client, "configured-demo"
    ).retrieval_modes == ["semantic", "keyword"]
    assert service.retrieval_capabilities(
        hybrid_client, "configured-demo"
    ).retrieval_modes == ["semantic", "keyword", "hybrid"]


def test_capability_missing_or_failed_collection_is_safely_dense_only() -> None:
    missing = MagicMock()
    missing.collection_exists.return_value = False
    failed = MagicMock()
    failed.collection_exists.side_effect = RuntimeError(
        "secret host /private/path collection=private"
    )

    assert service.retrieval_capabilities(
        missing, "configured-demo"
    ).retrieval_modes == ["semantic", "keyword"]
    assert service.retrieval_capabilities(
        failed, "configured-demo"
    ).retrieval_modes == ["semantic", "keyword"]
    missing.get_collection.assert_not_called()


@pytest.mark.anyio
async def test_capability_api_uses_server_collection_and_hides_failures(
    monkeypatch: Any,
) -> None:
    reader = MagicMock()
    reader.collection_exists.return_value = True
    reader.get_collection.return_value = _collection_schema(sparse=True)
    monkeypatch.setattr(routes, "demo_qdrant_reader", reader)
    monkeypatch.setattr(routes, "user_config", {"collection_name": "configured-demo"})
    monkeypatch.setattr(routes, "run_in_threadpool", _run_inline)

    async with AsyncClient(
        transport=ASGITransport(app=_app()),
        base_url="http://testserver",
    ) as client:
        response = await client.get(
            "/ragdemo/api/capabilities?collection_name=private"
        )

    assert response.status_code == 200
    assert response.json() == {
        "retrieval_modes": ["semantic", "keyword", "hybrid"]
    }
    reader.get_collection.assert_called_once_with("configured-demo")
    assert "configured-demo" not in response.text


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
    assert "./assets/BMFTR.png" in response.text
    assert "https://www.bmftr.bund.de/" in response.text
    assert "BMFTR funding logo" in response.text
    assert "bmwi.svg" not in response.text.lower()
    assert "BMWK funding logo" not in response.text
    assert "cdnjs.cloudflare.com" not in response.text
    assert 'href="/models"' not in response.text


@pytest.mark.anyio
async def test_bmftr_asset_is_served_only_by_the_explicit_demo_route() -> None:
    response = await routes.bmftr_logo()
    asset_path = Path(response.path)
    route_paths = {getattr(route, "path", "") for route in router.routes}

    assert response.status_code == 200
    assert response.media_type == "image/png"
    assert asset_path == Path(routes.__file__).resolve().parent / "static/images/BMFTR.png"
    assert asset_path.read_bytes().startswith(b"\xff\xd8\xff")
    assert "/assets/BMFTR.png" in route_paths
    assert "/assets/bmwi.svg" not in route_paths
    assert not any("{path" in path for path in route_paths)


def test_phase12_readme_documents_the_supported_demo_workflow() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )

    assert "uv sync --no-install-project" in readme
    assert "./start" in readme
    assert "http://localhost:9000" in readme
    assert "http://localhost:<pipeline-port>/ragdemo/" in readme
    assert "not necessarily `9003`" in readme
    assert "demo_questions.json" in readme
    assert "One question prepares Semantic, Keyword, Hybrid, and Generative AI" in readme
    assert "does not rerun retrieval or generation" in readme
    assert "uses no retrieved documents" in readme
    assert "PCA, t-SNE, and UMAP" in readme
    assert "visualization-only" in readme
    assert "`dense_sparse` is the intended default" in readme
    assert "existing dense-only collections" in readme
    assert "reindexed" in readme
    assert "opt_config.json" not in readme
    assert "EN / DE" in readme
    assert "Indexed source documents are never modified" in readme
    assert "changes static interface text only" in readme
    assert "Retrieved chunks, example questions, user questions, prompts" in readme
    assert "multilingual embedding" not in readme
    assert "mixed or indeterminate corpus" not in readme
    assert "restart the pipeline" in readme.lower()
    assert "Ctrl + F5" in readme


@pytest.mark.anyio
async def test_ask_rag_page_contains_query_and_result_containers() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=_app()),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/ragdemo/")

    assert response.status_code == 200
    assert 'id="question-input"' in response.text
    assert 'name="retrieval_mode"' not in response.text
    assert "One question prepares Semantic, Keyword, Hybrid and Generative AI results." in response.text
    assert 'id="ask-button"' in response.text
    assert 'id="retrieval-change-note"' not in response.text
    assert 'id="comparison-actions"' in response.text
    assert 'id="comparison-buttons"' in response.text
    assert 'class="comparison-card-grid"' in response.text
    assert "Compare results" in response.text
    assert 'id="generative-result"' in response.text
    assert 'id="direct-answer-content"' in response.text
    assert 'id="rag-stages"' in response.text
    assert 'id="answer-container"' in response.text
    assert 'id="search-results"' in response.text
    assert 'id="visualization-canvas"' in response.text
    assert 'id="visualization-zoom-in"' in response.text
    assert 'id="visualization-zoom-out"' in response.text
    assert 'id="visualization-zoom-level"' in response.text
    assert "Current embedding space zoom: 100%" in response.text
    assert 'id="visualization-reset"' in response.text
    assert "Explore the embedding space" in response.text
    assert "3D PCA projection of dense embeddings" in response.text
    assert 'id="projection-selector"' in response.text
    assert 'data-projection="pca"' in response.text
    assert 'data-projection="tsne"' in response.text
    assert 'data-projection="umap"' in response.text
    assert 'id="prompt-messages"' in response.text
    assert "Prompt sent to the model" in response.text
    assert "Application-level chat messages" in response.text
    assert "<details" in response.text
    assert 'data-i18n="embed_question">Embed question' in response.text
    assert 'data-i18n="retrieve_chunks">Retrieve top-ranked chunks' in response.text
    assert 'data-i18n="add_chunks_prompt">Add chunks to prompt' in response.text
    assert 'data-i18n="generate_answer">Generate answer' in response.text
    assert '<details id="answer-stage" class="rag-stage" open>' in response.text
    assert response.text.index('id="embed-stage"') < response.text.index('id="retrieve-stage"')
    assert response.text.index('id="retrieve-stage"') < response.text.index('id="prompt-stage"')
    assert response.text.index('id="prompt-stage"') < response.text.index('id="answer-stage"')
    assert response.text.index('id="retrieve-stage"') < response.text.index('id="search-results"') < response.text.index('id="prompt-stage"')
    assert response.text.index('id="prompt-stage"') < response.text.index('id="prompt-messages"') < response.text.index('id="answer-stage"')
    assert response.text.index('id="answer-stage"') < response.text.index('id="answer-content"')
    assert response.text.count('id="search-results"') == 1
    assert response.text.count('id="prompt-messages"') == 1
    assert response.text.count('id="answer-content"') == 1
    assert "embedding-process-summary" not in response.text
    assert "Prompt inspection will be added in the next phase." not in response.text


def test_prompt_frontend_uses_safe_text_insertion() -> None:
    javascript = (
        Path(__file__).resolve().parents[1] / "static" / "ragdemo.js"
    ).read_text(encoding="utf-8")

    assert "messageContent.textContent = message.content;" in javascript
    assert "setTranslatedText(promptNote, prompt.note);" in javascript
    assert "innerHTML" not in javascript


def test_localization_contract_covers_static_ui_and_preserves_data_content() -> None:
    ragdemo_dir = Path(__file__).resolve().parents[1]
    template = (ragdemo_dir / "templates" / "index.html").read_text(
        encoding="utf-8"
    )
    javascript = (ragdemo_dir / "static" / "ragdemo.js").read_text(
        encoding="utf-8"
    )
    language_switch = javascript[
        javascript.index("function setLanguage"):
        javascript.index('  const tabs = document.querySelectorAll')
    ]

    assert '<html lang="en">' in template
    assert 'data-language="en"' in template
    assert 'data-language="de"' in template
    assert 'data-i18n="ask_rag"' in template
    assert 'data-i18n="question_prepares_all"' in template
    assert 'data-i18n="compare_loaded_results"' in template
    assert 'data-i18n-aria-label="language_selector"' in template
    assert 'data-i18n-aria-label="close_chunk_details"' in template
    assert 'data-i18n-title="zoom_out"' in template
    assert 'data-i18n-alt="bmftr_logo_alt"' in template
    assert 'const languageStorageKey = "learn2rag-ragdemo-language";' in javascript
    assert "en: Object.freeze" in javascript
    assert "de: Object.freeze" in javascript
    assert 'stored === "de" || stored === "en" ? stored : "en"' in javascript
    assert 'document.documentElement.lang = currentLanguage;' in language_switch
    assert 'localStorage.setItem(languageStorageKey, currentLanguage);' in language_switch
    assert "fetch(" not in language_switch
    assert "requestSubmit" not in language_switch
    assert "renderVisualization" not in language_switch
    assert "renderViewerScene" not in language_switch
    assert "questionInput.value" not in language_switch
    assert "retrievalModeInputs" not in language_switch
    assert "innerHTML" not in javascript
    assert "messageContent.textContent = message.content;" in javascript
    assert "answerText.textContent = data.answer;" in javascript
    assert "chunk.textContent = result.content;" in javascript
    assert "button.textContent = question;" in javascript
    assert "questionInput.value = question;" in javascript
    assert "text.textContent = chunk.content;" in javascript
    assert "messageContent.textContent = message.content;" in javascript
    assert "source.textContent = result.source;" in javascript
    assert "details.content" in javascript
    assert './api/translate' not in javascript
    assert "registerTranslation" not in javascript
    assert "translation_unavailable" not in javascript
    assert "prompt-translated-view" not in template
    assert 'setLocalizedText(visualizationInspectorTitle, "chunk_details")' in javascript
    assert re.search(r"setLocalizedText\(\s*retrievalStatus,", javascript)
    assert 'point.retrieved ? "retrieved_rank" : "not_retrieved"' in javascript
    assert '"semantic_similarity"' in javascript
    assert '"fusion_score"' in javascript
    assert '"bm25_score"' in javascript


def test_runtime_translation_and_retrieval_bridge_are_absent() -> None:
    javascript = (
        Path(__file__).resolve().parents[1] / "static" / "ragdemo.js"
    ).read_text(encoding="utf-8")
    service_source = (
        Path(__file__).resolve().parents[1] / "service.py"
    ).read_text(encoding="utf-8")
    routes_source = (
        Path(__file__).resolve().parents[1] / "routes.py"
    ).read_text(encoding="utf-8")

    assert "translationCache" not in javascript
    assert "requestDisplayTranslations" not in javascript
    assert 'fetch("./api/translate"' not in javascript
    assert 'fetch("./api/query/all"' in javascript
    assert "translate_texts" not in service_source
    assert "question_for_retrieval" not in service_source
    assert "detect_corpus_language" not in service_source
    assert '"/api/translate"' not in routes_source
    assert "innerHTML" not in javascript


def test_language_switch_executes_without_changing_demo_state() -> None:
    """Exercise initialization, persistence, and switching in the actual script."""
    import subprocess

    javascript_path = Path(__file__).resolve().parents[1] / "static/ragdemo.js"
    harness = r'''
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

class ClassList {
  constructor(owner) { this.owner = owner; }
  values() { return new Set(this.owner.className.split(/\s+/).filter(Boolean)); }
  toggle(name, force) {
    const values = this.values();
    const enabled = force === undefined ? !values.has(name) : Boolean(force);
    if (enabled) values.add(name); else values.delete(name);
    this.owner.className = [...values].join(" ");
    return enabled;
  }
}

class Element {
  constructor(tag = "div") {
    this.tag = tag; this.children = []; this.events = {}; this.attrs = {};
    this.dataset = {}; this.hidden = false; this.disabled = false; this.value = "";
    this.checked = false; this.className = ""; this._text = "";
    this.classList = new ClassList(this);
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(""); }
  appendChild(child) { this.children.push(child); child.parent = this; return child; }
  append(...children) { children.forEach(child => this.appendChild(child)); }
  replaceChildren(...children) { this.children = []; this._text = ""; this.append(...children); }
  setAttribute(key, value) { this.attrs[key] = String(value); }
  addEventListener(name, handler) { (this.events[name] ||= []).push(handler); }
  async click() { if (!this.disabled) for (const handler of this.events.click || []) await handler({target: this}); }
  querySelectorAll(selector) {
    return this.children.filter(child => selector === ".example-question"
      && child.className.includes("example-question"));
  }
  focus() { this.focused = true; }
  requestSubmit() { throw new Error("Language switching must not submit a RAG query"); }
}

async function run(initialLanguage) {
  const elements = new Map();
  const get = selector => {
    if (!elements.has(selector)) elements.set(selector, new Element());
    return elements.get(selector);
  };
  const semantic = new Element("input"); semantic.value = "semantic";
  const keyword = new Element("input"); keyword.value = "keyword"; keyword.checked = true;
  const hybrid = new Element("input"); hybrid.value = "hybrid";
  const retrievalInputs = [semantic, keyword, hybrid];
  const english = new Element("button"); english.dataset.language = "en";
  const german = new Element("button"); german.dataset.language = "de";
  const languageButtons = [english, german];
  const majorLabel = new Element("h1"); majorLabel.dataset.i18n = "ask_rag";
  const storage = new Map();
  if (initialLanguage !== null) storage.set("learn2rag-ragdemo-language", initialLanguage);
  const calls = [];
  const document = {
    documentElement: {lang: "en"},
    querySelector: get,
    querySelectorAll(selector) {
      if (selector === "input[name='retrieval_mode']") return retrievalInputs;
      if (selector === "[data-language]") return languageButtons;
      if (selector === "[data-i18n]") return [majorLabel];
      return [];
    },
    createElement: tag => new Element(tag),
    createTextNode(text) { const node = new Element("#text"); node.textContent = text; return node; },
    addEventListener() {},
  };
  const localStorage = {
    getItem: key => storage.has(key) ? storage.get(key) : null,
    setItem: (key, value) => storage.set(key, value),
  };
  const context = {
    document,
    localStorage,
      fetch: async (url, options = {}) => {
        calls.push(url);
      if (url === "./api/capabilities") return {ok: true, json: async () => ({retrieval_modes: ["semantic", "keyword"]})};
      if (url === "./api/examples") return {ok: true, json: async () => ({display_count: 1, question_sets: [{questions: ["English dataset question?"]}]})};
          if (url === "./api/index") return {ok: true, json: async () => ({document_count: 0, chunk_count: 0, truncated: false, documents: []})};
          throw new Error(`Unexpected request ${url}`);
    },
  };
  const question = get("#question-input");
  question.value = "Was ist der Berlin-Prozess?";
  get("#query-results").hidden = false;
  get("#answer-content").textContent = "Existing generated answer";
  get("#search-results").textContent = "Rank 1 · 0.875 · Evidence text";
  get("#visualization-canvas").textContent = "Stable geometry";
  vm.runInNewContext(fs.readFileSync(process.argv[1], "utf8"), context);
  await new Promise(resolve => setImmediate(resolve));
  return {document, get, semantic, keyword, english, german, majorLabel, storage, calls};
}

(async () => {
  const fresh = await run(null);
  assert.equal(fresh.document.documentElement.lang, "en");
  assert.equal(fresh.majorLabel.textContent, "Ask RAG");
  assert.equal(fresh.calls.filter(url => url === "./api/query/all").length, 0);

  const demo = await run("invalid");
  assert.equal(demo.document.documentElement.lang, "en");
  assert.equal(demo.majorLabel.textContent, "Ask RAG");
  assert.equal(demo.english.attrs["aria-pressed"], "true");
  assert.equal(demo.calls.filter(url => url === "./api/query/all").length, 0);
  const callsBeforeSwitch = demo.calls.length;
  const questionBefore = demo.get("#question-input").value;
  const answerBefore = demo.get("#answer-content").textContent;
  const resultsBefore = demo.get("#search-results").textContent;
  const geometryBefore = demo.get("#visualization-canvas").textContent;
  await demo.german.click();
  assert.equal(demo.document.documentElement.lang, "de");
  assert.equal(demo.majorLabel.textContent, "RAG fragen");
  assert.equal(demo.storage.get("learn2rag-ragdemo-language"), "de");
  assert.equal(demo.get("#question-input").value, questionBefore);
  assert.equal(demo.get("#answer-content").textContent, answerBefore);
  assert.equal(demo.get("#search-results").textContent, resultsBefore);
  assert.equal(demo.get("#visualization-canvas").textContent, geometryBefore);
  assert.equal(demo.keyword.checked, true);
  assert.equal(demo.semantic.checked, false);
  assert.equal(demo.calls.filter(url => url === "./api/query/all").length, 0);
  await demo.english.click();
  assert.equal(demo.document.documentElement.lang, "en");
  assert.equal(demo.majorLabel.textContent, "Ask RAG");
  assert.equal(demo.calls.filter(url => url === "./api/query/all").length, 0);

  const persisted = await run("de");
  assert.equal(persisted.document.documentElement.lang, "de");
  assert.equal(persisted.majorLabel.textContent, "RAG fragen");
  assert.equal(persisted.german.attrs["aria-pressed"], "true");
  assert.equal(persisted.calls.filter(url => url === "./api/query/all").length, 0);

  assert.equal(demo.get("#example-questions").children[0].textContent, "English dataset question?");
})().catch(error => { console.error(error); process.exitCode = 1; });
'''
    result = subprocess.run(
        ["node", "-e", harness, str(javascript_path)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_loaded_comparison_toggle_never_submits_or_fetches_again() -> None:
    javascript = (
        Path(__file__).resolve().parents[1] / "static" / "ragdemo.js"
    ).read_text(encoding="utf-8")
    stale_clear = javascript[
        javascript.index("function clearComparisonState"):
        javascript.index("function deepFreeze")
    ]
    comparison_handler = javascript[
        javascript.index('comparisonButtons.addEventListener("click"'):
        javascript.index('zoomOutButton.addEventListener("click"')
    ]

    assert "queryResults.hidden = true;" in stale_clear
    assert "comparisonActions.hidden = true;" in stale_clear
    assert "comparisonResultSet = null;" in stale_clear
    assert "retrieval_changed" not in javascript
    assert '["generative", "semantic", "keyword", "hybrid"]' in javascript
    assert "renderComparisonMode(button.dataset.comparisonMode);" in comparison_handler
    assert "requestSubmit" not in comparison_handler
    assert "questionInput.value =" not in comparison_handler
    assert "fetch(" not in comparison_handler
    assert 'fetch("./api/query/all"' in javascript
    assert "const requestVersion = ++queryRequestVersion;" in javascript
    assert "requestVersion !== queryRequestVersion" in javascript
    assert "comparisonResultSet = deepFreeze(data);" in javascript
    assert "innerHTML" not in javascript


def test_aggregate_browser_flow_is_preloaded_immutable_and_truthful() -> None:
    javascript = (
        Path(__file__).resolve().parents[1] / "static" / "ragdemo.js"
    ).read_text(encoding="utf-8")
    template = (
        Path(__file__).resolve().parents[1] / "templates" / "index.html"
    ).read_text(encoding="utf-8")
    submit = javascript[
        javascript.index("async function submitQuery"):
        javascript.index('refreshButton.addEventListener("click"')
    ]
    direct = javascript[
        javascript.index("function renderDirectQueryResponse"):
        javascript.index("function renderComparisonMode")
    ]
    projection_handler = javascript[
        javascript.index("[pcaProjectionButton, tsneProjectionButton, umapProjectionButton]"):
        javascript.index('zoomOutButton.addEventListener("click"')
    ]

    assert submit.count('fetch("./api/query/all"') == 1
    assert "body: JSON.stringify({question})" in submit
    assert "retrieval_mode" not in submit
    assert "clearComparisonState();" in submit
    assert "const requestVersion = ++queryRequestVersion;" in submit
    assert submit.count("requestVersion !== queryRequestVersion") == 2
    assert "comparisonResultSet = deepFreeze(data);" in javascript
    assert '["generative", "semantic", "keyword", "hybrid"]' in javascript
    assert "fetch(" not in projection_handler
    assert 'renderVisualization(null, "generative");' in direct
    assert "ragStages.hidden = true;" in direct
    assert "generativeResult.hidden = false;" in direct
    assert "directAnswerContent.appendChild(answerText);" in direct
    assert "searchContent.hidden = true;" in direct
    assert "searchResults.replaceChildren();" in direct
    assert "renderPrompt(" not in direct
    assert "setStagePresentation(" not in direct
    assert "answerContent" not in direct
    assert "renderSearchResults" not in direct
    assert 'id="generative-result"' in template
    assert 'id="direct-answer-content"' in template
    assert 'id="rag-stages"' in template
    assert 'id="projection-umap"' in template
    assert "Answer without retrieved documents." in javascript
    assert "Antwort ohne abgerufene Dokumente." in javascript


def test_comparison_selector_uses_accessible_responsive_result_cards() -> None:
    ragdemo_dir = Path(__file__).resolve().parents[1]
    javascript = (ragdemo_dir / "static" / "ragdemo.js").read_text(encoding="utf-8")
    stylesheet = (ragdemo_dir / "static" / "ragdemo.css").read_text(encoding="utf-8")
    selector = javascript[
        javascript.index("function configureComparisonSelector"):
        javascript.index("function setStagePresentation")
    ]
    initial_selection = javascript[
        javascript.index("function storeAndRenderComparison"):
        javascript.index("async function submitQuery")
    ]

    assert '["generative", "semantic", "keyword", "hybrid"].forEach' in selector
    assert 'button.type = "button";' in selector
    assert 'button.className = `comparison-card${active ? " is-selected" : ""}`;' in selector
    assert 'button.setAttribute("aria-pressed", String(active));' in selector
    assert 'title.className = "comparison-card-title";' in selector
    assert 'description.className = "comparison-card-description";' in selector
    assert '["generative", "semantic", "keyword", "hybrid"]' in initial_selection
    for text in (
        "Answer without retrieved documents.",
        "Meaning-based dense retrieval.",
        "Keyword-based BM25 retrieval.",
        "Combines dense and sparse retrieval.",
        "Antwort ohne abgerufene Dokumente.",
        "Bedeutungsbasierter dichter Abruf.",
        "Stichwortbasierter BM25-Abruf.",
        "Kombiniert dichten und spärlichen Abruf.",
    ):
        assert text in javascript
    assert "grid-template-columns: repeat(4, minmax(0, 1fr));" in stylesheet
    assert "grid-template-columns: repeat(2, minmax(0, 1fr));" in stylesheet
    assert ".comparison-card.is-selected" in stylesheet
    assert ".comparison-card:focus-visible" in stylesheet


def test_projection_selector_is_semantic_only_and_does_not_change_rag_data() -> None:
    javascript = (
        Path(__file__).resolve().parents[1] / "static" / "ragdemo.js"
    ).read_text(encoding="utf-8")
    selector = javascript[
        javascript.index("function selectVisualizationProjection"):
        javascript.index("function setProjectionControls")
    ]
    renderer = javascript[
        javascript.index("function renderVisualization"):
        javascript.index("function renderPromptMessages")
    ]

    assert '["tsne", "umap"].includes(requestedMethod)' in selector
    assert "points: mergeProjectionPoints(visualization.points, projection)" in selector
    assert "...point, ...coordinatesById.get(point.id)" in selector
    assert 'const isSemantic = searchMode === "semantic";' in renderer
    assert "visualizationSection.hidden = !isSemantic;" in renderer
    for forbidden in ("fetch(", "renderSearchResults", "renderPrompt", "answerContent"):
        assert forbidden not in selector


def test_public_demo_collection_scope_is_documented() -> None:
    service_source = (
        Path(__file__).resolve().parents[1] / "service.py"
    ).read_text(encoding="utf-8")

    assert "dedicated public demo collection" in service_source
    assert "arbitrary private production collection" in service_source


def test_visualization_frontend_has_one_coordinate_graph_stage() -> None:
    ragdemo_dir = Path(__file__).resolve().parents[1]
    template = (ragdemo_dir / "templates" / "index.html").read_text(encoding="utf-8")
    stylesheet = (ragdemo_dir / "static" / "ragdemo.css").read_text(encoding="utf-8")

    assert template.count('id="visualization-panel"') == 1
    assert 'id="visualization-panel" class="embedding-map-stage"' in template
    assert "card embedding-map-card" not in template
    assert template.count('class="embedding-map-stage"') == 1
    assert 'id="visualization-canvas" class="embedding-map-canvas"' in template
    assert 'id="visualization-inspector" class="embedding-map-inspector"' in template
    assert 'id="visualization-inspector-close"' in template
    assert ".embedding-map-canvas" in stylesheet
    assert "height: clamp(38.75rem, 72vh, 45rem);" in stylesheet
    assert "height: clamp(28rem, 68vh, 36rem);" in stylesheet
    stage_rule = stylesheet[
        stylesheet.index(".embedding-map-stage {"):
        stylesheet.index(".visualization-intro")
    ]
    assert stage_rule.count("border:") == 1
    assert "width: calc(100% + 3rem);" in stage_rule
    assert ".embedding-map-background {\n  fill: transparent;\n}" in stylesheet
    assert "Spatial proximity is approximate because PCA compresses" in template
    assert 'id="projection-explanation"' in template
    assert "original full-dimensional vectors" in template
    assert "Query vector" in template
    assert "Retrieved chunks" in template
    assert "Other indexed chunks" in template
    assert "Query → retrieved link" in template
    assert 'id="visualization-expand"' in template
    assert 'id="visualization-restore"' in template
    assert 'id="visualization-expanded-close"' in template
    assert "embedding-process-summary" not in template


def test_visualization_uses_data_focused_fit_and_decorations_do_not_shrink_it() -> None:
    javascript = (
        Path(__file__).resolve().parents[1] / "static" / "ragdemo.js"
    ).read_text(encoding="utf-8")
    geometry = javascript[
        javascript.index("function buildSceneGeometry"):
        javascript.index("function computeFittedSceneScale")
    ]
    fit = javascript[
        javascript.index("function computeFittedSceneScale"):
        javascript.index("function formatCoordinate")
    ]

    assert "const dataAnchors = numericCoordinates.concat({ x: 0, y: 0, z: 0 });" in geometry
    assert "point.x - dataCenter.x" in geometry
    assert "point.y - dataCenter.y" in geometry
    assert "point.z - dataCenter.z" in geometry
    assert "numericCoordinates.concat(axisEndpoints" not in geometry
    assert "gridCorners" not in geometry
    assert "state.cuboid" not in fit
    assert "sceneGeometry.cuboid" not in fit
    assert "axisRanges[" not in fit
    assert "gridLines" not in fit
    assert "Math.min(safeWidth, safeHeight) / 2" in fit
    assert "const scenePadding = 12;" in javascript
    assert "const maximumMarkerExtent = 30;" in javascript
    assert "world.x - state.sceneCenterWorld.x" in javascript


def test_visualization_builds_one_cuboid_and_three_reference_planes() -> None:
    static_dir = Path(__file__).resolve().parents[1] / "static"
    javascript = (static_dir / "ragdemo.js").read_text(encoding="utf-8")
    stylesheet = (static_dir / "ragdemo.css").read_text(encoding="utf-8")

    assert javascript.count('svgElement("g", "embedding-cuboid")') == 1
    assert 'svgElement("circle", "embedding-cuboid-corner")' in javascript
    assert 'svgElement("line", "embedding-cuboid-edge")' in javascript
    assert "const cuboidEdgeIndices = [" in javascript
    assert "coordinateRanges[dimension].minimum" in javascript
    assert "coordinateRanges[dimension].maximum" in javascript
    assert "state.cuboid.edges.forEach" in javascript
    assert "projectWorldPoint(corner.world, state)" in javascript
    assert '"xy"' in javascript
    assert '"xz"' in javascript
    assert '"yz"' in javascript
    assert ".embedding-cuboid-edge" in stylesheet
    assert '.embedding-grid-line[data-plane="xz"]' in stylesheet
    assert '.embedding-grid-line[data-plane="yz"]' in stylesheet


def test_visualization_presentation_mode_reuses_graph_state() -> None:
    ragdemo_dir = Path(__file__).resolve().parents[1]
    template = (ragdemo_dir / "templates" / "index.html").read_text(encoding="utf-8")
    javascript = (ragdemo_dir / "static" / "ragdemo.js").read_text(encoding="utf-8")
    stylesheet = (ragdemo_dir / "static" / "ragdemo.css").read_text(encoding="utf-8")
    presentation = javascript[
        javascript.index("function setPresentationMode"):
        javascript.index("function renderVisualization")
    ]
    expanded_rule = stylesheet[
        stylesheet.index(".embedding-map-stage.is-expanded {"):
        stylesheet.index("body:has")
    ]

    assert "position: fixed;" in expanded_rule
    assert "inset: clamp(0.75rem, 2vw, 1.5rem);" in expanded_rule
    assert "z-index: 1080;" in expanded_rule
    assert "grid-template-rows: auto minmax(0, 1fr) auto;" in expanded_rule
    assert 'visualizationPanel.classList.toggle("is-expanded", expanded);' in presentation
    assert "resizeViewerToCanvas(viewerState);" in presentation
    assert "viewerState =" not in presentation
    assert ".rotation =" not in presentation
    assert ".zoom =" not in presentation
    assert "pinnedChunkId =" not in presentation
    assert 'expandViewButton.addEventListener("click", () => setPresentationMode(true));' in javascript
    assert 'restoreViewButton.addEventListener("click", () => setPresentationMode(false));' in javascript
    assert 'closeExpandedViewButton.addEventListener("click", () => setPresentationMode(false));' in javascript
    assert "Embed question" in template
    assert "Retrieve top-ranked chunks" in template
    assert "Add chunks to prompt" in template
    assert "Generate answer" in template
    assert "embedding-process-summary" not in template


def test_visualization_hides_only_the_redundant_success_note() -> None:
    javascript = (
        Path(__file__).resolve().parents[1] / "static" / "ragdemo.js"
    ).read_text(encoding="utf-8")
    note_filter = javascript[
        javascript.index("function visualizationDisplayNote"):
        javascript.index("function renderVisualization")
    ]

    assert 'if (message === redundantVisualizationNote)' in note_filter
    assert 'return "";' in note_filter
    assert 'message.startsWith(`${redundantVisualizationNote} `)' in note_filter
    assert "message.slice(redundantVisualizationNote.length).trim()" in note_filter
    assert "return message;" in note_filter
    assert "visualizationNote.hidden = !canRender || !displayNote;" in javascript


def test_process_summary_matches_ranked_set_retrieval_and_prompt_assembly() -> None:
    ragdemo_dir = Path(__file__).resolve().parents[1]
    javascript = (ragdemo_dir / "static" / "ragdemo.js").read_text(encoding="utf-8")
    service_source = (ragdemo_dir / "service.py").read_text(encoding="utf-8")
    generate_source = (
        ragdemo_dir.parent / "pipeline" / "generate.py"
    ).read_text(encoding="utf-8")

    assert "points = retrieval.points" in service_source
    assert "for rank, point in enumerate(points, start=1):" in service_source
    assert "build_prompt_messages(" in service_source
    assert "answer = invoke_prompt_messages(messages)" in service_source
    assert 'context = "\\n\\n".join(context_parts)' in generate_source
    assert "state.connections.set(point.id, connection);" in javascript
    assert "previousChunk" not in javascript
    assert "chunk-to-chunk" not in javascript


def test_visualization_frontend_builds_axes_origin_query_vector_and_retrieval_links() -> None:
    javascript = (
        Path(__file__).resolve().parents[1] / "static" / "ragdemo.js"
    ).read_text(encoding="utf-8")
    scene_render = javascript[
        javascript.index("function renderViewerScene"):
        javascript.index("function installCameraInteraction")
    ]
    retrieved_branch = javascript[
        javascript.index('      if (point.retrieved) {\n        const rank'):
        javascript.index("      const entry = {", javascript.index('      if (point.retrieved) {\n        const rank'))
    ]

    assert '["x", "y", "z"].forEach((dimension) =>' in javascript
    assert "embedding-axis-negative" in javascript
    assert "embedding-axis-positive" in javascript
    assert 'currentProjectionMethod === "umap" ? "UMAP" : "PCA"' in javascript
    assert 'originLabel.textContent = "O";' in javascript
    assert "embedding-grid-origin-line" in javascript
    assert "sceneGeometry.ticks.x.values.forEach" in javascript
    assert "sceneGeometry.ticks.y.values.forEach" in javascript
    assert "sceneGeometry.ticks.z.values.forEach" in javascript
    assert 'svgElement("line", "embedding-axis-tick")' in javascript
    assert 'svgElement("text", "embedding-axis-tick-label")' in javascript
    assert "formatTickValue(tick.value" in javascript
    assert "sceneGeometry.origin" in javascript
    assert "setProjectedLine(axis.visual.negativeLine, projectedOrigin" in scene_render
    assert "setProjectedLine(axis.visual.positiveLine, projectedOrigin" in scene_render
    assert 'svgElement("line", "query-vector")' in javascript
    assert 'queryVector.setAttribute("marker-end", "url(#query-vector-arrowhead)")' in javascript
    assert 'const showsLinearOrigin = currentProjectionMethod === "pca";' in javascript
    assert "if (showsLinearOrigin) originLayer.appendChild(originGroup);" in javascript
    assert "if (showsLinearOrigin) queryVectorLayer.appendChild(queryVector);" in javascript
    assert "if (state.showsLinearOrigin)" in scene_render
    assert "shortenProjectedLineEnd" in scene_render
    assert "setProjectedLine(state.query.visual.vector, projectedOrigin, projectedVectorEnd)" in scene_render
    assert "world: sceneGeometry.points[visualization.points.length]" in javascript
    assert 'queryLabel.textContent = "Q";' in javascript

    assert "visualization.points.forEach((point, index) =>" in javascript
    assert 'rank.textContent = `#${point.rank}`;' in retrieved_branch
    assert 'svgElement("line", "query-retrieval-connection")' in retrieved_branch
    assert "state.connections.set(point.id, connection);" in retrieved_branch
    assert "state.connections.forEach((connection, chunkId) =>" in scene_render
    assert "setProjectedLine(connection, projectedQuery, projectedChunk);" in scene_render
    assert 'connection.classList.toggle("is-active", chunkId === activeId);' in scene_render
    assert 'svgElement("line", "query-retrieval-connection")' not in javascript[
        javascript.index("visualization.points.forEach((point, index) =>"):
        javascript.index('      if (point.retrieved) {\n        const rank')
    ]
    assert "rank.textContent" not in javascript[
        javascript.index("visualization.points.forEach((point, index) =>"):
        javascript.index('      if (point.retrieved) {\n        const rank')
    ]


def test_visualization_frontend_uses_safe_linked_coordinate_tooltips() -> None:
    static_dir = Path(__file__).resolve().parents[1] / "static"
    javascript = (static_dir / "ragdemo.js").read_text(encoding="utf-8")
    stylesheet = (static_dir / "ragdemo.css").read_text(encoding="utf-8")

    assert "card.dataset.chunkId = result.id;" in javascript
    assert "group.dataset.chunkId = point.id;" in javascript
    assert "visualizationPointByChunkId.set(point.id, group);" in javascript
    assert "setHoveredChunk(point.id);" in javascript
    assert "setFocusedChunk(point.id)" in javascript
    assert "openChunkInspector(point.id, group)" in javascript
    assert 'event.key === "Enter" || event.key === " "' in javascript
    assert 'point.classList.toggle("is-linked", chunkId === activeId);' in javascript
    assert 'card.classList.toggle("is-linked", chunkId === activeId);' in javascript

    assert "function renderQueryHoverLabel(query, projected, state)" in javascript
    assert "function renderChunkHoverLabel(point, projected, state)" in javascript
    assert "appendHoverCoordinates(visualizationTooltip, query);" in javascript
    assert "appendHoverCoordinates(visualizationTooltip, point);" in javascript
    hover_renderer = javascript[
        javascript.index("function renderChunkHoverLabel"):
        javascript.index("function clearPointTooltip")
    ]
    assert "safePointPreview" not in hover_renderer
    assert "preview" not in hover_renderer
    assert "function renderPinnedInspector()" in javascript
    assert 'setLocalizedText(visualizationInspectorTitle, "query");' in javascript
    assert '"Loading chunk…"' in javascript
    assert '"This chunk could not be loaded."' in javascript
    assert 'point.retrieved ? "retrieved_rank" : "not_retrieved"' in javascript
    assert 'retrieved_rank: "Retrieved · Rank #{rank}"' in javascript
    assert '"Chunk content"' in javascript
    assert "details.content" in javascript
    assert "appendCoordinateReadout(visualizationInspectorContent" in javascript
    assert 'visualizationInspectorClose.addEventListener("click", () => clearPinnedSelection(true));' in javascript
    assert "visualizationInspector.style" not in javascript
    inspector_rule = stylesheet[
        stylesheet.index(".embedding-map-inspector {"):
        stylesheet.index(".embedding-inspector-header")
    ]
    assert "position: absolute;" in inspector_rule
    assert "right: 0.9rem;" in inspector_rule
    assert "bottom: 0.9rem;" in inspector_rule
    assert "z-index: 4;" in inspector_rule
    hidden_rule = stylesheet[
        stylesheet.index(".embedding-map-inspector[hidden]"):
        stylesheet.index(".embedding-map-stage.is-expanded .embedding-map-inspector")
    ]
    assert "display: none;" in hidden_rule
    assert '["X", "Y", "Z"].forEach((label) =>' in javascript
    assert "heading.textContent = point.retrieved" in javascript
    assert "setLocalizedText(\n        nativeTooltip," in javascript
    assert "point.vector" not in javascript
    assert "visualization.vector" not in javascript
    assert "innerHTML" not in javascript

    inspector_open = javascript[
        javascript.index("async function openChunkInspector"):
        javascript.index("function togglePinnedQuery")
    ]
    inspector_close = javascript[
        javascript.index("function clearPinnedSelection"):
        javascript.index("function updateLinkedInteraction")
    ]
    for inspector_action in (inspector_open, inspector_close):
        assert "resizeViewerToCanvas" not in inspector_action
        assert "setPresentationMode" not in inspector_action
        assert ".rotation =" not in inspector_action
        assert ".zoom =" not in inspector_action
        assert "fittedSceneScale" not in inspector_action


def test_visualization_frontend_uses_one_stable_scene_and_free_trackball_rotation() -> None:
    static_dir = Path(__file__).resolve().parents[1] / "static"
    javascript = (static_dir / "ragdemo.js").read_text(encoding="utf-8")
    stylesheet = (static_dir / "ragdemo.css").read_text(encoding="utf-8")
    projection = javascript[
        javascript.index("function projectWorldPoint"):
        javascript.index("function positionPointTooltip")
    ]
    scene_render = javascript[
        javascript.index("function renderViewerScene"):
        javascript.index("function installCameraInteraction")
    ]

    assert "function buildSceneGeometry(coordinates)" in javascript
    assert "numericCoordinates.map(toWorld)" in javascript
    assert "Number(point.x) / normalizationRadius" in javascript
    assert "Number(point.y) / normalizationRadius" in javascript
    assert "Number(point.z) / normalizationRadius" in javascript
    assert "const dataCenter" in javascript
    assert "sceneCenter: toWorld(dataCenter)" in javascript
    assert "const sceneBounds = Object.freeze" in javascript
    assert "computeFittedSceneScale(sceneBounds.width, sceneBounds.height)" in javascript
    assert "const edgePadding = Math.max(scenePadding, maximumMarkerExtent);" in javascript
    assert "const maximumMarkerExtent = 30;" in javascript
    assert "sceneWidth - 2 * edgePadding" in javascript
    assert "resizeViewerToCanvas" in javascript
    assert "visualizationCanvas.getBoundingClientRect()" in javascript
    assert "state.fittedSceneScale * state.zoom" in projection
    assert "rotateWorldPoint(centered, state.rotation)" in projection
    assert "perspective" not in projection
    assert "x: clamp(" not in projection
    assert "y: clamp(" not in projection
    assert "projectWorldPoint(state.origin.world, state)" in scene_render
    assert "projectWorldPoint(state.query.world, state)" in scene_render
    assert "projectWorldPoint(entry.world, state)" in scene_render
    assert "projectWorldPoint(axis.negativeWorld, state)" in scene_render
    assert "projected.depth - second.projected.depth" in scene_render
    assert "const depthScale" in scene_render
    assert "projected.depthRatio" in scene_render
    assert "state.gridLines.forEach" in scene_render
    assert "axis.ticks.forEach" in scene_render
    assert "state.pointLayer.appendChild(activePoint.entry.visual.group);" in scene_render
    assert "entry.world" not in javascript[
        javascript.index("const activePoint ="):
        javascript.index("state.query.visual.group.setAttribute")
    ]
    previous_fit = (600 - 2 * max(18, 34)) / 2
    current_fit = (600 - 2 * max(12, 30)) / 2
    assert current_fit > previous_fit

    assert "function trackballVector(event, svg)" in javascript
    assert "function quaternionBetween(first, second)" in javascript
    assert "state.rotation = quaternionMultiply(deltaRotation, state.rotation);" in javascript
    assert 'svg.addEventListener("pointerdown"' in javascript
    assert 'svg.addEventListener("pointermove"' in javascript
    assert 'svg.addEventListener("pointerup", async (event) =>' in javascript
    assert 'svg.addEventListener("pointercancel", endDrag);' in javascript
    assert 'svg.addEventListener("lostpointercapture", endDrag);' in javascript
    assert "pointerDragThreshold = 5" in javascript
    assert "event.clientX - state.pointerStartClientX" in javascript
    assert "event.clientY - state.pointerStartClientY" in javascript
    assert "await openChunkInspector(pressedChunkId, pressedChunkElement);" in javascript
    assert "svg.setPointerCapture(event.pointerId);" in javascript
    pointer_down = javascript[
        javascript.index('svg.addEventListener("pointerdown"'):
        javascript.index('svg.addEventListener("pointermove"')
    ]
    pointer_move = javascript[
        javascript.index('svg.addEventListener("pointermove"'):
        javascript.index("const endDrag =")
    ]
    assert "setPointerCapture" not in pointer_down
    assert "svg.setPointerCapture(event.pointerId);" in pointer_move
    assert "state.pointerId = null;" in javascript
    assert "state.lastTrackballVector = null;" in javascript
    assert "touch-action: none" in stylesheet
    assert "user-select: none" in stylesheet
    assert "cursor: grab" not in stylesheet
    assert "cursor: grabbing" not in stylesheet
    for external_library in ("three.js", "plotly", "d3.js", "babylon", "cdnjs"):
        assert external_library not in javascript.casefold()


def test_visualization_zoom_reset_and_semantic_visibility_are_preserved() -> None:
    javascript = (
        Path(__file__).resolve().parents[1] / "static" / "ragdemo.js"
    ).read_text(encoding="utf-8")
    pointer_reset = javascript[
        javascript.index("function resetPointerInteraction"):
        javascript.index("function resetViewer")
    ]
    viewer_reset = javascript[
        javascript.index("function resetViewer"):
        javascript.index("function renderVisualization")
    ]
    camera_reset = javascript[
        javascript.index("function resetViewer"):
        javascript.index("function resizeViewerToCanvas")
    ]

    assert "zoom: 1" in javascript
    assert "minimumZoom = 0.55" in javascript
    assert "maximumZoom = 2.6" in javascript
    assert "clamp(viewerState.zoom * factor, minimumZoom, maximumZoom)" in javascript
    assert 'svg.addEventListener("wheel", (event) =>' in javascript
    assert "{ passive: false }" in javascript
    assert "event.deltaY < 0 ? wheelZoomFactor : 1 / wheelZoomFactor" in javascript
    assert "event.preventDefault();" in javascript[
        javascript.index('svg.addEventListener("wheel"'):
        javascript.index('svg.addEventListener("pointerdown"')
    ]
    assert 'zoomOutButton.addEventListener("click", () => zoomViewer(1 / 1.12));' in javascript
    assert 'zoomInButton.addEventListener("click", () => zoomViewer(1.12));' in javascript
    assert "resetViewButton.addEventListener" in javascript
    assert 'zoomLevel.textContent = `${percentage}%`;' in javascript

    assert "const initialCamera = Object.freeze" in javascript
    assert "state.rotation = { ...state.initialCamera.rotation };" in viewer_reset
    assert "state.zoom = state.initialCamera.zoom;" in viewer_reset
    assert "hoveredChunkId = null;" in viewer_reset
    assert "focusedChunkId = null;" in viewer_reset
    assert "pinnedChunkId = null;" in viewer_reset
    assert "pinnedQuery = false;" in viewer_reset
    assert "state.hoveredScenePointId = null;" in viewer_reset
    assert "state.queryHovered = false;" in viewer_reset
    assert "state.queryFocused = false;" in viewer_reset
    assert "resetPointerInteraction(state);" in viewer_reset
    assert "state.svg.releasePointerCapture(capturedPointerId);" in pointer_reset
    assert 'state.svg.classList.remove("is-dragging");' in pointer_reset
    assert "clearPointTooltip();" in viewer_reset
    assert "renderPinnedInspector();" in viewer_reset
    assert "updateLinkedInteraction(false);" in viewer_reset
    assert "renderViewerScene();" in viewer_reset
    assert "setPresentationMode" not in camera_reset

    assert "visualizationSection.hidden = !isSemantic;" in javascript
    assert 'const isSemantic = searchMode === "semantic";' in javascript
    assert "visualizationPanel.hidden = !canRender;" in javascript


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
    assert set(body) == {"question", "answer", "search", "visualization", "prompt"}
    assert body["question"] == "Where is the demo intended to be shown?"
    assert body["answer"] == "The supporting source is demo_event.txt."
    assert body["search"]["mode"] == "semantic"
    assert body["search"]["label"] == "Semantic vector search"
    assert body["search"]["technical_label"] == "Dense / BGE-M3"
    assert body["search"]["score_label"] == "Semantic similarity"
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
        "matched_terms",
    }
    assert body["search"]["results"][0]["matched_terms"] == []
    assert body["visualization"]["status"] == "unavailable"
    assert body["visualization"]["points"] == []
    assert body["visualization"]["query"] is None
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
async def test_semantic_and_hybrid_use_isolated_native_configs_and_same_points(
    monkeypatch: Any,
) -> None:
    point = _point(
        "native-point",
        content="Berlin evidence",
        source="/private/berlin.pdf",
    )
    point.score = 0.73
    operator = FakeSearchOperator(documents=[point])
    reader = _client(([], None))
    reader.get_collection.return_value = _collection_schema(sparse=True)
    model = CapturingModel(answer="Berlin answer")
    monkeypatch.setattr(pipeline_generate, "llm", model)
    shared_config = {
        **KEYWORD_CONFIG,
        "search_mode": "dense_sparse",
        "fusion_mode": "DBSF",
        "prefetch_limit_dense": 25,
        "prefetch_limit_sparse": 17,
    }
    original_config = dict(shared_config)
    shared_user_config = {"collection_name": "hybrid-demo", "llm": "demo"}

    semantic = await service.execute_query(
        operator,
        "Question",
        shared_config,
        retrieval_mode="semantic",
        qdrant_reader=reader,
        collection_name="hybrid-demo",
        runtime_user_config=shared_user_config,
    )
    hybrid = await service.execute_query(
        operator,
        "Question",
        shared_config,
        retrieval_mode="hybrid",
        qdrant_reader=reader,
        collection_name="hybrid-demo",
        runtime_user_config=shared_user_config,
    )
    semantic_after_hybrid = await service.execute_query(
        operator,
        "Question",
        shared_config,
        retrieval_mode="semantic",
        qdrant_reader=reader,
        collection_name="hybrid-demo",
        runtime_user_config=shared_user_config,
    )

    assert shared_config == original_config
    assert shared_user_config == {"collection_name": "hybrid-demo", "llm": "demo"}
    assert [config[1]["search_mode"] for config in operator.config_calls] == [
        "dense",
        "dense_sparse",
        "dense",
    ]
    assert operator.config_calls[0][1] is not operator.config_calls[1][1]
    assert operator.config_calls[1][1] is not operator.config_calls[2][1]
    assert operator.config_calls[1][1]["fusion_mode"] == "DBSF"
    assert operator.config_calls[1][1]["prefetch_limit_dense"] == 25
    assert operator.config_calls[1][1]["prefetch_limit_sparse"] == 17
    assert operator.config_calls[1][1]["top_k"] == 5
    assert operator.config_calls[0][0]["collection_name"] == "hybrid-demo"
    assert operator.config_calls[1][0]["collection_name"] == "hybrid-demo"
    assert semantic.search.mode == "semantic"
    assert semantic.search.score_label == "Semantic similarity"
    assert semantic.visualization.status == "unavailable"
    assert hybrid.search.mode == "hybrid"
    assert hybrid.search.technical_label == "BGE-M3 dense + sparse · DBSF"
    assert hybrid.search.score_label == "Fusion score"
    assert hybrid.visualization.status == "unsupported"
    assert "shown only for Semantic vectors" in hybrid.visualization.note
    assert [result.id for result in semantic.search.results] == [
        result.id for result in hybrid.search.results
    ]
    assert semantic_after_hybrid.search.mode == "semantic"
    assert operator.config_calls[2][1]["search_mode"] == "dense"
    for response in (semantic, hybrid, semantic_after_hybrid):
        assert response.search.results[0].content == "Berlin evidence"
        assert "Berlin evidence" in response.prompt.messages[0].content
    assert len(operator.calls) == 3
    assert len(model.calls) == 3


@pytest.mark.anyio
async def test_semantic_uses_original_question_for_search_and_prompt(
    monkeypatch: Any,
) -> None:
    question = "Was sind die vier Ziele des Berliner Prozesses?"
    point = _point(
        "semantic",
        content="The Berlin Process has four declared objectives.",
        source="berlin.pdf",
    )
    point.score = 0.91
    operator = FakeSearchOperator(documents=[point])
    model = CapturingModel(answer="Der Berliner Prozess hat vier Ziele.")
    monkeypatch.setattr(pipeline_generate, "llm", model)
    response = await service.execute_query(
        operator,
        question,
        KEYWORD_CONFIG,
        retrieval_mode="semantic",
        qdrant_reader=_client(([], None)),
        collection_name="demo",
    )

    assert operator.calls[0]["question"] == question
    assert model.calls[0][1].content == question
    assert response.question == question
    assert response.search.results[0].rank == 1
    assert response.search.results[0].score == 0.91


@pytest.mark.anyio
async def test_keyword_uses_original_question_for_bm25_and_prompt(
    monkeypatch: Any,
) -> None:
    question = "What are the four objectives of the Berlin Process?"
    record = _point(
        "keyword",
        content=(
            "The Berlin Process has four objectives and this English source "
            "describes those objectives in detail."
        ),
        source="berlin.pdf",
        loader_id="local",
        document_id="berlin",
    )

    async def authorize(user: str, response: Any) -> list[Any]:
        assert user == DEMO_USER
        return list(response.points)

    monkeypatch.setattr(service, "filter_authorized", authorize)
    model = CapturingModel(answer="Vier Ziele werden beschrieben.")
    monkeypatch.setattr(pipeline_generate, "llm", model)

    response = await service.execute_query(
        FakeSearchOperator(),
        question,
        KEYWORD_CONFIG,
        retrieval_mode="keyword",
        qdrant_reader=_client(([record], None)),
        collection_name="demo",
    )

    assert response.search.results[0].content == record.payload["content"]
    assert response.search.results[0].rank == 1
    assert response.search.results[0].score > 0
    assert model.calls[0][1].content == question


@pytest.mark.anyio
async def test_hybrid_uses_original_question_in_native_path_and_generation(
    monkeypatch: Any,
) -> None:
    question = "Was sind die vier Ziele des Berliner Prozesses?"
    sample = _point(
        "sample",
        content="This English corpus explains the four objectives of the Berlin Process.",
        source="berlin.pdf",
    )
    result_point = _point(
        "hybrid-result",
        content="The Berlin Process has four declared objectives.",
        source="berlin.pdf",
    )
    result_point.score = 0.73
    reader = _client(([sample], None))
    reader.get_collection.return_value = _collection_schema(sparse=True)
    operator = FakeSearchOperator(documents=[result_point])

    model = CapturingModel(answer="Vier Ziele werden beschrieben.")
    monkeypatch.setattr(pipeline_generate, "llm", model)
    config = {**KEYWORD_CONFIG, "fusion_mode": "DBSF"}

    response = await service.execute_query(
        operator,
        question,
        config,
        retrieval_mode="hybrid",
        qdrant_reader=reader,
        collection_name="demo",
        runtime_user_config={"collection_name": "demo"},
    )

    assert operator.calls == [{"question": question, "user": DEMO_USER}]
    assert operator.config_calls[0][1]["search_mode"] == "dense_sparse"
    assert operator.config_calls[0][1]["top_k"] == KEYWORD_CONFIG["top_k"]
    assert response.search.score_label == "Fusion score"
    assert response.search.results[0].score == 0.73
    assert model.calls[0][1].content == question


@pytest.mark.anyio
async def test_hybrid_is_rejected_before_retrieval_for_dense_only_collection(
    monkeypatch: Any,
) -> None:
    operator = FakeSearchOperator(documents=[])
    reader = MagicMock()
    reader.collection_exists.return_value = True
    reader.get_collection.return_value = _collection_schema(sparse=False)
    monkeypatch.setattr(routes, "demo_search_operator", operator)
    monkeypatch.setattr(
        routes,
        "opt_config",
        {**KEYWORD_CONFIG, "fusion_mode": "DBSF"},
    )
    monkeypatch.setattr(routes, "user_config", {"collection_name": "dense-demo"})
    monkeypatch.setattr(routes, "demo_qdrant_reader", reader)

    async with AsyncClient(
        transport=ASGITransport(app=_app()),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/ragdemo/api/query",
            json={"question": "Question", "retrieval_mode": "hybrid"},
        )

    assert response.status_code == 409
    assert response.json()["message"] == (
        "Hybrid search is unavailable for the current index."
    )
    assert operator.calls == []


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
async def test_unknown_retrieval_mode_is_rejected_before_search(monkeypatch: Any) -> None:
    search_operator = FakeSearchOperator(documents=[])
    monkeypatch.setattr(routes, "demo_search_operator", search_operator)

    async with AsyncClient(
        transport=ASGITransport(app=_app()),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/ragdemo/api/query",
            json={"question": "Where is the demo?", "retrieval_mode": "sparse"},
        )

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


def test_configured_prompt_uses_real_newlines_and_public_view_matches_invocation(
    monkeypatch: Any,
) -> None:
    config_path = Path(__file__).resolve().parents[2] / "pipeline" / "opt_config.json"
    configured = json.loads(config_path.read_text(encoding="utf-8"))
    configured_prompt = configured["prompt"]
    assert "\n" in configured_prompt
    assert "\\n" not in configured_prompt

    document = ScoredPoint(
        id="qdrant-newline-test",
        version=0,
        score=0.7,
        payload={"source": "/private/source.txt", "content": "Line one.\nLine two."},
    )
    messages = pipeline_generate.build_prompt_messages(
        "What does the source say?",
        [document],
        configured,
        source_transform=service._safe_prompt_source,
    )
    public_prompt = service._public_prompt(messages)
    model = CapturingModel(answer="Generated answer")
    monkeypatch.setattr(pipeline_generate, "llm", model)

    assert pipeline_generate.invoke_prompt_messages(messages) == "Generated answer"
    assert len(model.calls) == 1
    assert public_prompt.messages[0].content == model.calls[0][0].content
    assert public_prompt.messages[1].content == model.calls[0][1].content
    assert "\n" in public_prompt.messages[0].content
    assert "\\n" not in public_prompt.messages[0].content


def test_bm25_is_deterministic_and_ranks_stronger_lexical_matches() -> None:
    strong = _point(
        "strong-private-id",
        content="Berlin demo Berlin demo event",
        source="/private/strong.txt",
        loader_id="directory",
        document_id="strong-document",
    )
    weak = _point(
        "weak-private-id",
        content="Berlin hosts an event with unrelated details",
        source="/private/weak.txt",
        loader_id="directory",
        document_id="weak-document",
    )
    zero = _point(
        "zero-private-id",
        content="Completely unrelated material",
        source="/private/zero.txt",
        loader_id="directory",
        document_id="zero-document",
    )

    ranked, matched = service._bm25_candidates(
        "BERLIN demo",
        [weak, zero, strong],
    )
    repeated, repeated_matched = service._bm25_candidates(
        "BERLIN demo",
        [weak, zero, strong],
    )

    assert [point.id for point in ranked] == ["strong-private-id", "weak-private-id"]
    assert ranked[0].score > ranked[1].score > 0
    assert [point.score for point in ranked] == [point.score for point in repeated]
    assert matched == repeated_matched
    assert matched[service._point_display_id(ranked[0])] == ("berlin", "demo")
    assert matched[service._point_display_id(ranked[1])] == ("berlin",)
    assert all(point.id != "zero-private-id" for point in ranked)

    many_terms = [f"term{index}" for index in range(20)]
    bounded_record = _point(
        "bounded-private-id",
        content="the of for and " + " ".join(many_terms),
        source="/private/bounded.txt",
        loader_id="directory",
        document_id="bounded-document",
    )
    bounded_points, bounded_matches = service._bm25_candidates(
        "the of for and " + " ".join(many_terms),
        [bounded_record],
    )
    terms = bounded_matches[service._point_display_id(bounded_points[0])]
    assert len(terms) == service.MAX_MATCHED_TERMS == 12
    assert terms == tuple(many_terms[: service.MAX_MATCHED_TERMS])
    assert all(len(term) <= service.MAX_MATCHED_TERM_LENGTH for term in terms)


def test_weka_stopword_resource_is_normalized_and_loads_deterministically() -> None:
    lines = service._STOPWORDS_PATH.read_text(encoding="utf-8").splitlines()
    first = service._load_keyword_presentation_stopwords()
    second = service._load_keyword_presentation_stopwords()

    assert lines
    assert all(
        line
        and line == line.strip()
        and line == line.casefold()
        and service._WORD_TOKEN.fullmatch(line)
        for line in lines
    )
    assert first == frozenset(lines)
    assert first is second
    assert {"the", "of", "for", "and", "ll", "ve"} <= first


def test_keyword_matched_terms_filter_exact_normalized_weka_tokens() -> None:
    question = "THE theater authors improve what we'll recommend and they've documented"
    record = _point(
        "presentation-private-id",
        content=question,
        source="/private/presentation.txt",
        loader_id="directory",
        document_id="presentation-document",
    )

    ranked, matched = service._bm25_candidates(question, [record])

    assert service._tokenize_keyword_text(question) == [
        "the",
        "theater",
        "authors",
        "improve",
        "what",
        "we",
        "ll",
        "recommend",
        "and",
        "they",
        "ve",
        "documented",
    ]
    assert "theater" not in service._load_keyword_presentation_stopwords()
    assert matched[service._point_display_id(ranked[0])] == (
        "theater",
        "authors",
        "improve",
        "recommend",
        "documented",
    )


def test_stopword_presentation_filter_does_not_change_bm25_scores_or_ranking(
    monkeypatch: Any,
) -> None:
    records = [
        _point(
            "strong-stopword-id",
            content="the Berlin process and Berlin authors recommend to improve",
            source="/private/strong.txt",
        ),
        _point(
            "weak-stopword-id",
            content="the authors describe the Berlin process",
            source="/private/weak.txt",
        ),
    ]
    question = "What do the authors recommend to improve the Berlin Process?"
    expected_query_tokens = service._tokenize_keyword_text(question)
    stopwords = service._load_keyword_presentation_stopwords()

    monkeypatch.setattr(
        service,
        "_load_keyword_presentation_stopwords",
        lambda: frozenset(),
    )
    unfiltered_points, unfiltered_matches = service._bm25_candidates(question, records)
    monkeypatch.setattr(
        service,
        "_load_keyword_presentation_stopwords",
        lambda: stopwords,
    )
    filtered_points, filtered_matches = service._bm25_candidates(question, records)

    assert service._tokenize_keyword_text(question) == expected_query_tokens
    assert [point.id for point in filtered_points] == [
        point.id for point in unfiltered_points
    ]
    assert [point.score for point in filtered_points] == pytest.approx(
        [point.score for point in unfiltered_points]
    )
    strong_public_id = service._point_display_id(filtered_points[0])
    assert {"the", "to"} <= set(unfiltered_matches[strong_public_id])
    assert {"the", "to"}.isdisjoint(filtered_matches[strong_public_id])
    assert filtered_matches[strong_public_id] == (
        "authors",
        "recommend",
        "improve",
        "berlin",
        "process",
    )


def test_all_stopword_query_can_still_produce_bm25_results() -> None:
    record = _point(
        "all-stopwords-private-id",
        content="the of for and the",
        source="/private/all-stopwords.txt",
    )

    ranked, matched = service._bm25_candidates("the of for and", [record])

    assert len(ranked) == 1
    assert ranked[0].score > 0
    assert matched[service._point_display_id(ranked[0])] == ()


def test_keyword_scan_is_paginated_bounded_and_requests_no_vectors() -> None:
    client = _client(
        ([_point("1", content="first", source="/private/first.txt")], "next"),
        (
            [
                _point("2", content="second", source="/private/second.txt"),
                _point("3", content="third", source="/private/third.txt"),
            ],
            "more",
        ),
    )

    records, truncated = service._scan_keyword_chunks(
        client,
        "learn2rag",
        page_size=2,
        max_chunks=2,
    )

    assert [point.id for point in records] == ["1", "2"]
    assert truncated is True
    assert client.scroll.call_count == 2
    assert [call.kwargs["limit"] for call in client.scroll.call_args_list] == [2, 1]
    assert all(call.kwargs["with_vectors"] is False for call in client.scroll.call_args_list)
    assert all(
        call.kwargs["with_payload"] == service.KEYWORD_PAYLOAD_FIELDS
        for call in client.scroll.call_args_list
    )


@pytest.mark.anyio
async def test_keyword_mode_uses_authorized_bm25_points_for_search_prompt_and_model(
    monkeypatch: Any,
) -> None:
    denied = _point(
        "raw-denied-id",
        content="Berlin demo Berlin demo Berlin demo",
        source="/home/private/denied.txt",
        loader_id="private-loader",
        document_id="denied-document",
        content_hash="denied-content-hash",
        chunk_hash="denied-chunk-hash",
    )
    allowed_first = _point(
        "raw-allowed-id-1",
        content="The Berlin demo welcomes visitors.",
        source="/home/private/allowed-one.txt",
        loader_id="private-loader",
        document_id="allowed-document-1",
        content_hash="allowed-content-hash-1",
        chunk_hash="allowed-chunk-hash-1",
    )
    allowed_second = _point(
        "raw-allowed-id-2",
        content="This demo explains retrieval.",
        source="/home/private/allowed-two.txt",
        loader_id="private-loader",
        document_id="allowed-document-2",
        content_hash="allowed-content-hash-2",
        chunk_hash="allowed-chunk-hash-2",
    )
    zero = _point(
        "raw-zero-id",
        content="Unrelated content only.",
        source="/home/private/zero.txt",
        loader_id="private-loader",
        document_id="zero-document",
    )
    keyword_client = _client(([denied, allowed_second, zero, allowed_first], None))
    search_operator = FakeSearchOperator(documents=[])
    model = CapturingModel(answer="See /home/private/allowed-one.txt.")
    authorization_calls: list[tuple[str, list[ScoredPoint]]] = []

    async def authorize(user: str, response: Any) -> list[ScoredPoint]:
        points = list(response.points)
        authorization_calls.append((user, points))
        return [
            point
            for point in points
            if point.payload and point.payload.get("document_id") != "denied-document"
        ]

    monkeypatch.setattr(routes, "demo_search_operator", search_operator)
    monkeypatch.setattr(routes, "demo_qdrant_reader", keyword_client)
    monkeypatch.setattr(routes, "user_config", {"collection_name": "learn2rag"})
    monkeypatch.setattr(routes, "opt_config", {**KEYWORD_CONFIG, "top_k": 2})
    monkeypatch.setattr(service, "filter_authorized", authorize)
    monkeypatch.setattr(pipeline_generate, "llm", model)

    async with AsyncClient(
        transport=ASGITransport(app=_app()),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/ragdemo/api/query",
            json={"question": "Berlin demo", "retrieval_mode": "keyword"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["search"]["mode"] == "keyword"
    assert body["search"]["label"] == "Keyword search"
    assert body["search"]["technical_label"] == "BM25 lexical ranking"
    assert body["search"]["score_label"] == "BM25 score"
    assert [result["content"] for result in body["search"]["results"]] == [
        allowed_first.payload["content"],
        allowed_second.payload["content"],
    ]
    assert body["search"]["results"][0]["matched_terms"] == ["berlin", "demo"]
    assert body["search"]["results"][1]["matched_terms"] == ["demo"]
    assert body["search"]["results"][0]["id"] == service._chunk_display_id(
        allowed_first,
        allowed_first.payload,
    )
    assert body["search"]["results"][0]["source"] == "allowed-one.txt"
    assert body["answer"] == "See allowed-one.txt."
    assert body["visualization"]["status"] == "unsupported"
    assert body["visualization"]["points"] == []
    assert search_operator.calls == []
    assert len(model.calls) == 1
    assert authorization_calls[0][0] == DEMO_USER
    assert {point.id for point in authorization_calls[0][1]} == {
        "raw-denied-id",
        "raw-allowed-id-1",
        "raw-allowed-id-2",
    }
    assert keyword_client.scroll.call_count == 1
    assert keyword_client.scroll.call_args.kwargs["with_vectors"] is False
    assert keyword_client.query_points.call_count == 0

    system_content = model.calls[0][0].content
    assert isinstance(system_content, str)
    assert allowed_first.payload["content"] in system_content
    assert allowed_second.payload["content"] in system_content
    assert denied.payload["content"] not in system_content
    assert system_content.index(allowed_first.payload["content"]) < system_content.index(
        allowed_second.payload["content"]
    )
    assert body["prompt"]["messages"] == _serialized_captured_messages(model)
    serialized = response.text
    for internal_value in (
        "/home/private/allowed-one.txt",
        "raw-allowed-id-1",
        "private-loader",
        "allowed-document-1",
        "allowed-content-hash-1",
        "allowed-chunk-hash-1",
    ):
        assert internal_value not in serialized


@pytest.mark.anyio
async def test_keyword_no_match_does_not_fall_back_and_still_invokes_model_once(
    monkeypatch: Any,
) -> None:
    no_match = _point(
        "raw-no-match",
        content="Material about a different subject.",
        source="/private/no-match.txt",
        loader_id="directory",
        document_id="no-match-document",
    )
    keyword_client = _client(([no_match], None))
    search_operator = FakeSearchOperator(documents=[])
    raw_model_answer = "RAW MODEL SPECULATION MUST NOT BE PUBLIC"
    model = CapturingModel(answer=raw_model_answer)
    authorization = MagicMock()
    monkeypatch.setattr(routes, "demo_search_operator", search_operator)
    monkeypatch.setattr(routes, "demo_qdrant_reader", keyword_client)
    monkeypatch.setattr(routes, "user_config", {"collection_name": "learn2rag"})
    monkeypatch.setattr(routes, "opt_config", KEYWORD_CONFIG)
    monkeypatch.setattr(service, "filter_authorized", authorization)
    monkeypatch.setattr(pipeline_generate, "llm", model)

    async with AsyncClient(
        transport=ASGITransport(app=_app()),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/ragdemo/api/query",
            json={"question": "quantum mechanics", "retrieval_mode": "keyword"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["search"]["mode"] == "keyword"
    assert body["search"]["results"] == []
    assert body["answer"] == service.NO_KEYWORD_EVIDENCE_ANSWER
    assert raw_model_answer not in response.text
    assert body["visualization"]["status"] == "unsupported"
    assert search_operator.calls == []
    assert authorization.call_count == 0
    assert keyword_client.scroll.call_count == 1
    assert len(model.calls) == 1
    assert no_match.payload["content"] not in str(model.calls[0][0].content)
    expected_zero_context_prompt = str(KEYWORD_CONFIG["prompt"]).format(context="")
    assert model.calls[0][0].content == expected_zero_context_prompt
    assert body["prompt"]["messages"] == _serialized_captured_messages(model)


@pytest.mark.anyio
async def test_semantic_zero_evidence_uses_guard_after_one_retrieval_and_model_call(
    monkeypatch: Any,
) -> None:
    raw_model_answer = "UNGROUNDED SEMANTIC MODEL OUTPUT"
    search_operator, model = _install_query_boundaries(
        monkeypatch,
        [],
        raw_model_answer,
    )

    async with AsyncClient(
        transport=ASGITransport(app=_app()),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/ragdemo/api/query",
            json={"question": "What is absent?", "retrieval_mode": "semantic"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["search"]["mode"] == "semantic"
    assert body["search"]["results"] == []
    assert body["answer"] == service.NO_EVIDENCE_ANSWER
    assert raw_model_answer not in response.text
    assert body["prompt"]["messages"] == _serialized_captured_messages(model)
    assert len(search_operator.calls) == 1
    assert len(model.calls) == 1


def test_dense_visualization_projects_safe_points_and_matches_search_ids(
    monkeypatch: Any,
) -> None:
    raw_source = "/private/customer/demo_event.txt"
    retrieved = _vector_point(
        "raw-qdrant-id-1",
        [1.0, 0.0, 0.2],
        content="Retrieved content",
        source=raw_source,
        loader_id="private-loader",
        document_id="private-document-id",
        content_hash="private-content-hash",
        chunk_hash="private-chunk-hash",
    )
    other = _vector_point(
        "raw-qdrant-id-2",
        [0.0, 1.0, 0.1],
        content="Other content",
        source="/private/customer/other.txt",
        loader_id="private-loader",
        document_id="other-private-document-id",
        content_hash="other-private-content-hash",
        chunk_hash="other-private-chunk-hash",
    )
    retrieved.score = 0.91
    search_result = service._public_search_result(retrieved, rank=1)
    embedding_calls: list[tuple[list[str], str, str]] = []

    def fake_embeddings(
        inputs: list[str], model_name: str, embedding_mode: str
    ) -> dict[str, np.ndarray[Any, Any]]:
        embedding_calls.append((inputs, model_name, embedding_mode))
        return {"dense_vecs": np.asarray([[0.8, 0.1, 0.2]])}

    monkeypatch.setattr(service, "create_embeddings", fake_embeddings)
    client = _client(([retrieved, other], None))

    result = build_query_visualization(
        client,
        "learn2rag",
        "Where is the demo?",
        DEMO_CONFIG,
        [search_result],
    )

    assert result.status == "ready"
    assert result.projection_method == "pca"
    assert result.query is not None
    assert len(result.points) == 2
    assert all(np.isfinite([point.x, point.y, point.z]).all() for point in result.points)
    assert np.isfinite([result.query.x, result.query.y, result.query.z]).all()
    retrieved_point = next(point for point in result.points if point.retrieved)
    other_point = next(point for point in result.points if not point.retrieved)
    assert retrieved_point.id == search_result.id
    assert retrieved_point.rank == search_result.rank == 1
    assert retrieved_point.source == "demo_event.txt"
    assert other_point.rank is None
    expected_retrieved_similarity = float(
        np.dot([0.8, 0.1, 0.2], [1.0, 0.0, 0.2])
        / (np.linalg.norm([0.8, 0.1, 0.2]) * np.linalg.norm([1.0, 0.0, 0.2]))
    )
    expected_other_similarity = float(
        np.dot([0.8, 0.1, 0.2], [0.0, 1.0, 0.1])
        / (np.linalg.norm([0.8, 0.1, 0.2]) * np.linalg.norm([0.0, 1.0, 0.1]))
    )
    assert retrieved_point.similarity == pytest.approx(expected_retrieved_similarity)
    assert other_point.similarity == pytest.approx(expected_other_similarity)
    assert all(math.isfinite(point.similarity) for point in result.points)
    assert [projection.method for projection in result.projections] == ["pca", "tsne", "umap"]
    expected_ids = [point.id for point in result.points]
    for projection in result.projections:
        assert [point.id for point in projection.points] == expected_ids
        assert all(
            np.isfinite([point.x, point.y, point.z]).all()
            for point in projection.points
        )
        assert np.isfinite(
            [projection.query.x, projection.query.y, projection.query.z]
        ).all()
    assert result.projections[2].fallback == "pca"
    assert embedding_calls == [
        (["Where is the demo?"], "BAAI/bge-m3", "dense")
    ]
    scroll_call = client.scroll.call_args
    assert scroll_call.kwargs["with_vectors"] == ["dense"]
    assert scroll_call.kwargs["with_payload"] == service.VISUALIZATION_PAYLOAD_FIELDS
    assert "content" not in scroll_call.kwargs["with_payload"]

    serialized = result.model_dump_json()
    for private_value in (
        raw_source,
        "raw-qdrant-id-1",
        "private-loader",
        "private-document-id",
        "private-content-hash",
        "private-chunk-hash",
    ):
        assert private_value not in serialized
    assert all(
        set(point.model_dump())
        == {"id", "source", "x", "y", "z", "retrieved", "rank", "similarity", "preview"}
        for point in result.points
    )
    assert '"vector":' not in serialized
    assert "0.91" not in serialized


def test_visualization_scan_is_bounded_and_reports_partial(monkeypatch: Any) -> None:
    points = [
        _vector_point(
            str(index),
            [float(index), 1.0],
            source=f"/private/document-{index}.txt",
        )
        for index in range(3)
    ]
    client = _client((points, "more-points"))
    monkeypatch.setattr(
        service,
        "create_embeddings",
        lambda *args, **kwargs: {"dense_vecs": np.asarray([[0.5, 0.5]])},
    )

    result = build_query_visualization(
        client,
        "learn2rag",
        "question",
        DEMO_CONFIG,
        [],
        page_size=10,
        max_chunks=2,
    )

    assert result.status == "partial"
    assert result.truncated is True
    assert len(result.points) == 2
    assert "partial index snapshot" in result.note
    assert client.scroll.call_args.kwargs["limit"] == 2


def test_semantic_similarity_uses_original_vectors_not_pca_coordinates(
    monkeypatch: Any,
) -> None:
    first = _vector_point("first", [1.0, 0.0], source="/private/first.pdf")
    second = _vector_point("second", [0.0, 1.0], source="/private/second.pdf")
    monkeypatch.setattr(
        service,
        "create_embeddings",
        lambda *args, **kwargs: {"dense_vecs": np.asarray([[1.0, 0.0]])},
    )
    monkeypatch.setattr(
        service,
        "_pca_3d",
        lambda matrix: np.asarray(
            [
                [9.0, 9.0, 9.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
    )

    result = build_query_visualization(
        _client(([first, second], None)),
        "learn2rag",
        "question",
        DEMO_CONFIG,
        [],
    )

    assert [point.similarity for point in result.points] == [1.0, 0.0]
    assert result.points[0].x == 9.0
    assert result.points[1].x == 0.0
    assert all(point.rank is None for point in result.points)


def test_malformed_visualization_vectors_are_ignored_without_leaking(
    monkeypatch: Any,
) -> None:
    missing = _point("missing-vector", source="/secret/missing.txt")
    non_finite = _vector_point(
        "non-finite-vector",
        [float("nan"), 1.0],
        source="/secret/non-finite.txt",
        content_hash="secret-hash",
    )
    usable = _vector_point("usable-vector", [0.2, 0.7], source="/secret/usable.txt")
    monkeypatch.setattr(
        service,
        "create_embeddings",
        lambda *args, **kwargs: {"dense_vecs": np.asarray([[0.4, 0.6]])},
    )

    result = build_query_visualization(
        _client(([missing, non_finite, usable], None)),
        "learn2rag",
        "question",
        DEMO_CONFIG,
        [],
    )

    assert result.status == "ready"
    assert len(result.points) == 1
    serialized = result.model_dump_json()
    assert "missing-vector" not in serialized
    assert "non-finite-vector" not in serialized
    assert "secret-hash" not in serialized
    assert "nan" not in serialized.casefold()


def test_visualization_with_no_usable_vectors_is_safely_unavailable() -> None:
    malformed = _vector_point(
        "private-point-id",
        [float("inf"), 0.0],
        source="/secret/path.txt",
    )

    result = build_query_visualization(
        _client(([malformed], None)),
        "learn2rag",
        "question",
        DEMO_CONFIG,
        [],
    )

    assert result.status == "unavailable"
    assert result.points == []
    assert result.query is None
    assert "private-point-id" not in result.model_dump_json()
    assert "/secret/path.txt" not in result.model_dump_json()


def test_non_dense_visualization_is_unsupported_without_scanning_or_embedding(
    monkeypatch: Any,
) -> None:
    client = MagicMock()
    embeddings = MagicMock()
    monkeypatch.setattr(service, "create_embeddings", embeddings)
    sparse_config = {**DEMO_CONFIG, "search_mode": "sparse"}

    result = build_query_visualization(
        client,
        "learn2rag",
        "question",
        sparse_config,
        [],
    )

    assert result.status == "unsupported"
    assert result.query is None
    client.collection_exists.assert_not_called()
    client.scroll.assert_not_called()
    embeddings.assert_not_called()


def test_pca_projection_handles_three_components_and_degenerate_inputs() -> None:
    ordinary = np.asarray([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    lower_dimensional = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    degenerate = np.asarray([[1.0], [1.0]])

    ordinary_projection = service._pca_3d(ordinary)
    repeated_projection = service._pca_3d(ordinary)
    lower_projection = service._pca_3d(lower_dimensional)
    degenerate_projection = service._pca_3d(degenerate)

    assert ordinary_projection.shape == (4, 3)
    assert np.isfinite(ordinary_projection).all()
    assert not np.allclose(ordinary_projection[:, 2], 0.0)
    assert np.allclose(ordinary_projection, repeated_projection)
    assert lower_projection.shape == (3, 3)
    assert np.isfinite(lower_projection).all()
    assert np.allclose(lower_projection[:, 2], 0.0)
    assert degenerate_projection.shape == (2, 3)
    assert np.isfinite(degenerate_projection).all()
    assert np.allclose(degenerate_projection, 0.0)


def test_projection_dispatch_accepts_pca_tsne_and_umap_only() -> None:
    matrix = np.asarray([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    assert np.allclose(service._project_embeddings_3d(matrix, "pca"), service._pca_3d(matrix))
    assert service._project_embeddings_3d(matrix, "tsne").shape == (4, 3)
    assert service._project_embeddings_3d(matrix, "umap").shape == (4, 3)
    with pytest.raises(ValueError, match="Unsupported visualization projection"):
        service._project_embeddings_3d(matrix, "unknown")  # type: ignore[arg-type]


def test_tsne_projection_is_deterministic_finite_and_safe_for_small_samples() -> None:
    ordinary = np.asarray([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    two_samples = ordinary[:2]

    first = service._tsne_3d(ordinary)
    repeated = service._tsne_3d(ordinary)
    small = service._tsne_3d(two_samples)

    assert first.shape == (4, 3)
    assert np.isfinite(first).all()
    assert np.allclose(first, repeated)
    assert small.shape == (2, 3)
    assert np.isfinite(small).all()
    assert service._tsne_perplexity(2) == 1.0
    assert service._tsne_perplexity(1000) == service.TSNE_DEFAULT_PERPLEXITY
    assert service._tsne_perplexity(2) < 2


def test_umap_projection_is_deterministic_finite_and_safe_for_small_samples() -> None:
    ordinary = np.asarray([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0, 0.0],
    ])
    two_samples = ordinary[:2]

    first = service._umap_3d(ordinary)
    repeated = service._umap_3d(ordinary)
    small = service._umap_3d(two_samples)

    assert first.shape == (5, 3)
    assert np.isfinite(first).all()
    assert np.allclose(first, repeated)
    assert np.allclose(small, service._pca_3d(two_samples))
    assert service._umap_n_neighbors(2) == 2
    assert service._umap_n_neighbors(4) == 3
    assert service._umap_n_neighbors(1_000) == service.UMAP_DEFAULT_NEIGHBORS
    assert service.UMAP_RANDOM_STATE == 42
    assert service.UMAP_METRIC == "cosine"


def test_umap_projection_preserves_stable_explicit_parameters(
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}

    class CapturingUmap:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def fit_transform(self, matrix: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            return np.zeros((matrix.shape[0], 3), dtype=float)

    monkeypatch.setattr(service, "UMAP", CapturingUmap)
    matrix = np.eye(5, dtype=float)

    projected = service._umap_3d(matrix)

    assert projected.shape == (5, 3)
    assert captured == {
        "n_components": 3,
        "n_neighbors": 4,
        "metric": "cosine",
        "random_state": 42,
        "init": "random",
        "n_jobs": 1,
    }


@pytest.mark.anyio
async def test_visualization_failure_preserves_rag_response_and_call_counts(
    monkeypatch: Any,
) -> None:
    document = _point(
        "retrieved-private-id",
        content="A valid retrieved chunk.",
        source="/private/valid.txt",
    )
    document.score = 0.8
    search_operator = FakeSearchOperator(documents=[document])
    model = CapturingModel(answer="A valid answer.")
    failing_client = MagicMock()
    failing_client.collection_exists.return_value = True
    failing_client.scroll.side_effect = RuntimeError("secret vector scan failure")
    monkeypatch.setattr(routes, "demo_search_operator", search_operator)
    monkeypatch.setattr(routes, "opt_config", DEMO_CONFIG)
    monkeypatch.setattr(routes, "demo_qdrant_reader", failing_client)
    monkeypatch.setattr(pipeline_generate, "llm", model)

    async with AsyncClient(
        transport=ASGITransport(app=_app()),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/ragdemo/api/query",
            json={"question": "What is valid?"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "A valid answer."
    assert len(body["search"]["results"]) == 1
    assert len(body["prompt"]["messages"]) == 2
    assert body["visualization"]["status"] == "unavailable"
    assert "secret vector scan failure" not in response.text
    assert len(search_operator.calls) == 1
    assert len(model.calls) == 1
    assert failing_client.scroll.call_count == 1
    assert not hasattr(failing_client, "query_points") or failing_client.query_points.call_count == 0


@pytest.mark.anyio
async def test_successful_visualization_keeps_one_retrieval_and_one_model_call(
    monkeypatch: Any,
) -> None:
    document = _vector_point(
        "retrieved-id",
        [0.9, 0.1],
        content="A retrieved chunk.",
        source="/private/retrieved.txt",
    )
    document.score = 0.9
    search_operator = FakeSearchOperator(documents=[document])
    model = CapturingModel(answer="Answer")
    vector_client = _client(([document], None))
    monkeypatch.setattr(routes, "demo_search_operator", search_operator)
    monkeypatch.setattr(routes, "opt_config", DEMO_CONFIG)
    monkeypatch.setattr(routes, "demo_qdrant_reader", vector_client)
    monkeypatch.setattr(pipeline_generate, "llm", model)
    monkeypatch.setattr(
        service,
        "create_embeddings",
        lambda *args, **kwargs: {"dense_vecs": np.asarray([[0.8, 0.2]])},
    )

    async with AsyncClient(
        transport=ASGITransport(app=_app()),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/ragdemo/api/query",
            json={"question": "Question"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["visualization"]["status"] == "ready"
    assert body["search"]["mode"] == "semantic"
    assert body["search"]["results"][0]["matched_terms"] == []
    assert body["visualization"]["points"][0]["id"] == body["search"]["results"][0]["id"]
    assert body["visualization"]["points"][0]["rank"] == 1
    assert set(body["visualization"]["points"][0]) == {
        "id", "source", "x", "y", "z", "retrieved", "rank", "similarity", "preview",
    }
    assert set(body["visualization"]["query"]) == {"x", "y", "z"}
    assert len(search_operator.calls) == 1
    assert len(model.calls) == 1
    assert vector_client.scroll.call_count == 1
    assert vector_client.scroll.call_args.kwargs["with_vectors"] == ["dense"]
    assert not hasattr(vector_client, "query_points") or vector_client.query_points.call_count == 0


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


# Phase 8: example questions and read-only document/chunk inspection.
def test_packaged_examples_and_html_separation() -> None:
    config = service.load_example_questions()
    assert config.display_count == 3
    assert [question_set.id for question_set in config.question_sets] == [
        "berlin_process",
        "berlin_tourism",
        "climate_neutral_berlin",
    ]
    questions = [
        question
        for question_set in config.question_sets
        for question in question_set.questions
    ]
    assert [len(question_set.questions) for question_set in config.question_sets] == [5, 6, 6]
    assert len(questions) == 17
    template = (Path(__file__).resolve().parents[1] / "templates/index.html").read_text()
    assert all(question not in template for question in questions)
    assert "data-question=" not in template
    assert 'id="refresh-examples"' in template
    assert 'type="button" disabled data-i18n="suggest_examples">Suggest more examples' in template
    assert "Refresh examples" not in template


@pytest.mark.parametrize("questions", [
    [], None, "Question?", {}, [None], [123], [True], [""], [" \n\t "],
    ["x" * 501], [f"Question {i}" for i in range(51)],
])
def test_example_config_rejects_invalid_questions(questions: Any) -> None:
    with pytest.raises(ValidationError):
        ExampleQuestions.model_validate({
            "display_count": 1,
            "question_sets": [{"id": "topic", "questions": questions}],
        })


@pytest.mark.parametrize("display_count", [0, -1, 3, True, 1.5, "1", None])
def test_example_config_rejects_invalid_display_count(display_count: Any) -> None:
    with pytest.raises(ValidationError):
        ExampleQuestions.model_validate({
            "display_count": display_count,
            "question_sets": [{"id": "topic", "questions": ["A", "B"]}],
        })


@pytest.mark.parametrize("question_sets", [
    [], None, "topic", {},
    [{"id": f"topic_{index}", "questions": [str(index)]} for index in range(21)],
    [{"id": "unsafe-id", "questions": ["Question?"]}],
    [{"id": "x" * 41, "questions": ["Question?"]}],
])
def test_example_config_rejects_invalid_question_sets(question_sets: Any) -> None:
    with pytest.raises(ValidationError):
        ExampleQuestions.model_validate({"display_count": 1, "question_sets": question_sets})


def test_example_config_normalizes_ids_and_trims_questions() -> None:
    config = ExampleQuestions.model_validate({
        "display_count": 2,
        "question_sets": [
            {"id": "  FIRST_TOPIC ", "questions": ["  What is RAG?  "]},
            {"id": "second_topic", "questions": [" Another?\n"]},
        ],
    })
    assert [question_set.id for question_set in config.question_sets] == [
        "first_topic", "second_topic",
    ]
    assert [question_set.questions for question_set in config.question_sets] == [
        ["What is RAG?"], ["Another?"],
    ]
    assert len(ExampleQuestions.model_validate({
        "display_count": 1,
        "question_sets": [{"id": "topic", "questions": ["x" * 500]}],
    }).question_sets[0].questions[0]) == 500


def test_example_config_rejects_duplicate_normalized_set_ids() -> None:
    with pytest.raises(ValidationError, match="unique"):
        ExampleQuestions.model_validate({
            "display_count": 1,
            "question_sets": [
                {"id": " Topic ", "questions": ["First?"]},
                {"id": "TOPIC", "questions": ["Second?"]},
            ],
        })


def test_example_config_rejects_duplicate_questions_across_sets() -> None:
    with pytest.raises(ValidationError, match="unique across all sets"):
        ExampleQuestions.model_validate({
            "display_count": 1,
            "question_sets": [
                {"id": "first", "questions": ["What is RAG?"]},
                {"id": "second", "questions": ["  what  IS rag?  "]},
            ],
        })


def test_example_config_enforces_global_question_limit() -> None:
    with pytest.raises(ValidationError, match="50 questions in total"):
        ExampleQuestions.model_validate({
            "display_count": 1,
            "question_sets": [
                {"id": "first", "questions": [f"First {index}" for index in range(30)]},
                {"id": "second", "questions": [f"Second {index}" for index in range(21)]},
            ],
        })


def test_example_config_allows_fifty_questions_across_sets() -> None:
    config = ExampleQuestions.model_validate({
        "display_count": 50,
        "question_sets": [
            {"id": "first", "questions": [f"First {index}" for index in range(25)]},
            {"id": "second", "questions": [f"Second {index}" for index in range(25)]},
        ],
    })
    assert sum(len(question_set.questions) for question_set in config.question_sets) == 50


@pytest.mark.anyio
async def test_examples_api_only_returns_packaged_public_data() -> None:
    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://testserver") as client:
        response = await client.get("/ragdemo/api/examples?path=/private/config.json")
    assert response.status_code == 200
    assert response.json() == service.load_public_example_questions().model_dump()
    assert set(response.json()) == {"display_count", "question_sets"}
    assert all(set(question_set) == {"questions"} for question_set in response.json()["question_sets"])
    assert "berlin_process" not in response.text
    assert "/private" not in response.text


@pytest.mark.anyio
@pytest.mark.parametrize(
    "contents",
    [None, "{invalid json", '{"display_count": 1, "question_sets": []}'],
)
async def test_examples_api_safe_error_for_unavailable_or_invalid_config(
    monkeypatch: Any, tmp_path: Path, contents: str | None,
) -> None:
    config_path = tmp_path / "private-config.json"
    if contents is not None:
        config_path.write_text(contents)
    monkeypatch.setattr(service, "_QUESTIONS_PATH", config_path)
    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://testserver") as client:
        response = await client.get("/ragdemo/api/examples")
    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable", "message": "Example questions are temporarily unavailable."
    }
    assert str(tmp_path) not in response.text


def _document_public_id(point: SimpleNamespace) -> str:
    return inspect_index(_client(([point], None)), "demo").documents[0].id


@pytest.mark.anyio
async def test_chunk_details_endpoint_returns_one_safe_public_projection(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(routes, "to_thread", _run_inline)
    raw_source = "https://user:secret@example.com/docs/report.pdf?token=private"
    point = _point(
        "raw-qdrant-id",
        content=f"Evidence reproduced from {raw_source}.",
        source=raw_source,
        loader_id="private-loader",
        document_id="private-document",
        content_hash="private-content-hash",
        chunk_hash="private-chunk-hash",
        credentials="private-secret",
    )
    public_id = service._point_display_id(point)
    reader = _client(([point], None))
    monkeypatch.setattr(routes, "demo_qdrant_reader", reader)
    monkeypatch.setattr(routes, "user_config", {"collection_name": "configured-demo"})

    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://testserver") as client:
        response = await client.get(
            f"/ragdemo/api/chunks/{public_id}",
            params={
                "collection_name": "private",
                "source": "/private/path",
                "loader_id": "private-loader",
                "content_hash": "private-hash",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "id": public_id,
        "source": "report.pdf — example.com",
        "content": "Evidence reproduced from report.pdf — example.com.",
        "truncated": False,
    }
    assert set(response.json()) == {"id", "source", "content", "truncated"}
    for forbidden in (
        raw_source,
        "raw-qdrant-id",
        "private-loader",
        "private-document",
        "private-content-hash",
        "private-chunk-hash",
        "credentials",
        "vector",
        "/private/",
    ):
        assert forbidden not in response.text
    call = reader.scroll.call_args.kwargs
    assert call["collection_name"] == "configured-demo"
    assert call["with_vectors"] is False
    assert call["with_payload"] == service.KEYWORD_PAYLOAD_FIELDS
    assert call["limit"] <= service.MAX_CHUNK_LOOKUP_CHUNKS


@pytest.mark.anyio
@pytest.mark.parametrize("public_id", ["0" * 24, "raw-qdrant-id", "A" * 24])
async def test_chunk_details_endpoint_unknown_and_malformed_ids_are_safe(
    monkeypatch: Any, public_id: str,
) -> None:
    monkeypatch.setattr(routes, "to_thread", _run_inline)
    reader = _client(([], None))
    monkeypatch.setattr(routes, "demo_qdrant_reader", reader)
    monkeypatch.setattr(routes, "user_config", {"collection_name": "configured-demo"})
    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://testserver") as client:
        response = await client.get(f"/ragdemo/api/chunks/{public_id}")
    assert response.status_code == 404
    assert response.json()["message"] == "The chunk was not found."
    assert public_id not in response.text
    if not re.fullmatch(r"[0-9a-f]{24}", public_id):
        reader.collection_exists.assert_not_called()
        reader.scroll.assert_not_called()


def test_chunk_details_lookup_is_paginated_bounded_and_truncates_content() -> None:
    first = _point("first", content="First", source="/private/first.txt")
    second = _point("second", content="Second", source="/private/second.txt")
    target = _point("target", content="x" * 8_001, source="/private/target.txt")
    target_id = service._point_display_id(target)
    reader = _client(([first, second], "next"), ([target], None))

    result = inspect_chunk_details(
        reader,
        "configured-demo",
        target_id,
        page_size=2,
        max_chunks=3,
    )

    assert result is not None
    assert result.id == target_id
    assert result.source == "target.txt"
    assert len(result.content) == MAX_PUBLIC_CHUNK_CHARS
    assert result.truncated is True
    assert [call.kwargs["limit"] for call in reader.scroll.call_args_list] == [2, 1]
    assert all(call.kwargs["with_vectors"] is False for call in reader.scroll.call_args_list)
    assert all(call.kwargs["collection_name"] == "configured-demo" for call in reader.scroll.call_args_list)

    incomplete = _client(([first], "more"))
    with pytest.raises(ValueError, match="incomplete"):
        inspect_chunk_details(incomplete, "configured-demo", "0" * 24, max_chunks=1)
    with pytest.raises(ValueError, match="bounds"):
        inspect_chunk_details(reader, "configured-demo", target_id, max_chunks=service.MAX_CHUNK_LOOKUP_CHUNKS + 1)


@pytest.mark.anyio
async def test_chunk_details_endpoint_failure_is_generic(monkeypatch: Any) -> None:
    monkeypatch.setattr(routes, "to_thread", _run_inline)
    reader = _client()
    reader.scroll.side_effect = RuntimeError("secret /home/private collection=private")
    monkeypatch.setattr(routes, "demo_qdrant_reader", reader)
    monkeypatch.setattr(routes, "user_config", {"collection_name": "configured-demo"})
    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://testserver") as client:
        response = await client.get(f"/ragdemo/api/chunks/{'0' * 24}")
    assert response.status_code == 503
    assert response.json()["message"] == "The chunk is temporarily unavailable. Please try again shortly."
    assert "secret" not in response.text
    assert "/home" not in response.text


@pytest.mark.anyio
async def test_chunks_endpoint_uses_index_id_and_shared_chunk_id(monkeypatch: Any) -> None:
    monkeypatch.setattr(routes, "to_thread", _run_inline)
    point = _vector_point(
        "private-point-id", [0.1, 0.2], content="An actual chunk.",
        source="/home/private/guide.txt", loader_id="private-loader",
        document_id="private-document", content_hash="private-content-hash",
        chunk_hash="private-chunk-hash", credentials="private-secret",
    )
    point.score = 0.7
    index = inspect_index(_client(([point], None)), "demo")
    assert set(index.model_dump()) == {
        "collection", "status", "document_count", "chunk_count", "documents", "truncated"
    }
    assert set(index.documents[0].model_dump()) == {"id", "name", "chunk_count", "source_type"}
    public_id = index.documents[0].id
    reader = _client(([point], None))
    monkeypatch.setattr(routes, "demo_qdrant_reader", reader)
    monkeypatch.setattr(routes, "user_config", {"collection_name": "configured-demo"})
    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://testserver") as client:
        response = await client.get(f"/ragdemo/api/index/{public_id}/chunks?collection_name=private")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"document_id", "document_name", "chunks", "truncated"}
    assert body["document_id"] == public_id
    assert body["document_name"] == "guide.txt"
    assert body["truncated"] is False
    chunk = body["chunks"][0]
    assert set(chunk) == {"id", "content", "display_order", "truncated"}
    assert chunk["content"] == "An actual chunk."
    assert chunk["display_order"] == 1
    assert len(chunk["id"]) == 24
    assert set(chunk["id"]) <= set("0123456789abcdef")
    assert chunk["id"] == service._public_search_result(point, 1).id
    assert chunk["id"] == service._point_display_id(point)  # Explorer's identity helper.
    for forbidden in ("/home/", "private-", "loader_id", "content_hash", "chunk_hash", "vector", "credentials"):
        assert forbidden not in response.text
    call = reader.scroll.call_args.kwargs
    assert call["collection_name"] == "configured-demo"
    assert call["with_vectors"] is False
    assert set(call["with_payload"]) == set(service.KEYWORD_PAYLOAD_FIELDS)
    assert call["limit"] == service.SCROLL_PAGE_SIZE


@pytest.mark.anyio
@pytest.mark.parametrize("public_id", ["0" * 24, "raw-qdrant-id", "raw-loader-id"])
async def test_chunks_endpoint_unknown_ids_are_safe(monkeypatch: Any, public_id: str) -> None:
    monkeypatch.setattr(routes, "to_thread", _run_inline)
    reader = _client(([], None))
    monkeypatch.setattr(routes, "demo_qdrant_reader", reader)
    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://testserver") as client:
        response = await client.get(f"/ragdemo/api/index/{public_id}/chunks")
    assert response.status_code == 404
    assert response.json()["message"] == "The indexed document was not found."
    assert public_id not in response.text
    if len(public_id) != 24:
        reader.scroll.assert_not_called()


@pytest.mark.anyio
async def test_chunks_endpoint_failure_is_generic(monkeypatch: Any) -> None:
    monkeypatch.setattr(routes, "to_thread", _run_inline)
    reader = _client()
    reader.scroll.side_effect = RuntimeError("secret /home/private collection=private")
    monkeypatch.setattr(routes, "demo_qdrant_reader", reader)
    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://testserver") as client:
        response = await client.get(f"/ragdemo/api/index/{'0' * 24}/chunks")
    assert response.status_code == 503
    assert response.json()["message"] == "Document chunks are temporarily unavailable. Please try again shortly."
    assert "secret" not in response.text
    assert "/home" not in response.text


def test_document_chunks_pagination_grouping_order_and_bounds() -> None:
    points = [_point(str(i), source="/private/doc.txt", content=f"Text {i}") for i in range(4)]
    other = _point("other", source="/private/other.txt", content="Unrelated")
    public_id = _document_public_id(points[0])
    reader = _client(([points[0], other], "offset-1"), (points[1:3], "offset-2"), ([points[3]], None))
    result = service.inspect_document_chunks(reader, "demo", public_id, page_size=2)
    assert result is not None and result.truncated is False
    assert [chunk.display_order for chunk in result.chunks] == [1, 2, 3, 4]
    assert [chunk.id for chunk in result.chunks] == sorted(service._point_display_id(p) for p in points)
    assert all(chunk.content != "Unrelated" for chunk in result.chunks)
    assert [call.kwargs["offset"] for call in reader.scroll.call_args_list] == [None, "offset-1", "offset-2"]
    assert all(call.kwargs["limit"] == 2 and call.kwargs["with_vectors"] is False for call in reader.scroll.call_args_list)
    reverse = service.inspect_document_chunks(_client((list(reversed(points)), None)), "demo", public_id)
    assert reverse == result
    limited = service.inspect_document_chunks(
        _client((points, None)), "demo", public_id, max_document_chunks=2,
    )
    assert limited is not None and limited.truncated
    assert limited.chunks == result.chunks[:2]
    reader = _client((points[:2], "next"), ([points[2]], "more"))
    scanned = service.inspect_document_chunks(reader, "demo", public_id, page_size=2, max_chunks=3)
    assert scanned is not None and scanned.truncated
    assert len(scanned.chunks) == 3
    assert [call.kwargs["limit"] for call in reader.scroll.call_args_list] == [2, 1]


@pytest.mark.parametrize("second_page", [[], [_point("1", source="/private/doc.txt", content="Text")]])
def test_document_chunks_stops_empty_or_repeated_cursor(second_page: list[SimpleNamespace]) -> None:
    point = _point("1", source="/private/doc.txt", content="Text")
    reader = _client(([point], "repeat"), (second_page, "repeat"))
    result = service.inspect_document_chunks(reader, "demo", _document_public_id(point))
    assert result is not None and result.truncated
    assert reader.scroll.call_count == 2
    assert len(result.chunks) == 1


def test_document_chunks_incomplete_unknown_lookup_and_oversized_page_fail() -> None:
    point = _point("1", source="/private/doc.txt", content="Text")
    with pytest.raises(ValueError, match="incomplete"):
        service.inspect_document_chunks(_client(([point], "next")), "demo", "0" * 24, max_chunks=1)
    with pytest.raises(ValueError, match="page limit"):
        service.inspect_document_chunks(_client(([point, point], None)), "demo", "0" * 24, page_size=1)
    missing = _client()
    missing.collection_exists.return_value = False
    assert service.inspect_document_chunks(missing, "demo", "0" * 24) is None
    missing.scroll.assert_not_called()


@pytest.mark.parametrize("length", [8_000, 8_001, 20_000])
def test_document_chunk_content_is_bounded_and_marked(length: int) -> None:
    point = _point("1", source="/private/doc.txt", content="x" * length)
    result = service.inspect_document_chunks(_client(([point], None)), "demo", _document_public_id(point))
    assert result is not None
    assert len(result.chunks[0].content) == min(length, 8_000)
    assert result.chunks[0].truncated == (length > 8_000)
    assert result.truncated == (length > 8_000)


def test_document_chunk_reproduced_source_is_sanitized() -> None:
    source = "https://user:secret@example.com/doc.txt?token=private"
    point = _point("1", source=source, content=f"See {source}")
    result = service.inspect_document_chunks(_client(([point], None)), "demo", _document_public_id(point))
    assert result is not None
    assert result.chunks[0].content == "See doc.txt — example.com"
    assert "secret" not in result.model_dump_json()


def test_phase8_frontend_interactions_execute_in_node() -> None:
    """Run the actual script with a small DOM/fetch double; no JS dependency."""
    import subprocess

    javascript_path = Path(__file__).resolve().parents[1] / "static/ragdemo.js"
    assert "innerHTML" not in javascript_path.read_text()
    harness = r'''
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
class Element {
  constructor(tag = "div") {
    this.tag = tag; this.children = []; this.events = {}; this.attrs = {};
    this.dataset = {}; this.hidden = false; this.disabled = false; this.value = "";
    this.className = ""; this._text = "";
  }
  set textContent(value) { this._text = value; this.children = []; }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(""); }
  appendChild(child) {
    if (child.parent) child.parent.children = child.parent.children.filter(candidate => candidate !== child);
    this.children.push(child); child.parent = this; return child;
  }
  append(...children) { children.forEach(c => this.appendChild(c)); }
  replaceChildren() { this.children = []; this._text = ""; }
  setAttribute(key, value) { this.attrs[key] = value; }
  addEventListener(event, fn) { this.events[event] = fn; }
  querySelectorAll(selector) {
    return this.children.filter(c => selector === ".example-question" && c.className.includes("example-question"));
  }
  focus() { this.focused = true; }
  requestSubmit() { throw new Error("Unexpected RAG submit"); }
  click() { if (!this.disabled) return this.events.click?.(); }
}
(async () => {
  const elements = new Map();
  const get = selector => {
    if (!elements.has(selector)) elements.set(selector, new Element());
    return elements.get(selector);
  };
  const semanticInput = new Element("input");
  semanticInput.value = "semantic";
  semanticInput.checked = true;
  const keywordInput = new Element("input");
  keywordInput.value = "keyword";
  keywordInput.checked = false;
  const retrievalInputs = [semanticInput, keywordInput];
  const calls = [];
      let resolveChunks;
      let examplesRequestCount = 0;
      let chunkARequestCount = 0;
  const chunkData = {document_id: "a", document_name: "Doc A", truncated: true,
    chunks: [{id: "safe-chunk", content: "<script>untrusted text</script>", display_order: 1, truncated: true}]};
  const initialExamples = {display_count: 3, question_sets: [
    {questions: ["Process A?", "Process B?"]},
    {questions: ["Tourism A?", "Tourism B?"]},
    {questions: ["Climate A?", "Climate B?"]},
  ]};
  const refreshedExamples = {display_count: 3, question_sets: [
    {questions: ["New process?"]},
    {questions: ["New tourism?"]},
    {questions: ["New climate?"]},
  ]};
  const context = {
    document: {
      querySelector: get,
      querySelectorAll: selector => selector === "input[name='retrieval_mode']" ? retrievalInputs : [],
      createElement: tag => new Element(tag),
    },
    Math: Object.assign(Object.create(Math), {random: () => 0.5}),
      fetch: async (url, options = {}) => {
        calls.push(url);
      if (url === "./api/capabilities") return {ok: true, json: async () => ({retrieval_modes: ["semantic", "keyword"]})};
      if (url === "./api/examples") {
        examplesRequestCount += 1;
        if (examplesRequestCount === 1) return {ok: true, json: async () => initialExamples};
        if (examplesRequestCount === 2) return {ok: true, json: async () => refreshedExamples};
        return {ok: false};
      }
          if (url === "./api/index") return {ok: true, json: async () => ({document_count: 2, chunk_count: 2, truncated: false,
            documents: [{id: "a", name: "Doc A", chunk_count: 1}, {id: "b", name: "Doc B", chunk_count: 1}]})};
      if (url === "./api/index/a/chunks") {
        chunkARequestCount += 1;
        if (chunkARequestCount === 1) return await new Promise(resolve => { resolveChunks = resolve; });
        return {ok: true, json: async () => chunkData};
      }
      if (url === "./api/index/b/chunks") return {ok: false};
      throw new Error("Unexpected request " + url);
    },
  };
  vm.runInNewContext(fs.readFileSync(process.argv[1], "utf8"), context);
  const flush = () => new Promise(resolve => setImmediate(resolve));
  await flush();
  assert.deepEqual(calls.sort(), ["./api/examples", "./api/index"]);
  const examples = get("#example-questions");
  const examplesStatus = get("#examples-status");
  assert.equal(examples.children.length, 3);
  assert.deepEqual(
    examples.children.map(c => c.textContent.split(" ")[0]).sort(),
    ["Climate", "Process", "Tourism"],
  );
  assert.equal(examplesStatus.hidden, true);
  assert.equal(examplesStatus.textContent, "");
  ["curated", "static", "hardcoded", "generated"].forEach(word => {
    assert(!examplesStatus.textContent.toLowerCase().includes(word));
  });
  const original = examples.children.map(c => c.textContent).sort().join();
  const question = get("#question-input");
  question.value = "  My typed question\n";
  semanticInput.checked = false;
  keywordInput.checked = true;
  get("#query-results").hidden = false;
  get("#answer-content").textContent = "Existing answer";
  get("#search-results").textContent = "Existing search";
  get("#visualization-canvas").textContent = "Existing visualization";
  get("#prompt-messages").textContent = "Existing prompt";
  const exampleRefresh = get("#refresh-examples").click();
  assert.equal(get("#refresh-examples").disabled, true);
  await exampleRefresh;
  assert.equal(get("#refresh-examples").disabled, false);
  assert.equal(examples.children.length, 3);
  assert.notEqual(examples.children.map(c => c.textContent).sort().join(), original);
  assert.deepEqual(
    examples.children.map(c => c.textContent).sort(),
    ["New climate?", "New process?", "New tourism?"],
  );
  assert.equal(examplesRequestCount, 2);
  assert.equal(question.value, "  My typed question\n");
  assert.equal(semanticInput.checked, false);
  assert.equal(keywordInput.checked, true);
  assert.equal(get("#query-results").hidden, false);
  assert.equal(get("#answer-content").textContent, "Existing answer");
  assert.equal(get("#search-results").textContent, "Existing search");
  assert.equal(get("#visualization-canvas").textContent, "Existing visualization");
  assert.equal(get("#prompt-messages").textContent, "Existing prompt");
  assert.equal(examplesStatus.hidden, true);
  assert.equal(examplesStatus.textContent, "");
  const visibleBeforeFailure = examples.children.map(c => c.textContent);
  await get("#refresh-examples").click();
  assert.equal(examplesRequestCount, 3);
  assert.equal(examplesStatus.hidden, false);
  assert.equal(examplesStatus.textContent, "Example questions are temporarily unavailable.");
  assert.deepEqual(examples.children.map(c => c.textContent), visibleBeforeFailure);
  assert(examples.children.every(c => c.type === "button"));
  await examples.children[0].click();
  assert.equal(question.value, examples.children[0].textContent);
  assert.equal(question.focused, true);
  const cards = get("#document-list").children.map(column => column.children[0].children[0]);
  const firstButton = cards[0].children.find(c => c.tag === "button");
  const firstChunks = cards[0].children.at(-1);
  const otherButton = cards[1].children.find(c => c.tag === "button");
  const otherChunks = cards[1].children.at(-1);
  assert.equal(firstChunks.parent, cards[0]);
  assert.equal(firstChunks.hidden, true);
      assert.equal(firstButton.textContent, "Show chunks");
      assert.equal(firstButton.type, "button");
      const pending = firstButton.click();
  assert.equal(firstButton.disabled, true);
  assert.equal(firstChunks.textContent, "Loading chunks…");
  await firstButton.click();
  await otherButton.click();
  assert.match(otherChunks.textContent, /could not be loaded/);
  assert.equal(otherButton.disabled, false);
      resolveChunks({ok: true, json: async () => chunkData});
      await pending;
      await flush();
  assert.equal(firstButton.disabled, false);
  assert.equal(firstButton.textContent, "Hide chunks");
  assert.equal(firstButton.attrs["aria-expanded"], "true");
  const chunk = firstChunks.children.find(c => c.tag === "article");
  assert.equal(chunk.dataset.chunkId, "safe-chunk");
  assert.equal(chunk.children[1].textContent, "<script>untrusted text</script>");
  assert.equal(chunk.children[1].children.length, 0);
  assert.match(firstChunks.textContent, /shortened/);
  assert.match(firstChunks.textContent, /limited preview/);
  assert.match(firstChunks.textContent, /display order/);
  await firstButton.click();
  assert.equal(firstChunks.hidden, true);
  await firstButton.click();
  assert.equal(firstChunks.hidden, false);
  assert.equal(calls.filter(url => url.endsWith("/a/chunks")).length, 1);
  await get("#refresh-index").click();
  const rebuilt = get("#document-list").children[0].children[0].children[0];
  await rebuilt.children.find(c => c.tag === "button").click();
  assert.equal(calls.filter(url => url.endsWith("/a/chunks")).length, 2);
  await otherButton.click();
  await otherButton.click();
  assert.equal(calls.filter(url => url.endsWith("/b/chunks")).length, 2);
  assert(!calls.includes("./api/query/all"));
})().catch(error => { console.error(error); process.exitCode = 1; });
'''
    result = subprocess.run(
        ["node", "-e", harness, str(javascript_path)], capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_phase9_coordinate_graph_executes_in_node() -> None:
    """Exercise the actual Phase 9 renderer and controls with a small DOM double."""
    import subprocess

    javascript_path = Path(__file__).resolve().parents[1] / "static/ragdemo.js"
    harness = r'''
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

class ClassList {
  constructor(owner) { this.owner = owner; }
  values() { return new Set(this.owner.className.split(/\s+/).filter(Boolean)); }
  write(values) { this.owner.className = [...values].join(" "); }
  add(name) { const values = this.values(); values.add(name); this.write(values); }
  remove(name) { const values = this.values(); values.delete(name); this.write(values); }
  contains(name) { return this.values().has(name); }
  toggle(name, force) {
    const values = this.values();
    const enabled = force === undefined ? !values.has(name) : Boolean(force);
    if (enabled) values.add(name); else values.delete(name);
    this.write(values);
    return enabled;
  }
}

class Element {
  constructor(tag = "div", ownerDocument = null) {
    this.tag = tag; this.ownerDocument = ownerDocument; this.children = [];
    this.events = {}; this.eventOptions = {}; this.attrs = {}; this.dataset = {}; this.style = {
      removeProperty: key => { delete this.style[key]; },
    };
    this.hidden = false; this.disabled = false; this.value = ""; this.checked = false;
    this.className = ""; this._text = ""; this.classList = new ClassList(this);
    this.capturedPointers = new Set(); this.reparentCount = 0;
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(""); }
  appendChild(child) {
    if (child.parent) {
      child.parent.children = child.parent.children.filter(candidate => candidate !== child);
      child.reparentCount += 1;
    }
    this.children.push(child);
    child.parent = this; child.parentElement = this; child.parentNode = this;
    return child;
  }
  append(...children) { children.forEach(child => this.appendChild(child)); }
  insertBefore(child, reference) {
    const index = this.children.indexOf(reference);
    if (index < 0) return this.appendChild(child);
    this.children.splice(index, 0, child); child.parent = this; return child;
  }
  replaceChildren(...children) { this.children = []; this._text = ""; this.append(...children); }
  setAttribute(key, value) {
    this.attrs[key] = String(value);
    if (key === "class") this.className = String(value);
  }
  removeAttribute(key) { delete this.attrs[key]; }
  addEventListener(name, handler, options) {
    (this.events[name] ||= []).push(handler);
    (this.eventOptions[name] ||= []).push(options);
  }
  async dispatch(name, event = {}) {
    event.preventDefault ||= () => { event.defaultPrevented = true; };
    event.stopPropagation ||= () => { event.propagationStopped = true; };
    for (const handler of this.events[name] || []) await handler(event);
  }
  click() { if (!this.disabled) return this.dispatch("click"); }
  focus() { this.ownerDocument.activeElement = this; return this.dispatch("focus"); }
  blur() { this.ownerDocument.activeElement = null; return this.dispatch("blur"); }
  contains(candidate) {
    return candidate === this || this.children.some(child => child.contains(candidate));
  }
  closest(selector) {
    if (selector === "button[data-comparison-mode]" && this.dataset.comparisonMode) return this;
    return this.parent ? this.parent.closest(selector) : null;
  }
  querySelectorAll(selector) {
    return walk(this).filter(child => selector === ".example-question"
      && child.classList.contains("example-question"));
  }
  getBoundingClientRect() {
    if (this.classList.contains("embedding-map-tooltip")) {
      return {left: 0, top: 0, width: 180, height: 42};
    }
    if (this.selector === "#visualization-canvas"
        && this.ownerDocument.querySelector("#visualization-panel").classList.contains("is-expanded")) {
      return {left: 0, top: 0, width: 1400, height: 800};
    }
    return {left: 0, top: 0, width: 960, height: 600};
  }
  setPointerCapture(pointerId) { this.capturedPointers.add(pointerId); }
  hasPointerCapture(pointerId) { return this.capturedPointers.has(pointerId); }
  releasePointerCapture(pointerId) { this.capturedPointers.delete(pointerId); }
}

const walk = root => root.children.flatMap(child => [child, ...walk(child)]);
const byClass = (root, name) => walk(root).filter(child => child.classList.contains(name));
const geometry = element => ({...element.attrs, transform: element.attrs.transform || ""});

(async () => {
  const elements = new Map();
  const document = {
    activeElement: null,
    documentElement: {lang: "en"},
    querySelector(selector) {
      if (!elements.has(selector)) {
        const element = new Element("div", document);
        element.selector = selector;
        elements.set(selector, element);
      }
      return elements.get(selector);
    },
    querySelectorAll(selector) {
      if (selector === "input[name='retrieval_mode']") return retrievalInputs;
      if (selector === "[data-language]") return languageButtons;
      return [];
    },
    createElement(tag) { return new Element(tag, document); },
    createElementNS(_namespace, tag) { return new Element(tag, document); },
    createTextNode(text) { const node = new Element("#text", document); node.textContent = text; return node; },
  };
  const semantic = new Element("input", document);
  semantic.value = "semantic"; semantic.checked = true;
  const keyword = new Element("input", document);
  keyword.value = "keyword";
  const retrievalInputs = [semantic, keyword];
  const englishLanguage = new Element("button", document);
  englishLanguage.dataset.language = "en";
  const germanLanguage = new Element("button", document);
  germanLanguage.dataset.language = "de";
  const languageButtons = [englishLanguage, germanLanguage];
  const storedValues = new Map();
  let queryRequestCount = 0;
  const chunkDetailRequests = new Map();
  const semanticResponse = {
    question: "Where is the evidence?",
    answer: "Safe answer",
    search: {mode: "semantic", label: "Search", technical_label: "Dense", score_label: "Score", results: [
      {id: "one", rank: 1, source: "one.pdf", score: 0.9, content: "First safe chunk", matched_terms: []},
      {id: "two", rank: 2, source: "two.pdf", score: 0.8, content: "Second safe chunk", matched_terms: []},
    ]},
    visualization: {status: "ready", label: "Explore the embedding space", technical_label: "3D PCA", note: "3D proximity is approximate. Semantic similarity is calculated in the original full-dimensional embedding space used for retrieval.", points: [
      {id: "one", source: "one.pdf", x: -2, y: 1, z: 0.5, retrieved: true, rank: 1, similarity: 0.9, preview: "First safe chunk"},
      {id: "two", source: "two.pdf", x: 1, y: 2, z: -1, retrieved: true, rank: 2, similarity: 0.8, preview: "Second safe chunk"},
      {id: "other", source: "other.pdf", x: 3, y: -1, z: 2, retrieved: false, rank: null, similarity: 0.4, preview: "Background chunk"},
    ], query: {x: 0.75, y: -0.5, z: 1.25}, projection_method: "pca", projections: [
      {method: "pca", points: [
        {id: "one", x: -2, y: 1, z: 0.5},
        {id: "two", x: 1, y: 2, z: -1},
        {id: "other", x: 3, y: -1, z: 2},
      ], query: {x: 0.75, y: -0.5, z: 1.25}},
      {method: "tsne", points: [
        {id: "one", x: 20, y: -8, z: 4},
        {id: "two", x: -12, y: 16, z: 7},
        {id: "other", x: 5, y: 2, z: -18},
      ], query: {x: -4, y: -3, z: 11}},
      {method: "umap", points: [
        {id: "one", x: 2, y: -3, z: 4},
        {id: "two", x: -5, y: 6, z: 7},
        {id: "other", x: 8, y: 9, z: -10},
      ], query: {x: -1, y: -2, z: 3}, fallback: null},
    ]},
    prompt: {label: "Prompt", technical_label: "Messages", note: "Safe", messages: [
      {role: "user", content: "Question"},
    ]},
  };
  const keywordResponse = {
    ...semanticResponse,
    search: {...semanticResponse.search, mode: "keyword"},
    visualization: {status: "unsupported", label: "", technical_label: "", note: "", points: [], query: null},
  };
  const hybridResponse = {
    ...semanticResponse,
    search: {...semanticResponse.search, mode: "hybrid", score_label: "Fusion score"},
    visualization: {status: "unsupported", label: "", technical_label: "", note: "", points: [], query: null, projections: []},
  };
  const generativeResponse = {
    question: semanticResponse.question,
    answer: "Direct model answer",
    mode: "generative",
    label: "Generative AI",
    description: "Answer directly from the language model without retrieved documents.",
    retrieval_used: false,
    indexed_documents_used: false,
    prompt: {label: "Prompt", technical_label: "Messages", note: "Safe", messages: [
      {role: "system", content: "No retrieved documents or RAG context are provided."},
      {role: "human", content: semanticResponse.question},
    ]},
  };
  const aggregateResponse = {
    question: semanticResponse.question,
    results: {
      semantic: {status: "ready", result: semanticResponse, message: null},
      keyword: {status: "ready", result: keywordResponse, message: null},
      hybrid: {status: "ready", result: hybridResponse, message: null},
      generative: {status: "ready", result: generativeResponse, message: null},
    },
  };
  const context = {
    document,
    localStorage: {
      getItem: key => storedValues.has(key) ? storedValues.get(key) : null,
      setItem: (key, value) => storedValues.set(key, value),
    },
    fetch: async (url, options = {}) => {
      if (url === "./api/examples") return {ok: true, json: async () => ({display_count: 1, question_sets: [{questions: ["Example?"]}]})};
      if (url === "./api/index") return {ok: true, json: async () => ({document_count: 0, chunk_count: 0, truncated: false, documents: []})};
      if (url === "./api/query/all") {
        queryRequestCount += 1;
        const request = JSON.parse(options.body);
        assert.deepEqual(Object.keys(request), ["question"]);
        return {ok: true, json: async () => JSON.parse(JSON.stringify(aggregateResponse))};
      }
      if (url.startsWith("./api/chunks/")) {
        const id = decodeURIComponent(url.slice("./api/chunks/".length));
        const count = (chunkDetailRequests.get(id) || 0) + 1;
        chunkDetailRequests.set(id, count);
        if (id === "other" && count === 1) return {ok: false};
        const details = {
          one: {id: "one", source: "one.pdf", content: "First safe chunk", truncated: false},
          two: {id: "two", source: "two.pdf", content: "Second safe chunk", truncated: false},
          other: {id: "other", source: "other.pdf", content: "Background chunk content", truncated: false},
        }[id];
        return {ok: Boolean(details), json: async () => details};
      }
      throw new Error(`Unexpected request: ${url}`);
    },
  };
  document.querySelector("#projection-pca").dataset.projection = "pca";
  document.querySelector("#projection-tsne").dataset.projection = "tsne";
  document.querySelector("#projection-umap").dataset.projection = "umap";
  vm.runInNewContext(fs.readFileSync(process.argv[1], "utf8"), context);
  await new Promise(resolve => setImmediate(resolve));

  const question = document.querySelector("#question-input");
  question.value = "Where is the evidence?";
  await document.querySelector("#query-form").dispatch("submit");
  assert.equal(queryRequestCount, 1);
  const comparisonButtons = document.querySelector("#comparison-buttons");
  const comparisonButton = mode => comparisonButtons.children.find(button => button.dataset.comparisonMode === mode);
  const switchMode = mode => comparisonButtons.dispatch("click", {target: comparisonButton(mode)});
  assert.deepEqual(
    comparisonButtons.children.map(button => button.dataset.comparisonMode),
    ["generative", "semantic", "keyword", "hybrid"],
  );
  assert.deepEqual(
    comparisonButtons.children.map(button => button.attrs["aria-pressed"]),
    ["true", "false", "false", "false"],
  );
  assert.equal(document.querySelector("#generative-result").hidden, false);
  assert.equal(document.querySelector("#rag-stages").hidden, true);
  assert.equal(document.querySelector("#direct-answer-content").textContent, "Direct model answer");
  assert.equal(document.querySelector("#prompt-messages").textContent, "");
  await switchMode("semantic");
  assert.deepEqual(
    comparisonButtons.children.map(button => button.attrs["aria-pressed"]),
    ["false", "true", "false", "false"],
  );
  assert.equal(document.querySelector("#generative-result").hidden, true);
  assert.equal(document.querySelector("#rag-stages").hidden, false);
  assert.equal(queryRequestCount, 1);
  assert.equal(document.querySelector("#embed-stage").open, false);
  assert.equal(document.querySelector("#retrieve-stage").open, false);
  assert.equal(document.querySelector("#prompt-stage").open, false);
  assert.equal(document.querySelector("#answer-stage").open, true);
  document.querySelector("#embed-stage").open = true;
  document.querySelector("#retrieve-stage").open = true;
  document.querySelector("#prompt-stage").open = true;
  assert.equal(queryRequestCount, 1);
  const canvas = document.querySelector("#visualization-canvas");
  const svg = canvas.children.find(child => child.tag === "svg");
  assert(svg);
  const dispatchScenePointer = async (target, pointerId, movement = {x: 0, y: 0}) => {
    const pointerTarget = target.children.find(child => child.tag !== "title") || target;
    const pointer = {button: 0, pointerId, clientX: 480, clientY: 300, target: pointerTarget};
    await target.dispatch("pointerenter", pointer);
    const reparentCountBeforePress = target.reparentCount;
    await svg.dispatch("pointerdown", pointer);
    if (document.activeElement && document.activeElement !== target) {
      await document.activeElement.blur();
    }
    await target.focus();
    if (movement.x || movement.y) {
      await svg.dispatch("pointermove", {
        ...pointer,
        clientX: pointer.clientX + movement.x,
        clientY: pointer.clientY + movement.y,
      });
    }
    const reorderedDuringPress = target.reparentCount > reparentCountBeforePress;
    await svg.dispatch("pointerup", {
      ...pointer,
      clientX: pointer.clientX + movement.x,
      clientY: pointer.clientY + movement.y,
    });
    if (!reorderedDuringPress) {
      await target.dispatch("click", {detail: 1, target});
    }
    return reorderedDuringPress;
  };
  const inspector = document.querySelector("#visualization-inspector");
  const inspectorTitle = document.querySelector("#visualization-inspector-title");
  const inspectorContent = document.querySelector("#visualization-inspector-content");
  assert.equal(inspector.hidden, true);
  assert.equal(document.querySelector("#visualization-panel").hidden, false);
  assert.equal(document.querySelector("#visualization-note").hidden, true);
  assert.equal(document.querySelector("#visualization-note").textContent, "");
  assert.equal(byClass(svg, "embedding-axis").length, 3);
  assert.equal(byClass(svg, "embedding-origin").length, 1);
  assert.equal(byClass(svg, "embedding-point").length, 3);
  assert.equal(byClass(svg, "retrieved-point").length, 2);
  assert.equal(byClass(svg, "indexed-point").length, 1);
  assert.deepEqual(byClass(svg, "embedding-rank").map(rank => rank.textContent).sort(), ["#1", "#2"]);
  assert.equal(byClass(svg, "query-vector").length, 1);
  assert.equal(byClass(svg, "query-retrieval-connection").length, 2);
  assert.deepEqual(byClass(svg, "query-retrieval-connection").map(line => line.dataset.chunkId), ["one", "two"]);
  assert(!byClass(svg, "query-retrieval-connection").some(line => line.dataset.chunkId === "other"));
  const cuboids = byClass(svg, "embedding-cuboid");
  assert.equal(cuboids.length, 1);
  const cuboid = cuboids[0];
  const cuboidCorners = byClass(cuboid, "embedding-cuboid-corner");
  const cuboidEdges = byClass(cuboid, "embedding-cuboid-edge");
  assert.equal(cuboidCorners.length, 8);
  assert.equal(cuboidEdges.length, 12);
  assert.equal(new Set(cuboidCorners.map(corner => `${corner.dataset.x},${corner.dataset.y},${corner.dataset.z}`)).size, 8);
  assert.notEqual(Math.abs(Number(cuboid.dataset.xMinimum)), Math.abs(Number(cuboid.dataset.xMaximum)));
  [semanticResponse.visualization.query, ...semanticResponse.visualization.points].forEach(point => {
    assert(point.x >= Number(cuboid.dataset.xMinimum) && point.x <= Number(cuboid.dataset.xMaximum));
    assert(point.y >= Number(cuboid.dataset.yMinimum) && point.y <= Number(cuboid.dataset.yMaximum));
    assert(point.z >= Number(cuboid.dataset.zMinimum) && point.z <= Number(cuboid.dataset.zMaximum));
  });
  const gridLines = byClass(svg, "embedding-grid-line");
  const tickLabels = byClass(svg, "embedding-axis-tick-label");
  assert(gridLines.length >= 20);
  assert.deepEqual([...new Set(gridLines.map(line => line.dataset.plane))].sort(), ["xy", "xz", "yz"]);
  assert(byClass(svg, "embedding-grid-origin-line").length >= 2);
  assert(tickLabels.length >= 9 && tickLabels.length <= 18);
  assert(tickLabels.every(label => /^-?\d+(\.\d{1,4})?$/.test(label.textContent)));

  const axisLines = byClass(svg, "embedding-axis-line");
  const origin = byClass(svg, "embedding-origin")[0];
  const query = byClass(svg, "embedding-query-point")[0];
  const queryVector = byClass(svg, "query-vector")[0];
  const initialAxes = axisLines.map(geometry);
  const initialGrid = gridLines.map(geometry);
  const initialCuboid = [...cuboidEdges, ...cuboidCorners].map(geometry);
  const initialPoints = byClass(svg, "embedding-point").map(geometry);
  const initialQueryVector = geometry(queryVector);
  assert(axisLines.every(line => line.attrs.x1 === queryVector.attrs.x1 && line.attrs.y1 === queryVector.attrs.y1));
  assert.match(origin.attrs.transform, /^translate\(/);
  const translatedPoint = element => element.attrs.transform.match(/^translate\(([^ ]+) ([^)]+)/).slice(1).map(Number);
  const originPosition = translatedPoint(origin);
  assert.notDeepEqual(originPosition, [480, 300]);
  const queryPosition = translatedPoint(query);
  const vectorEnd = [Number(queryVector.attrs.x2), Number(queryVector.attrs.y2)];
  assert(Math.hypot(vectorEnd[0] - originPosition[0], vectorEnd[1] - originPosition[1])
    < Math.hypot(queryPosition[0] - originPosition[0], queryPosition[1] - originPosition[1]));
  assert.equal(queryVector.attrs["marker-end"], "url(#query-vector-arrowhead)");
  const lineContainsPoint = (line, point) => {
    const start = [Number(line.attrs.x1), Number(line.attrs.y1)];
    const end = [Number(line.attrs.x2), Number(line.attrs.y2)];
    const cross = (point[0] - start[0]) * (end[1] - start[1])
      - (point[1] - start[1]) * (end[0] - start[0]);
    const dot = (point[0] - start[0]) * (end[0] - start[0])
      + (point[1] - start[1]) * (end[1] - start[1]);
    const squaredLength = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2;
    return Math.abs(cross) <= Math.sqrt(squaredLength) * 1e-8
      && dot >= -1e-8 && dot <= squaredLength + 1e-8;
  };
  assert(byClass(svg, "embedding-grid-origin-line").some(
    line => line.dataset.plane === "xy" && lineContainsPoint(line, originPosition),
  ));

  const projectedCoordinates = () => {
    return [origin, query, ...byClass(svg, "embedding-point")].map(translatedPoint);
  };
  const assertSafeBounds = () => projectedCoordinates().forEach(([x, y]) => {
    assert(x >= 29.9 && x <= 930.1, `x ${x} outside rotation-safe data bounds`);
    assert(y >= 29.9 && y <= 570.1, `y ${y} outside rotation-safe data bounds`);
  });
  assertSafeBounds();
  const initialCoordinates = projectedCoordinates();
  const initialSpan = Math.max(
    Math.max(...initialCoordinates.map(([x]) => x)) - Math.min(...initialCoordinates.map(([x]) => x)),
    Math.max(...initialCoordinates.map(([, y]) => y)) - Math.min(...initialCoordinates.map(([, y]) => y)),
  );
  assert(initialSpan / (600 - 2 * 30) >= 0.72, `100% data span ${initialSpan} is still too compact`);

  await query.dispatch("mouseenter");
  const tooltip = document.querySelector("#visualization-tooltip");
  assert.equal(tooltip.hidden, false);
  assert.match(tooltip.textContent, /Query/);
  assert.match(tooltip.textContent, /X 0.750/);
  assert.match(tooltip.textContent, /Y -0.500/);
  assert.match(tooltip.textContent, /Z 1.250/);
  assert(tooltip.textContent.length < 80);
  await dispatchScenePointer(query, 1, {x: 1, y: 1});
  assert.equal(chunkDetailRequests.size, 0);
  assert.equal(inspector.hidden, false);
  assert.equal(inspectorTitle.textContent, "Query");
  assert.match(inspectorContent.textContent, /X0.750/);
  assert.match(inspectorContent.textContent, /Y-0.500/);
  assert.match(inspectorContent.textContent, /Z1.250/);
  await query.dispatch("mouseleave");
  assert.equal(inspector.hidden, false);
  await document.querySelector("#visualization-inspector-close").click();
  assert.equal(inspector.hidden, true);

  const retrieved = byClass(svg, "retrieved-point").find(point => point.dataset.chunkId === "one");
  await retrieved.dispatch("mouseenter");
  const linkedCard = walk(document.querySelector("#search-results"))
    .find(child => child.dataset.chunkId === "one");
  assert(linkedCard.classList.contains("is-linked"));
  assert(retrieved.classList.contains("is-linked"));
  assert(byClass(svg, "query-retrieval-connection")[0].classList.contains("is-active"));
  assert.match(tooltip.textContent, /#1 · one\.pdf/);
  assert.doesNotMatch(tooltip.textContent, /First safe chunk/);
  assert.match(tooltip.textContent, /X -2\.0/);
  assert(tooltip.textContent.length < 90);
  const geometryBeforeInspectorOpen = axisLines.map(geometry);
  const boundsBeforeInspectorOpen = canvas.getBoundingClientRect();
  const zoomBeforeInspectorOpen = document.querySelector("#visualization-zoom-level").textContent;
  const retrievedReorderedDuringPress = await dispatchScenePointer(retrieved, 2, {x: 2, y: 1});
  assert.equal(retrievedReorderedDuringPress, true);
  assert(retrieved.classList.contains("is-pinned"));
  assert.equal(inspector.hidden, false);
  assert.equal(inspectorTitle.textContent, "Chunk details");
  assert.match(inspectorContent.textContent, /Retrieved · Rank #1/);
  assert.match(inspectorContent.textContent, /one\.pdf/);
  assert.match(inspectorContent.textContent, /First safe chunk/);
  assert.match(inspectorContent.textContent, /Semantic similarity0\.900/);
  assert.match(inspectorContent.textContent, /X-2\.0/);
  assert.equal(chunkDetailRequests.get("one"), 1);
  const geometryBeforeLanguageSwitch = axisLines.map(geometry);
  const resultTextBeforeLanguageSwitch = document.querySelector("#search-results").textContent;
  const questionBeforeLanguageSwitch = document.querySelector("#question-input").value;
  const answerBeforeLanguageSwitch = document.querySelector("#answer-content").textContent;
  const requestCountBeforeLanguageSwitch = queryRequestCount;
  await germanLanguage.click();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(document.documentElement.lang, "de");
  assert.equal(storedValues.get("learn2rag-ragdemo-language"), "de");
  assert.equal(inspectorTitle.textContent, "Chunk-Details");
  assert.match(inspectorContent.textContent, /Abgerufen · Rang #1/);
  assert.match(inspectorContent.textContent, /Semantische Ähnlichkeit0\.900/);
  assert.match(inspectorContent.textContent, /Quelleone\.pdf/);
  assert.match(inspectorContent.textContent, /Chunk-InhaltFirst safe chunk/);
  assert.equal(document.querySelector("#question-input").value, questionBeforeLanguageSwitch);
  assert.equal(document.querySelector("#answer-content").textContent, answerBeforeLanguageSwitch);
  assert.match(document.querySelector("#search-results").textContent, /#1one\.pdf/);
  assert.match(document.querySelector("#search-results").textContent, /0\.900First safe chunk/);
  assert.match(document.querySelector("#search-results").textContent, /#2two\.pdf/);
  assert.match(document.querySelector("#search-results").textContent, /0\.800Second safe chunk/);
  assert.equal(semantic.checked, true);
  assert.equal(keyword.checked, false);
  assert.equal(queryRequestCount, requestCountBeforeLanguageSwitch);
  assert.equal(chunkDetailRequests.get("one"), 1);
  assert.deepEqual(axisLines.map(geometry), geometryBeforeLanguageSwitch);
  await englishLanguage.click();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(document.documentElement.lang, "en");
  assert.equal(inspectorTitle.textContent, "Chunk details");
  assert.match(inspectorContent.textContent, /Retrieved · Rank #1/);
  assert.match(inspectorContent.textContent, /Semantic similarity0\.900/);
  assert.equal(document.querySelector("#search-results").textContent, resultTextBeforeLanguageSwitch);
  assert.equal(document.querySelector("#answer-content").textContent, answerBeforeLanguageSwitch);
  assert.equal(queryRequestCount, requestCountBeforeLanguageSwitch);
  assert.deepEqual(axisLines.map(geometry), geometryBeforeLanguageSwitch);
  await germanLanguage.click();
  await new Promise(resolve => setImmediate(resolve));
  assert.match(document.querySelector("#search-results").textContent, /First safe chunk/);
  assert.equal(queryRequestCount, requestCountBeforeLanguageSwitch);
  assert.match(document.querySelector("#prompt-messages").textContent, /Question/);
  await englishLanguage.click();
  await new Promise(resolve => setImmediate(resolve));
  await retrieved.dispatch("click", {detail: 1, target: retrieved});
  assert.equal(chunkDetailRequests.get("one"), 1);
  assert.deepEqual(axisLines.map(geometry), geometryBeforeInspectorOpen);
  assert.deepEqual(canvas.getBoundingClientRect(), boundsBeforeInspectorOpen);
  assert.equal(document.querySelector("#visualization-zoom-level").textContent, zoomBeforeInspectorOpen);
  assert.deepEqual(
    byClass(svg, "query-retrieval-connection").filter(line => line.classList.contains("is-active")).map(line => line.dataset.chunkId),
    ["one"],
  );
  await retrieved.dispatch("mouseleave");
  const geometryBeforeInspectorClose = axisLines.map(geometry);
  const zoomBeforeInspectorClose = document.querySelector("#visualization-zoom-level").textContent;
  await document.querySelector("#visualization-inspector-close").click();
  assert.equal(inspector.hidden, true);
  assert.deepEqual(axisLines.map(geometry), geometryBeforeInspectorClose);
  assert.equal(document.querySelector("#visualization-zoom-level").textContent, zoomBeforeInspectorClose);
  await dispatchScenePointer(retrieved, 3);
  assert.equal(inspector.hidden, false);
  assert.equal(chunkDetailRequests.get("one"), 1);
  const secondRetrieved = byClass(svg, "retrieved-point").find(point => point.dataset.chunkId === "two");
  await dispatchScenePointer(secondRetrieved, 9);
  assert.match(inspectorContent.textContent, /Retrieved · Rank #2/);
  assert.match(inspectorContent.textContent, /two\.pdf/);
  assert.match(inspectorContent.textContent, /Second safe chunk/);
  assert.doesNotMatch(inspectorContent.textContent, /First safe chunk/);
  assert.equal(chunkDetailRequests.get("two"), 1);
  await dispatchScenePointer(retrieved, 10);
  assert.match(inspectorContent.textContent, /Retrieved · Rank #1/);
  assert.equal(chunkDetailRequests.get("one"), 1);

  await svg.dispatch("pointerdown", {button: 0, pointerId: 7, clientX: 480, clientY: 300});
  assert.equal(svg.hasPointerCapture(7), false);
  await svg.dispatch("pointermove", {pointerId: 7, clientX: 690, clientY: 130});
  assert.equal(svg.hasPointerCapture(7), true);
  assert.notDeepEqual(axisLines.map(geometry), initialAxes);
  assert.notDeepEqual(gridLines.map(geometry), initialGrid);
  assert.notDeepEqual([...cuboidEdges, ...cuboidCorners].map(geometry), initialCuboid);
  assert.notDeepEqual(byClass(svg, "embedding-point").map(geometry), initialPoints);
  assert.notDeepEqual(geometry(queryVector), initialQueryVector);
  assertSafeBounds();
  await svg.dispatch("pointercancel", {pointerId: 7});
  assert.equal(svg.classList.contains("is-dragging"), false);
  assert.equal(svg.hasPointerCapture(7), false);
  await svg.dispatch("pointerdown", {button: 0, pointerId: 8, clientX: 180, clientY: 470});
  await svg.dispatch("pointermove", {pointerId: 8, clientX: 760, clientY: 420});
  assertSafeBounds();
  await svg.dispatch("pointerup", {pointerId: 8});
  await svg.dispatch("click");

  assert.equal(svg.eventOptions.wheel.length, 1);
  assert.equal(svg.eventOptions.wheel[0].passive, false);
  const wheelIn = {deltaY: -100};
  await svg.dispatch("wheel", wheelIn);
  assert.equal(wheelIn.defaultPrevented, true);
  assert.equal(document.querySelector("#visualization-zoom-level").textContent, "109%");
  const inspectorWheel = {deltaY: 100};
  await inspector.dispatch("wheel", inspectorWheel);
  assert.equal(inspectorWheel.propagationStopped, true);
  assert.equal(document.querySelector("#visualization-zoom-level").textContent, "109%");
  const panel = document.querySelector("#visualization-panel");
  const rotatedAt109 = axisLines.map(geometry);
  await document.querySelector("#visualization-expand").click();
  assert(panel.classList.contains("is-expanded"));
  assert.equal(document.querySelector("#visualization-expand").hidden, true);
  assert.equal(document.querySelector("#visualization-restore").hidden, false);
  assert.equal(document.querySelector("#visualization-expanded-close").hidden, false);
  assert.equal(document.querySelector("#visualization-zoom-level").textContent, "109%");
  assert(retrieved.classList.contains("is-pinned"));
  assert.equal(inspector.hidden, false);
  assert.notDeepEqual(axisLines.map(geometry), rotatedAt109);
  const expandedGeometryWithInspector = axisLines.map(geometry);
  const expandedBoundsWithInspector = canvas.getBoundingClientRect();
  await document.querySelector("#visualization-inspector-close").click();
  assert.equal(inspector.hidden, true);
  assert.equal(inspector.children.length, 0);
  assert.equal(inspectorContent.children.length, 0);
  assert(panel.classList.contains("is-expanded"));
  assert.equal(document.querySelector("#visualization-zoom-level").textContent, "109%");
  assert.deepEqual(axisLines.map(geometry), expandedGeometryWithInspector);
  assert.deepEqual(canvas.getBoundingClientRect(), expandedBoundsWithInspector);
  await dispatchScenePointer(retrieved, 4, {x: 2, y: 2});
  assert.equal(inspector.hidden, false);
  assert.equal(chunkDetailRequests.get("one"), 1);
  await document.querySelector("#visualization-restore").click();
  assert.equal(panel.classList.contains("is-expanded"), false);
  assert.deepEqual(axisLines.map(geometry), rotatedAt109);
  assert.equal(document.querySelector("#visualization-zoom-level").textContent, "109%");
  assert(retrieved.classList.contains("is-pinned"));

  await document.querySelector("#visualization-expand").click();
  await svg.dispatch("wheel", {deltaY: -100});
  assert.equal(document.querySelector("#visualization-zoom-level").textContent, "119%");
  await document.querySelector("#visualization-expanded-close").click();
  assert.equal(panel.classList.contains("is-expanded"), false);
  assert.equal(document.querySelector("#visualization-zoom-level").textContent, "119%");
  assert(retrieved.classList.contains("is-pinned"));

  await document.querySelector("#visualization-expand").click();
  await document.querySelector("#visualization-reset").click();
  assert(panel.classList.contains("is-expanded"));
  assert.equal(document.querySelector("#visualization-zoom-level").textContent, "100%");
  assert.equal(inspector.hidden, true);
  assert.equal(retrieved.classList.contains("is-pinned"), false);
  await document.querySelector("#visualization-expanded-close").click();
  assert.equal(panel.classList.contains("is-expanded"), false);
  assert.deepEqual(axisLines.map(geometry), initialAxes);
  assert.deepEqual(gridLines.map(geometry), initialGrid);
  assert.deepEqual([...cuboidEdges, ...cuboidCorners].map(geometry), initialCuboid);
  assert.deepEqual(byClass(svg, "embedding-point").map(geometry), initialPoints);
  assert.deepEqual(geometry(queryVector), initialQueryVector);
  assert.equal(tooltip.hidden, true);
  assert(byClass(svg, "query-retrieval-connection").every(line => !line.classList.contains("is-active")));

  const backgroundPoint = byClass(svg, "indexed-point")[0];
  await backgroundPoint.dispatch("mouseenter");
  assert.equal(tooltip.hidden, false);
  assert.match(tooltip.textContent, /other\.pdf/);
  assert.match(tooltip.textContent, /X 3\.0/);
  assert.doesNotMatch(tooltip.textContent, /Background chunk content/);
  await backgroundPoint.dispatch("mouseleave");
  const backgroundBeforeDrag = geometry(backgroundPoint);
  await dispatchScenePointer(backgroundPoint, 11, {x: 42, y: 18});
  assert.equal(chunkDetailRequests.has("other"), false);
  assert.equal(inspector.hidden, true);
  assert.notEqual(geometry(backgroundPoint).transform, backgroundBeforeDrag.transform);
  const backgroundGeometry = geometry(backgroundPoint);
  await germanLanguage.click();
  await new Promise(resolve => setImmediate(resolve));
  const backgroundReorderedDuringPress = await dispatchScenePointer(
    backgroundPoint, 5, {x: 1, y: 2},
  );
  assert.equal(backgroundReorderedDuringPress, true);
  assert.equal(inspectorTitle.textContent, "Chunk-Details");
  assert.match(inspectorContent.textContent, /Für diese Frage nicht abgerufen/);
  assert.match(inspectorContent.textContent, /other\.pdf/);
  assert.match(inspectorContent.textContent, /Dieser Chunk konnte nicht geladen werden/);
  assert.equal(chunkDetailRequests.get("other"), 1);
  await dispatchScenePointer(backgroundPoint, 6);
  await new Promise(resolve => setImmediate(resolve));
  assert.match(inspectorContent.textContent, /Chunk-InhaltBackground chunk content/);
  assert.match(inspectorContent.textContent, /Semantische Ähnlichkeit0\.400/);
  assert.equal(chunkDetailRequests.get("other"), 2);
  assert.equal(geometry(backgroundPoint).transform, backgroundGeometry.transform);
  assert(byClass(svg, "query-retrieval-connection").every(line => !line.classList.contains("is-active")));
  await document.querySelector("#visualization-inspector-close").click();
  await englishLanguage.click();
  await new Promise(resolve => setImmediate(resolve));

  semanticResponse.visualization.status = "partial";
  semanticResponse.visualization.note = "3D proximity is approximate. Semantic similarity is calculated in the original full-dimensional embedding space used for retrieval. The bounded display shows a partial index snapshot.";
  await document.querySelector("#query-form").dispatch("submit");
  await switchMode("semantic");
  assert.equal(document.querySelector("#visualization-note").hidden, false);
  assert.equal(document.querySelector("#visualization-note").textContent, "The bounded display shows a partial index snapshot.");

  const countBeforeProjectionSwitch = queryRequestCount;
  const answerBeforeProjectionSwitch = document.querySelector("#answer-content").textContent;
  const promptBeforeProjectionSwitch = document.querySelector("#prompt-messages").textContent;
  const resultsBeforeProjectionSwitch = document.querySelector("#search-results").textContent;
  const pcaSvg = document.querySelector("#visualization-canvas").children.find(child => child.tag === "svg");
  assert.deepEqual(byClass(pcaSvg, "embedding-axis-label").map(label => label.textContent), ["PCA 1", "PCA 2", "PCA 3"]);
  await document.querySelector("#projection-tsne").click();
  const tsneSvg = document.querySelector("#visualization-canvas").children.find(child => child.tag === "svg");
  assert.notEqual(tsneSvg, pcaSvg);
  assert.deepEqual(byClass(tsneSvg, "embedding-axis-label").map(label => label.textContent), ["t-SNE 1", "t-SNE 2", "t-SNE 3"]);
  assert.equal(byClass(tsneSvg, "embedding-origin").length, 0);
  assert.equal(byClass(tsneSvg, "query-vector").length, 0);
  assert.equal(byClass(tsneSvg, "embedding-query-point").length, 1);
  assert.equal(byClass(tsneSvg, "query-retrieval-connection").length, 2);
  assert.equal(document.querySelector("#query-legend-label").textContent, "Query");
  assert.deepEqual(byClass(tsneSvg, "embedding-point").map(point => point.dataset.chunkId).sort(), ["one", "other", "two"]);
  assert.deepEqual(byClass(tsneSvg, "embedding-rank").map(rank => rank.textContent).sort(), ["#1", "#2"]);
  assert.equal(queryRequestCount, countBeforeProjectionSwitch);
  assert.equal(document.querySelector("#answer-content").textContent, answerBeforeProjectionSwitch);
  assert.equal(document.querySelector("#prompt-messages").textContent, promptBeforeProjectionSwitch);
  assert.equal(document.querySelector("#search-results").textContent, resultsBeforeProjectionSwitch);
  assert.match(document.querySelector("#visualization-technical-label").textContent, /t-SNE/);
  assert.match(document.querySelector("#projection-explanation").textContent, /local neighbourhood/);
  const tsneRetrieved = byClass(tsneSvg, "retrieved-point").find(point => point.dataset.chunkId === "one");
  await tsneRetrieved.dispatch("click", {detail: 0, target: tsneRetrieved});
  assert.match(inspectorContent.textContent, /Semantic similarity0\.900/);
  assert.match(inspectorContent.textContent, /t-SNE coordinates/);
  await germanLanguage.click();
  assert.match(document.querySelector("#projection-explanation").textContent, /lokale Nachbarschaftsstrukturen/);
  assert.match(document.querySelector("#visualization-technical-label").textContent, /t-SNE-Projektion/);
  await document.querySelector("#projection-umap").click();
  const umapSvg = document.querySelector("#visualization-canvas").children.find(child => child.tag === "svg");
  assert.deepEqual(byClass(umapSvg, "embedding-axis-label").map(label => label.textContent), ["UMAP 1", "UMAP 2", "UMAP 3"]);
  assert.equal(byClass(umapSvg, "embedding-origin").length, 0);
  assert.equal(byClass(umapSvg, "query-vector").length, 0);
  assert.equal(byClass(umapSvg, "embedding-query-point").length, 1);
  assert.equal(byClass(umapSvg, "query-retrieval-connection").length, 2);
  assert.equal(document.querySelector("#query-legend-label").textContent, "Frage");
  assert.deepEqual(byClass(umapSvg, "embedding-point").map(point => point.dataset.chunkId).sort(), ["one", "other", "two"]);
  assert.deepEqual(byClass(umapSvg, "embedding-rank").map(rank => rank.textContent).sort(), ["#1", "#2"]);
  assert.equal(queryRequestCount, countBeforeProjectionSwitch);
  assert.match(document.querySelector("#projection-explanation").textContent, /Nachbarschaftsstrukturen/);
  const umapRetrieved = byClass(umapSvg, "retrieved-point").find(point => point.dataset.chunkId === "one");
  await umapRetrieved.dispatch("click", {detail: 0, target: umapRetrieved});
  assert.match(inspectorContent.textContent, /Semantische Ähnlichkeit0\.900/);
  assert.match(inspectorContent.textContent, /UMAP-Koordinaten/);
  await document.querySelector("#projection-pca").click();
  const restoredPcaSvg = document.querySelector("#visualization-canvas").children.find(child => child.tag === "svg");
  assert.equal(byClass(restoredPcaSvg, "embedding-origin").length, 1);
  assert.equal(byClass(restoredPcaSvg, "query-vector").length, 1);
  assert.equal(document.querySelector("#query-legend-label").textContent, "Fragevektor");
  assert.equal(queryRequestCount, countBeforeProjectionSwitch);
  assert.match(document.querySelector("#visualization-technical-label").textContent, /PCA-Projektion/);

  const sameQuestion = document.querySelector("#embedded-question").textContent;
  await switchMode("keyword");
  assert.equal(document.querySelector("#visualization-section").hidden, true);
  assert.equal(document.querySelector("#visualization-panel").hidden, true);
  assert.equal(document.querySelector("#embedded-question").textContent, sameQuestion);
  assert.equal(queryRequestCount, countBeforeProjectionSwitch);
  assert.match(document.querySelector("#search-results").textContent, /First safe chunk/);
  await switchMode("hybrid");
  assert.equal(document.querySelector("#score-label").textContent, "Fusionswert");
  assert.equal(queryRequestCount, countBeforeProjectionSwitch);
  await switchMode("generative");
  assert.equal(document.querySelector("#generative-result").hidden, false);
  assert.equal(document.querySelector("#rag-stages").hidden, true);
  assert.equal(document.querySelector("#direct-answer-content").textContent, "Direct model answer");
  assert.equal(document.querySelector("#prompt-messages").textContent, "");
  assert.equal(queryRequestCount, countBeforeProjectionSwitch);
  await switchMode("semantic");
  assert.equal(document.querySelector("#generative-result").hidden, true);
  assert.equal(document.querySelector("#rag-stages").hidden, false);
  assert.equal(document.querySelector("#embedded-question").textContent, sameQuestion);
  assert.equal(document.querySelector("#visualization-section").hidden, false);
  assert.equal(queryRequestCount, countBeforeProjectionSwitch);

  question.value = "A second question";
  await question.dispatch("input", {target: question});
  assert.equal(document.querySelector("#query-results").hidden, true);
  assert.equal(document.querySelector("#comparison-buttons").children.length, 0);
  aggregateResponse.question = question.value;
  [semanticResponse, keywordResponse, hybridResponse, generativeResponse]
    .forEach(result => { result.question = question.value; });
  await document.querySelector("#query-form").dispatch("submit");
  assert.equal(queryRequestCount, countBeforeProjectionSwitch + 1);
  assert.equal(document.querySelector("#comparison-buttons").children.length, 4);
  assert.equal(comparisonButton("generative").attrs["aria-pressed"], "true");
  assert.equal(document.querySelector("#direct-answer-content").textContent, "Direct model answer");
  await switchMode("semantic");
  assert.equal(document.querySelector("#embedded-question").textContent, "A second question");
})().catch(error => { console.error(error); process.exitCode = 1; });
'''
    result = subprocess.run(
        ["node", "-e", harness, str(javascript_path)], capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
