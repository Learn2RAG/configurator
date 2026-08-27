from pathlib import Path
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
from learn2rag.ragdemo.models import QueryRequest, QuerySearchResult
from learn2rag.ragdemo.service import DEMO_USER, build_query_visualization, inspect_index


DEMO_CONFIG = {
    "search_mode": "dense",
    "embedding_model": "BAAI/bge-m3",
    "prompt": "Use only the following information:\n{context}",
}
KEYWORD_CONFIG = {**DEMO_CONFIG, "top_k": 5}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/ragdemo")
    return app


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
    assert 'name="retrieval_mode" value="semantic" checked' in response.text
    assert 'name="retrieval_mode" value="keyword"' in response.text
    assert "Semantic vectors" in response.text
    assert "Search by meaning, not only exact words." in response.text
    assert "Keyword search" in response.text
    assert "Match the words you typed." in response.text
    assert 'id="ask-button"' in response.text
    assert 'id="retrieval-change-note"' in response.text
    assert 'id="comparison-actions"' in response.text
    assert 'id="compare-retrieval"' in response.text
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


def test_retrieval_change_hides_stale_results_and_comparison_submits_once() -> None:
    javascript = (
        Path(__file__).resolve().parents[1] / "static" / "ragdemo.js"
    ).read_text(encoding="utf-8")
    stale_clear = javascript[
        javascript.index("function clearStaleQueryResults"):
        javascript.index("function configureComparisonShortcut")
    ]
    comparison_handler = javascript[
        javascript.index('compareRetrievalButton.addEventListener("click"'):
        javascript.index('zoomOutButton.addEventListener("click"')
    ]

    assert "queryResults.hidden = true;" in stale_clear
    assert "comparisonActions.hidden = true;" in stale_clear
    assert "Retrieval method changed. Ask again to compare the same question." in stale_clear
    assert "questionInput.value" not in stale_clear
    assert (
        'input.addEventListener("change", () => clearStaleQueryResults());'
        in javascript
    )
    assert "targetMode = mode === \"semantic\" ? \"keyword\" : \"semantic\"" in javascript
    assert "targetInput.checked = true;" in comparison_handler
    assert "clearStaleQueryResults(false);" in comparison_handler
    assert comparison_handler.count("queryForm.requestSubmit();") == 1
    assert "questionInput.value =" not in comparison_handler
    assert "fetch(" not in comparison_handler
    assert "innerHTML" not in javascript


def test_public_demo_collection_scope_is_documented() -> None:
    service_source = (
        Path(__file__).resolve().parents[1] / "service.py"
    ).read_text(encoding="utf-8")

    assert "dedicated public demo collection" in service_source
    assert "arbitrary private production collection" in service_source


def test_visualization_frontend_uses_safe_dom_and_shared_chunk_ids() -> None:
    javascript = (
        Path(__file__).resolve().parents[1] / "static" / "ragdemo.js"
    ).read_text(encoding="utf-8")

    assert "document.createElementNS" in javascript
    assert "card.dataset.chunkId = result.id;" in javascript
    assert "group.dataset.chunkId = point.id;" in javascript
    assert "nativeTooltip.textContent" in javascript
    assert "rank.textContent = String(point.rank);" in javascript
    assert "content.textContent = preview;" in javascript
    assert 'document.createElement("mark")' in javascript
    assert "document.createTextNode" in javascript
    assert "highlight.textContent = token;" in javascript
    assert "chunk.textContent = result.content;" in javascript
    assert "retrieval_mode: retrievalMode" in javascript
    assert "visualizationSection.hidden = !isSemantic;" in javascript
    assert "innerHTML" not in javascript


def test_visualization_frontend_has_native_3d_interaction_and_no_external_dependency() -> None:
    static_dir = Path(__file__).resolve().parents[1] / "static"
    javascript = (static_dir / "ragdemo.js").read_text(encoding="utf-8")
    template = (
        Path(__file__).resolve().parents[1] / "templates" / "index.html"
    ).read_text(encoding="utf-8")
    service_source = (
        Path(__file__).resolve().parents[1] / "service.py"
    ).read_text(encoding="utf-8")

    assert 'svg.addEventListener("pointerdown"' in javascript
    assert 'svg.addEventListener("pointermove"' in javascript
    assert "svg.setPointerCapture(event.pointerId);" in javascript
    assert 'addEventListener("wheel"' not in javascript
    assert "event.deltaY" not in javascript
    assert "minimumZoom = 0.55" in javascript
    assert "maximumZoom = 2.6" in javascript
    assert "clamp(viewerState.zoom * factor, minimumZoom, maximumZoom)" in javascript
    assert 'zoomOutButton.addEventListener("click", () => zoomViewer(1 / 1.12));' in javascript
    assert 'zoomInButton.addEventListener("click", () => zoomViewer(1.12));' in javascript
    assert "resetViewButton.addEventListener" in javascript
    assert "rotationX: initialCamera.rotationX" in javascript
    assert 'svgElement("line", "query-connection")' in javascript
    assert "showConnectionLine(state.connectionLine" in javascript
    assert "togglePinnedChunk(point.id);" in javascript
    assert 'event.key === "Enter" || event.key === " "' in javascript
    assert "projectWorldPoint" in javascript
    assert "projected.depth - second.projected.depth" in javascript
    assert "Drag to rotate · Use +/− to zoom · Hover points to inspect" in template
    assert "Scroll to zoom" not in template
    assert "query_points" not in service_source
    for external_library in ("three.js", "plotly", "d3.js", "babylon", "cdnjs"):
        assert external_library not in template.casefold()


def test_visualization_frontend_fits_rotation_and_preserves_system_cursor() -> None:
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
    viewer_reset = javascript[
        javascript.index("function resetViewer"):
        javascript.index("function renderVisualization")
    ]

    assert "function normalizeWorldScene(coordinates)" in javascript
    assert "Math.hypot(point.x, point.y, point.z)" in javascript
    assert "point.x / normalizationRadius" in javascript
    assert "point.y / normalizationRadius" in javascript
    assert "point.z / normalizationRadius" in javascript
    assert "function computeFittedSceneScale(plotWidth, plotHeight)" in javascript
    assert "const edgePadding = plotPadding + maximumMarkerExtent;" in javascript
    assert "plotWidth - 2 * edgePadding" in javascript
    assert "plotHeight - 2 * edgePadding" in javascript
    assert "Math.min(safeWidth, safeHeight) / 2" in javascript
    assert "const plotBounds = Object.freeze" in javascript
    assert "width: width - plotInset * 2" in javascript
    assert "height: height - plotInset * 2" in javascript
    assert "computeFittedSceneScale(plotBounds.width, plotBounds.height)" in javascript
    assert 'background.setAttribute("width", String(plotBounds.width));' in javascript
    assert 'background.setAttribute("height", String(plotBounds.height));' in javascript
    assert "state.fittedSceneScale * state.zoom" in projection
    assert "rotatedX * sceneScale" in projection
    assert "rotatedY * sceneScale" in projection
    assert "perspective" not in projection
    assert "x: clamp(" not in projection
    assert "y: clamp(" not in projection
    assert "computeFittedSceneScale" not in scene_render

    assert "depth: rotatedZ" in projection
    assert "depthRatio:" in projection
    assert "projected.depth - second.projected.depth" in scene_render
    assert "const depthScale" in scene_render
    assert "projected.depthRatio" in scene_render
    assert "showConnectionLine(state.connectionLine, projectedQuery, activePoint.projected)" in scene_render

    assert "zoom: 1" in javascript
    assert "minimumZoom = 0.55" in javascript
    assert "maximumZoom = 2.6" in javascript
    assert "clamp(viewerState.zoom * factor, minimumZoom, maximumZoom)" in javascript
    assert "state.zoom = state.initialCamera.zoom;" in viewer_reset
    assert "state.fittedSceneScale =" not in viewer_reset

    assert 'document.querySelector("#visualization-zoom-level")' in javascript
    assert "function updateZoomControls(state)" in javascript
    assert "Math.round(state.zoom * 100)" in javascript
    assert 'zoomLevel.textContent = `${percentage}%`;' in javascript
    assert "updateZoomControls(viewerState);" in javascript
    assert "updateZoomControls(state);" in viewer_reset

    assert "function positionPointTooltip(projected, state)" in javascript
    assert "visualizationCanvas.getBoundingClientRect()" in javascript
    assert "state.svg.getBoundingClientRect()" in javascript
    assert "const tooltipLeft = clamp(" in javascript
    assert "const tooltipTop = clamp(" in javascript
    assert "positionPointTooltip(projected, state);" in javascript

    assert "cursor: grab" not in stylesheet
    assert "cursor: grabbing" not in stylesheet
    assert ".embedding-map.is-dragging .embedding-map-background" in stylesheet


def test_visualization_reset_clears_svg_line_camera_and_interaction_state() -> None:
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

    assert "function hideConnectionLine(line)" in javascript
    assert 'line.setAttribute("visibility", "hidden");' in javascript
    assert 'line.setAttribute("visibility", "visible");' in javascript
    assert '["x1", "y1", "x2", "y2"].forEach' in javascript
    assert "line.removeAttribute(attribute)" in javascript
    assert "connectionLine.hidden" not in javascript
    assert "state.connectionLine.hidden" not in javascript
    assert "hideConnectionLine(connectionLine);" in javascript

    assert "const initialCamera = Object.freeze" in javascript
    assert "initialCamera," in javascript
    assert "rotationX: initialCamera.rotationX" in javascript
    assert "rotationY: initialCamera.rotationY" in javascript
    assert "zoom: initialCamera.zoom" in javascript
    assert "state.rotationX = state.initialCamera.rotationX;" in viewer_reset
    assert "state.rotationY = state.initialCamera.rotationY;" in viewer_reset
    assert "state.zoom = state.initialCamera.zoom;" in viewer_reset

    assert "hoveredChunkId = null;" in viewer_reset
    assert "focusedChunkId = null;" in viewer_reset
    assert "pinnedChunkId = null;" in viewer_reset
    assert "resetPointerInteraction(state);" in viewer_reset
    assert "state.svg.releasePointerCapture(capturedPointerId);" in pointer_reset
    assert "state.pointerId = null;" in pointer_reset
    assert "state.lastPointerX = 0;" in pointer_reset
    assert "state.lastPointerY = 0;" in pointer_reset
    assert "state.dragMoved = false;" in pointer_reset
    assert 'state.svg.classList.remove("is-dragging");' in pointer_reset
    assert "activeElement.blur();" in viewer_reset
    assert "clearPointTooltip();" in viewer_reset
    assert "hideConnectionLine(state.connectionLine);" in viewer_reset
    assert "updateLinkedInteraction(false);" in viewer_reset
    assert "renderViewerScene();" in viewer_reset


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
        content=" ".join(many_terms),
        source="/private/bounded.txt",
        loader_id="directory",
        document_id="bounded-document",
    )
    bounded_points, bounded_matches = service._bm25_candidates(
        " ".join(many_terms),
        [bounded_record],
    )
    terms = bounded_matches[service._point_display_id(bounded_points[0])]
    assert len(terms) == service.MAX_MATCHED_TERMS == 12
    assert all(len(term) <= service.MAX_MATCHED_TERM_LENGTH for term in terms)


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
        == {"id", "source", "x", "y", "z", "retrieved", "rank", "preview"}
        for point in result.points
    )
    assert "vector" not in serialized
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
        "id", "source", "x", "y", "z", "retrieved", "rank", "preview",
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
