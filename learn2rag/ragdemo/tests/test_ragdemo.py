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
from learn2rag.ragdemo.models import ExampleQuestions, QueryRequest, QuerySearchResult
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
    assert "X/Y/Z show the first three PCA components" in template
    assert "Retrieval itself uses the full-dimensional embeddings." in template
    assert "Query vector" in template
    assert "Retrieved chunks" in template
    assert "Other indexed chunks" in template
    assert "Query → retrieved link" in template
    assert 'id="visualization-expand"' in template
    assert 'id="visualization-restore"' in template
    assert 'id="visualization-expanded-close"' in template
    assert "Top-ranked chunks retrieved as a set" in template
    assert "Retrieved chunks added to the prompt" in template


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
        javascript.index("function safePointPreview")
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
    assert "grid-template-rows: auto auto minmax(0, 1fr) auto;" in expanded_rule
    assert 'visualizationPanel.classList.toggle("is-expanded", expanded);' in presentation
    assert "resizeViewerToCanvas(viewerState);" in presentation
    assert "viewerState =" not in presentation
    assert ".rotation =" not in presentation
    assert ".zoom =" not in presentation
    assert "pinnedChunkId =" not in presentation
    assert 'expandViewButton.addEventListener("click", () => setPresentationMode(true));' in javascript
    assert 'restoreViewButton.addEventListener("click", () => setPresentationMode(false));' in javascript
    assert 'closeExpandedViewButton.addEventListener("click", () => setPresentationMode(false));' in javascript
    assert "Question embedded" in template
    assert "Top-ranked chunks retrieved as a set" in template
    assert "Retrieved chunks added to the prompt" in template
    assert "Model generates the answer" in template


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
    assert 'label.textContent = dimension.toUpperCase();' in javascript
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
    assert "togglePinnedChunk(point.id);" in javascript
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
    assert "visualizationInspectorTitle.textContent = \"Query\";" in javascript
    assert "preview.textContent = previewText;" in javascript
    assert "source.textContent = point.source" in javascript
    assert "appendCoordinateReadout(visualizationInspectorContent" in javascript
    assert 'visualizationInspectorClose.addEventListener("click", clearPinnedSelection);' in javascript
    assert "visualizationInspector.style" not in javascript
    inspector_rule = stylesheet[
        stylesheet.index(".embedding-map-inspector {"):
        stylesheet.index(".embedding-inspector-header")
    ]
    assert "position: absolute;" in inspector_rule
    assert "right: 0.9rem;" in inspector_rule
    assert "bottom: 0.9rem;" in inspector_rule
    assert '["X", "Y", "Z"].forEach((label) =>' in javascript
    assert "heading.textContent = point.retrieved" in javascript
    assert "nativeTooltip.textContent" in javascript
    assert "point.vector" not in javascript
    assert "visualization.vector" not in javascript
    assert "innerHTML" not in javascript


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
    assert 'svg.addEventListener("pointerup", endDrag);' in javascript
    assert 'svg.addEventListener("pointercancel", endDrag);' in javascript
    assert 'svg.addEventListener("lostpointercapture", endDrag);' in javascript
    assert "svg.setPointerCapture(event.pointerId);" in javascript
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
    assert 'type="button" disabled>Refresh examples' in template


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
    fetch: async url => {
      calls.push(url);
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
  assert(!calls.includes("./api/query"));
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
    this.capturedPointers = new Set();
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(""); }
  appendChild(child) {
    if (child.parent) child.parent.children = child.parent.children.filter(candidate => candidate !== child);
    this.children.push(child); child.parent = this; return child;
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
    querySelector(selector) {
      if (!elements.has(selector)) {
        const element = new Element("div", document);
        element.selector = selector;
        elements.set(selector, element);
      }
      return elements.get(selector);
    },
    querySelectorAll(selector) {
      return selector === "input[name='retrieval_mode']" ? retrievalInputs : [];
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
  const semanticResponse = {
    answer: "Safe answer",
    search: {mode: "semantic", label: "Search", technical_label: "Dense", score_label: "Score", results: [
      {id: "one", rank: 1, source: "one.pdf", score: 0.9, content: "First safe chunk", matched_terms: []},
      {id: "two", rank: 2, source: "two.pdf", score: 0.8, content: "Second safe chunk", matched_terms: []},
    ]},
    visualization: {status: "ready", label: "Explore the embedding space", technical_label: "3D PCA", note: "This is a 3D PCA projection. Retrieval itself uses the full-dimensional dense embedding space.", points: [
      {id: "one", source: "one.pdf", x: -2, y: 1, z: 0.5, retrieved: true, rank: 1, preview: "First safe chunk"},
      {id: "two", source: "two.pdf", x: 1, y: 2, z: -1, retrieved: true, rank: 2, preview: "Second safe chunk"},
      {id: "other", source: "other.pdf", x: 3, y: -1, z: 2, retrieved: false, rank: null, preview: "Background chunk"},
    ], query: {x: 0.75, y: -0.5, z: 1.25}},
    prompt: {label: "Prompt", technical_label: "Messages", note: "Safe", messages: [
      {role: "user", content: "Question"},
    ]},
  };
  const keywordResponse = {
    ...semanticResponse,
    search: {...semanticResponse.search, mode: "keyword"},
    visualization: {status: "unsupported", label: "", technical_label: "", note: "", points: [], query: null},
  };
  const context = {
    document,
    fetch: async (url, options = {}) => {
      if (url === "./api/examples") return {ok: true, json: async () => ({display_count: 1, question_sets: [{questions: ["Example?"]}]})};
      if (url === "./api/index") return {ok: true, json: async () => ({document_count: 0, chunk_count: 0, truncated: false, documents: []})};
      if (url === "./api/query") {
        const request = JSON.parse(options.body);
        return {ok: true, json: async () => request.retrieval_mode === "keyword" ? keywordResponse : semanticResponse};
      }
      throw new Error(`Unexpected request: ${url}`);
    },
  };
  vm.runInNewContext(fs.readFileSync(process.argv[1], "utf8"), context);
  await new Promise(resolve => setImmediate(resolve));

  const question = document.querySelector("#question-input");
  question.value = "Where is the evidence?";
  await document.querySelector("#query-form").dispatch("submit");
  const canvas = document.querySelector("#visualization-canvas");
  const svg = canvas.children.find(child => child.tag === "svg");
  assert(svg);
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
  await query.dispatch("click");
  const inspector = document.querySelector("#visualization-inspector");
  const inspectorTitle = document.querySelector("#visualization-inspector-title");
  const inspectorContent = document.querySelector("#visualization-inspector-content");
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
  await retrieved.dispatch("click");
  assert(retrieved.classList.contains("is-pinned"));
  assert.equal(inspector.hidden, false);
  assert.equal(inspectorTitle.textContent, "#1");
  assert.match(inspectorContent.textContent, /one\.pdf/);
  assert.match(inspectorContent.textContent, /First safe chunk/);
  assert.match(inspectorContent.textContent, /X-2\.0/);
  assert.deepEqual(
    byClass(svg, "query-retrieval-connection").filter(line => line.classList.contains("is-active")).map(line => line.dataset.chunkId),
    ["one"],
  );
  await retrieved.dispatch("mouseleave");

  await svg.dispatch("pointerdown", {button: 0, pointerId: 7, clientX: 480, clientY: 300});
  await svg.dispatch("pointermove", {pointerId: 7, clientX: 690, clientY: 130});
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

  assert.equal(svg.eventOptions.wheel.length, 1);
  assert.equal(svg.eventOptions.wheel[0].passive, false);
  const wheelIn = {deltaY: -100};
  await svg.dispatch("wheel", wheelIn);
  assert.equal(wheelIn.defaultPrevented, true);
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
  const backgroundGeometry = geometry(backgroundPoint);
  await backgroundPoint.dispatch("click");
  assert.equal(inspectorTitle.textContent, "Indexed chunk");
  assert.match(inspectorContent.textContent, /other\.pdf/);
  assert.match(inspectorContent.textContent, /Background chunk/);
  assert.equal(geometry(backgroundPoint).transform, backgroundGeometry.transform);
  assert(byClass(svg, "query-retrieval-connection").every(line => !line.classList.contains("is-active")));
  await document.querySelector("#visualization-inspector-close").click();

  semanticResponse.visualization.status = "partial";
  semanticResponse.visualization.note = "This is a 3D PCA projection. Retrieval itself uses the full-dimensional dense embedding space. The bounded display shows a partial index snapshot.";
  await document.querySelector("#query-form").dispatch("submit");
  assert.equal(document.querySelector("#visualization-note").hidden, false);
  assert.equal(document.querySelector("#visualization-note").textContent, "The bounded display shows a partial index snapshot.");

  semantic.checked = false; keyword.checked = true;
  await document.querySelector("#query-form").dispatch("submit");
  assert.equal(document.querySelector("#visualization-section").hidden, true);
  assert.equal(document.querySelector("#visualization-panel").hidden, true);
})().catch(error => { console.error(error); process.exitCode = 1; });
'''
    result = subprocess.run(
        ["node", "-e", harness, str(javascript_path)], capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
