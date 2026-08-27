(() => {
  "use strict";

  const tabs = document.querySelectorAll("[role='tab']");
  const statusBox = document.querySelector("#index-status");
  const content = document.querySelector("#index-content");
  const documentCount = document.querySelector("#document-count");
  const chunkCount = document.querySelector("#chunk-count");
  const documentList = document.querySelector("#document-list");
  const partialNote = document.querySelector("#partial-note");
  const refreshButton = document.querySelector("#refresh-index");
  const queryForm = document.querySelector("#query-form");
  const questionInput = document.querySelector("#question-input");
  const retrievalModeInputs = document.querySelectorAll("input[name='retrieval_mode']");
  const askButton = document.querySelector("#ask-button");
  const queryError = document.querySelector("#query-error");
  const queryLoading = document.querySelector("#query-loading");
  const retrievalChangeNote = document.querySelector("#retrieval-change-note");
  const queryResults = document.querySelector("#query-results");
  const comparisonActions = document.querySelector("#comparison-actions");
  const compareRetrievalButton = document.querySelector("#compare-retrieval");
  const answerKicker = document.querySelector("#answer-kicker");
  const answerHeading = document.querySelector("#answer-heading");
  const answerContent = document.querySelector("#answer-content");
  const searchLabel = document.querySelector("#search-label");
  const searchTechnicalLabel = document.querySelector("#search-technical-label");
  const scoreLabel = document.querySelector("#score-label");
  const searchResults = document.querySelector("#search-results");
  const visualizationLabel = document.querySelector("#visualization-label");
  const visualizationTechnicalLabel = document.querySelector("#visualization-technical-label");
  const visualizationNote = document.querySelector("#visualization-note");
  const visualizationStatus = document.querySelector("#visualization-status");
  const visualizationSection = document.querySelector("#visualization-section");
  const visualizationPanel = document.querySelector("#visualization-panel");
  const visualizationCanvas = document.querySelector("#visualization-canvas");
  const visualizationTooltip = document.querySelector("#visualization-tooltip");
  const zoomOutButton = document.querySelector("#visualization-zoom-out");
  const zoomLevel = document.querySelector("#visualization-zoom-level");
  const zoomInButton = document.querySelector("#visualization-zoom-in");
  const resetViewButton = document.querySelector("#visualization-reset");
  const promptLabel = document.querySelector("#prompt-label");
  const promptTechnicalLabel = document.querySelector("#prompt-technical-label");
  const promptNote = document.querySelector("#prompt-note");
  const promptMessages = document.querySelector("#prompt-messages");
  const exampleButtons = document.querySelectorAll(".example-question");
  const searchCardByChunkId = new Map();
  const searchResultByChunkId = new Map();
  const visualizationPointByChunkId = new Map();
  const svgNamespace = "http://www.w3.org/2000/svg";
  const defaultCamera = Object.freeze({ rotationX: -0.38, rotationY: 0.58, zoom: 1 });
  const minimumZoom = 0.55;
  const maximumZoom = 2.6;
  const plotPadding = 28;
  // Includes the Query diamond, its stroke, and the maximum depth emphasis.
  const maximumMarkerExtent = 30;
  let viewerState = null;
  let hoveredChunkId = null;
  let focusedChunkId = null;
  let pinnedChunkId = null;

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((candidate) => {
        const selected = candidate === tab;
        candidate.classList.toggle("active", selected);
        candidate.setAttribute("aria-selected", String(selected));
        document.querySelector(`#${candidate.dataset.panel}`).hidden = !selected;
      });
    });
  });

  function showState(message, isError = false) {
    statusBox.textContent = message;
    statusBox.className = `alert ${isError ? "alert-danger" : "alert-primary"} mt-4 mb-0`;
    statusBox.hidden = false;
    content.hidden = true;
  }

  function renderDocuments(data) {
    documentCount.textContent = String(data.document_count);
    chunkCount.textContent = String(data.chunk_count);
    partialNote.hidden = !data.truncated;
    documentList.replaceChildren();

    if (data.documents.length === 0) {
      const emptyColumn = document.createElement("div");
      emptyColumn.className = "col-12";
      const emptyState = document.createElement("p");
      emptyState.className = "alert alert-secondary mb-0 text-center";
      emptyState.textContent = "The index is currently empty. Documents will appear here after they are processed.";
      emptyColumn.appendChild(emptyState);
      documentList.appendChild(emptyColumn);
    }

    data.documents.forEach((indexedDocument) => {
      const column = document.createElement("div");
      column.className = "col-12 col-lg-6";
      const card = document.createElement("article");
      card.className = "card h-100 shadow-sm document-card";
      const cardBody = document.createElement("div");
      cardBody.className = "card-body";

      const heading = document.createElement("h3");
      heading.className = "card-title h5 mb-3";
      heading.textContent = indexedDocument.name;
      cardBody.appendChild(heading);

      const meta = document.createElement("p");
      meta.className = "d-flex flex-wrap gap-2 mb-0";
      const chunks = document.createElement("span");
      chunks.className = "badge text-bg-light border text-secondary";
      chunks.textContent = `${indexedDocument.chunk_count} ${indexedDocument.chunk_count === 1 ? "chunk" : "chunks"}`;
      meta.appendChild(chunks);

      if (indexedDocument.source_type) {
        const sourceType = document.createElement("span");
        sourceType.className = "badge text-bg-primary";
        sourceType.textContent = indexedDocument.source_type;
        meta.appendChild(sourceType);
      }
      cardBody.appendChild(meta);
      card.appendChild(cardBody);
      column.appendChild(card);
      documentList.appendChild(column);
    });

    statusBox.hidden = true;
    content.hidden = false;
  }

  async function loadIndex() {
    showState("Loading indexed documents…");
    refreshButton.disabled = true;
    try {
      const response = await fetch("./api/index", { headers: { Accept: "application/json" } });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.message || "The index could not be loaded.");
      }
      renderDocuments(data);
    } catch (error) {
      showState(error instanceof Error ? error.message : "The index could not be loaded.", true);
    } finally {
      refreshButton.disabled = false;
    }
  }

  function setQueryLoading(isLoading) {
    queryForm.setAttribute("aria-busy", String(isLoading));
    questionInput.disabled = isLoading;
    retrievalModeInputs.forEach((input) => { input.disabled = isLoading; });
    askButton.disabled = isLoading;
    exampleButtons.forEach((button) => { button.disabled = isLoading; });
    askButton.textContent = isLoading ? "Asking…" : "Ask RAG";
    queryLoading.hidden = !isLoading;
  }

  function showQueryError(message) {
    queryError.textContent = message;
    queryError.hidden = false;
    queryResults.hidden = true;
  }

  function clearStaleQueryResults(showChangeNote = true) {
    if (queryResults.hidden) {
      return;
    }
    // A selector describes the next request, so the previous mode's answer,
    // labels, and explorer must not remain presented beneath it.
    queryResults.hidden = true;
    comparisonActions.hidden = true;
    retrievalChangeNote.textContent = showChangeNote
      ? "Retrieval method changed. Ask again to compare the same question."
      : "";
    retrievalChangeNote.hidden = !showChangeNote;
  }

  function configureComparisonShortcut(mode) {
    const targetMode = mode === "semantic" ? "keyword" : "semantic";
    const targetLabel = targetMode === "keyword" ? "Keyword search" : "Semantic vectors";
    compareRetrievalButton.dataset.retrievalMode = targetMode;
    compareRetrievalButton.textContent = `Try same question with ${targetLabel}`;
    comparisonActions.hidden = false;
  }

  function appendKeywordHighlights(container, content, matchedTerms) {
    // Search content stays text-only: matched ranges become explicit <mark>
    // nodes, never an HTML string assembled from indexed or question text.
    const normalizedTerms = new Set(
      matchedTerms.map((term) => String(term).toLocaleLowerCase()),
    );
    const tokenPattern = /[\p{L}\p{N}]+/gu;
    let cursor = 0;
    for (const match of content.matchAll(tokenPattern)) {
      const start = match.index;
      const token = match[0];
      container.appendChild(document.createTextNode(content.slice(cursor, start)));
      if (normalizedTerms.has(token.toLocaleLowerCase())) {
        const highlight = document.createElement("mark");
        highlight.textContent = token;
        container.appendChild(highlight);
      } else {
        container.appendChild(document.createTextNode(token));
      }
      cursor = start + token.length;
    }
    container.appendChild(document.createTextNode(content.slice(cursor)));
  }

  function renderSearchResults(results, mode) {
    searchResults.replaceChildren();
    searchCardByChunkId.clear();
    searchResultByChunkId.clear();
    if (results.length === 0) {
      const emptyState = document.createElement("p");
      emptyState.className = "alert alert-secondary mb-0";
      emptyState.textContent = mode === "keyword"
        ? "No matching keyword chunks were found."
        : "No matching chunks were retrieved.";
      searchResults.appendChild(emptyState);
      return;
    }

    results.forEach((result) => {
      const card = document.createElement("article");
      card.className = "card retrieval-card";
      // The shared opaque ID is the only bridge between authoritative Search
      // results and the supplemental SVG projection.
      card.dataset.chunkId = result.id;
      card.tabIndex = 0;
      searchCardByChunkId.set(result.id, card);
      searchResultByChunkId.set(result.id, result);
      card.addEventListener("mouseenter", () => setHoveredChunk(result.id));
      card.addEventListener("mouseleave", () => clearHoveredChunk(result.id));
      card.addEventListener("focus", () => setFocusedChunk(result.id));
      card.addEventListener("blur", () => clearFocusedChunk(result.id));

      const header = document.createElement("div");
      header.className = "card-header d-flex flex-column flex-md-row align-items-md-center justify-content-between gap-2";
      const sourceGroup = document.createElement("div");
      sourceGroup.className = "d-flex align-items-center gap-2";
      const rank = document.createElement("span");
      rank.className = "badge text-bg-primary";
      rank.textContent = `#${result.rank}`;
      const source = document.createElement("h3");
      source.className = "h6 mb-0";
      source.textContent = result.source;
      sourceGroup.append(rank, source);

      const score = document.createElement("span");
      score.className = "badge text-bg-light border text-secondary retrieval-score";
      score.textContent = `Score: ${Number(result.score).toFixed(4)}`;
      header.append(sourceGroup, score);

      const body = document.createElement("div");
      body.className = "card-body";
      const chunk = document.createElement("p");
      chunk.className = "chunk-content mb-0";
      if (mode === "keyword" && result.matched_terms.length > 0) {
        appendKeywordHighlights(chunk, result.content, result.matched_terms);
      } else {
        // Semantic similarity does not imply literal term matches.
        chunk.textContent = result.content;
      }
      body.appendChild(chunk);

      card.append(header, body);
      searchResults.appendChild(card);
    });
  }

  function activeChunkId() {
    return hoveredChunkId || focusedChunkId || pinnedChunkId;
  }

  function setHoveredChunk(chunkId) {
    hoveredChunkId = chunkId;
    updateLinkedInteraction();
  }

  function clearHoveredChunk(chunkId) {
    if (hoveredChunkId === chunkId) {
      hoveredChunkId = null;
      updateLinkedInteraction();
    }
  }

  function setFocusedChunk(chunkId) {
    focusedChunkId = chunkId;
    updateLinkedInteraction();
  }

  function clearFocusedChunk(chunkId) {
    if (focusedChunkId === chunkId) {
      focusedChunkId = null;
      updateLinkedInteraction();
    }
  }

  function togglePinnedChunk(chunkId) {
    pinnedChunkId = pinnedChunkId === chunkId ? null : chunkId;
    updateLinkedInteraction();
  }

  function updateLinkedInteraction(renderScene = true) {
    const activeId = activeChunkId();
    searchCardByChunkId.forEach((card, chunkId) => {
      card.classList.toggle("is-linked", chunkId === activeId);
      card.classList.toggle("is-pinned", chunkId === pinnedChunkId);
    });
    visualizationPointByChunkId.forEach((point, chunkId) => {
      point.classList.toggle("is-linked", chunkId === activeId);
      point.classList.toggle("is-pinned", chunkId === pinnedChunkId);
      point.setAttribute("aria-pressed", String(chunkId === pinnedChunkId));
    });
    if (viewerState && renderScene) {
      renderViewerScene();
    }
  }

  function svgElement(name, className) {
    const element = document.createElementNS(svgNamespace, name);
    if (className) {
      element.setAttribute("class", className);
    }
    return element;
  }

  function clamp(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, value));
  }

  function normalizeWorldScene(coordinates) {
    const center = coordinates.reduce(
      (sum, point) => ({
        x: sum.x + Number(point.x),
        y: sum.y + Number(point.y),
        z: sum.z + Number(point.z),
      }),
      { x: 0, y: 0, z: 0 },
    );
    center.x /= coordinates.length;
    center.y /= coordinates.length;
    center.z /= coordinates.length;

    const centered = coordinates.map((point) => ({
      x: Number(point.x) - center.x,
      y: Number(point.y) - center.y,
      z: Number(point.z) - center.z,
    }));
    const sourceRadius = centered.reduce(
      (maximum, point) => Math.max(maximum, Math.hypot(point.x, point.y, point.z)),
      0,
    );
    const normalizationRadius = sourceRadius || 1;
    return {
      points: centered.map((point) => ({
        x: point.x / normalizationRadius,
        y: point.y / normalizationRadius,
        z: point.z / normalizationRadius,
      })),
      radius: sourceRadius > 0 ? 1 : 0,
    };
  }

  function computeFittedSceneScale(plotWidth, plotHeight) {
    // Orthographic positioning makes the normalized bounding sphere a strict
    // rotation-invariant fit; marker space is reserved inside the same SVG rect.
    const edgePadding = plotPadding + maximumMarkerExtent;
    const safeWidth = Math.max(2, plotWidth - 2 * edgePadding);
    const safeHeight = Math.max(2, plotHeight - 2 * edgePadding);
    return Math.min(safeWidth, safeHeight) / 2;
  }

  function safePointPreview(point) {
    const searchResult = searchResultByChunkId.get(point.id);
    const rawPreview = point.preview || (searchResult ? searchResult.content : "");
    const preview = String(rawPreview || "").replace(/\s+/g, " ").trim();
    return preview.length > 160 ? `${preview.slice(0, 157)}...` : preview;
  }

  function projectWorldPoint(world, state) {
    const cosX = Math.cos(state.rotationX);
    const sinX = Math.sin(state.rotationX);
    const cosY = Math.cos(state.rotationY);
    const sinY = Math.sin(state.rotationY);
    const rotatedY = world.y * cosX - world.z * sinX;
    const rotatedZAfterX = world.y * sinX + world.z * cosX;
    const rotatedX = world.x * cosY + rotatedZAfterX * sinY;
    const rotatedZ = -world.x * sinY + rotatedZAfterX * cosY;
    const sceneScale = state.fittedSceneScale * state.zoom;
    return {
      x: state.plotBounds.x + state.plotBounds.width / 2 + rotatedX * sceneScale,
      y: state.plotBounds.y + state.plotBounds.height / 2 - rotatedY * sceneScale,
      depth: rotatedZ,
      depthRatio: clamp((rotatedZ + 1.25) / 2.5, 0, 1),
    };
  }

  function positionPointTooltip(projected, state) {
    const panelBounds = visualizationCanvas.getBoundingClientRect();
    const svgBounds = state.svg.getBoundingClientRect();
    const svgScale = Math.min(
      svgBounds.width / state.width,
      svgBounds.height / state.height,
    );
    const svgOffsetX = svgBounds.left - panelBounds.left
      + (svgBounds.width - state.width * svgScale) / 2;
    const svgOffsetY = svgBounds.top - panelBounds.top
      + (svgBounds.height - state.height * svgScale) / 2;
    const anchorX = svgOffsetX + projected.x * svgScale;
    const anchorY = svgOffsetY + projected.y * svgScale;
    const tooltipBounds = visualizationTooltip.getBoundingClientRect();
    const overlayMargin = 12;
    const pointGap = 16;
    const maximumLeft = Math.max(
      overlayMargin,
      panelBounds.width - tooltipBounds.width - overlayMargin,
    );
    const maximumTop = Math.max(
      overlayMargin,
      panelBounds.height - tooltipBounds.height - overlayMargin,
    );
    const tooltipLeft = clamp(
      anchorX - tooltipBounds.width / 2,
      overlayMargin,
      maximumLeft,
    );
    const topAbovePoint = anchorY - tooltipBounds.height - pointGap;
    const preferredTop = topAbovePoint >= overlayMargin
      ? topAbovePoint
      : anchorY + pointGap;
    const tooltipTop = clamp(preferredTop, overlayMargin, maximumTop);
    visualizationTooltip.style.left = `${tooltipLeft}px`;
    visualizationTooltip.style.top = `${tooltipTop}px`;
  }

  function renderPointTooltip(point, projected, state) {
    visualizationTooltip.replaceChildren();
    const heading = document.createElement("strong");
    heading.className = "d-block";
    heading.textContent = `#${point.rank} · ${point.source}`;
    visualizationTooltip.appendChild(heading);
    const preview = safePointPreview(point);
    if (preview) {
      const content = document.createElement("span");
      content.className = "d-block mt-1";
      content.textContent = preview;
      visualizationTooltip.appendChild(content);
    }
    visualizationTooltip.hidden = false;
    // Overlay constraints are independent of the projected point geometry.
    positionPointTooltip(projected, state);
  }

  function clearPointTooltip() {
    visualizationTooltip.hidden = true;
    visualizationTooltip.replaceChildren();
    visualizationTooltip.style.removeProperty("left");
    visualizationTooltip.style.removeProperty("top");
  }

  function hideConnectionLine(line) {
    // SVG visibility is explicit because HTMLElement.hidden is not reliable on
    // SVG geometry; removing endpoints also prevents stale lines from returning.
    line.setAttribute("visibility", "hidden");
    ["x1", "y1", "x2", "y2"].forEach((attribute) => line.removeAttribute(attribute));
  }

  function showConnectionLine(line, queryPoint, chunkPoint) {
    line.setAttribute("x1", String(queryPoint.x));
    line.setAttribute("y1", String(queryPoint.y));
    line.setAttribute("x2", String(chunkPoint.x));
    line.setAttribute("y2", String(chunkPoint.y));
    line.setAttribute("visibility", "visible");
  }

  function renderViewerScene() {
    const state = viewerState;
    if (!state) {
      return;
    }

    const projectedQuery = projectWorldPoint(state.query.world, state);
    const projectedPoints = state.points.map((entry) => ({
      entry,
      projected: projectWorldPoint(entry.world, state),
    }));

    // SVG has no depth buffer, so far-to-near DOM ordering supplies the small
    // point cloud with a readable and dependency-free depth cue.
    projectedPoints.sort((first, second) => first.projected.depth - second.projected.depth);
    projectedPoints.forEach(({ entry, projected }) => {
      const depthScale = 0.82 + projected.depthRatio * 0.34;
      entry.visual.group.setAttribute(
        "transform",
        `translate(${projected.x} ${projected.y}) scale(${depthScale})`,
      );
      entry.visual.group.style.opacity = String(
        entry.data.retrieved
          ? 0.84 + projected.depthRatio * 0.16
          : 0.34 + projected.depthRatio * 0.46,
      );
      state.pointLayer.appendChild(entry.visual.group);
    });

    const queryScale = 0.9 + projectedQuery.depthRatio * 0.25;
    state.query.visual.group.setAttribute(
      "transform",
      `translate(${projectedQuery.x} ${projectedQuery.y}) scale(${queryScale})`,
    );

    const activeId = activeChunkId();
    const activePoint = projectedPoints.find(
      ({ entry }) => entry.data.retrieved && entry.data.id === activeId,
    );
    if (activePoint) {
      // This line is a presentation cue between two projected points, not a
      // retrieval vector, distance, angle, or replacement for the real score.
      showConnectionLine(state.connectionLine, projectedQuery, activePoint.projected);
      renderPointTooltip(activePoint.entry.data, activePoint.projected, state);
    } else {
      hideConnectionLine(state.connectionLine);
      clearPointTooltip();
    }
  }

  function installCameraInteraction(state) {
    const { svg } = state;
    svg.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) {
        return;
      }
      state.pointerId = event.pointerId;
      state.lastPointerX = event.clientX;
      state.lastPointerY = event.clientY;
      state.dragMoved = false;
      svg.setPointerCapture(event.pointerId);
      svg.classList.add("is-dragging");
      event.preventDefault();
    });
    svg.addEventListener("pointermove", (event) => {
      if (state.pointerId !== event.pointerId) {
        return;
      }
      const deltaX = event.clientX - state.lastPointerX;
      const deltaY = event.clientY - state.lastPointerY;
      if (Math.abs(deltaX) + Math.abs(deltaY) > 1) {
        state.dragMoved = true;
      }
      state.rotationY += deltaX * 0.008;
      state.rotationX = clamp(state.rotationX - deltaY * 0.008, -1.45, 1.45);
      state.lastPointerX = event.clientX;
      state.lastPointerY = event.clientY;
      renderViewerScene();
      event.preventDefault();
    });

    const endDrag = (event) => {
      if (state.pointerId !== event.pointerId) {
        return;
      }
      if (svg.hasPointerCapture(event.pointerId)) {
        svg.releasePointerCapture(event.pointerId);
      }
      state.pointerId = null;
      svg.classList.remove("is-dragging");
    };
    svg.addEventListener("pointerup", endDrag);
    svg.addEventListener("pointercancel", endDrag);
    svg.addEventListener("click", () => {
      if (!state.dragMoved) {
        const activeElement = document.activeElement;
        if (activeElement && svg.contains(activeElement) && typeof activeElement.blur === "function") {
          activeElement.blur();
        }
        focusedChunkId = null;
        pinnedChunkId = null;
        updateLinkedInteraction();
      }
      state.dragMoved = false;
    });
  }

  function updateZoomControls(state) {
    const percentage = Math.round(state.zoom * 100);
    zoomLevel.textContent = `${percentage}%`;
    zoomLevel.setAttribute("aria-label", `Current embedding space zoom: ${percentage}%`);
    zoomOutButton.disabled = state.zoom <= minimumZoom;
    zoomInButton.disabled = state.zoom >= maximumZoom;
  }

  function zoomViewer(factor) {
    if (!viewerState) {
      return;
    }
    viewerState.zoom = clamp(viewerState.zoom * factor, minimumZoom, maximumZoom);
    updateZoomControls(viewerState);
    renderViewerScene();
  }

  function resetPointerInteraction(state) {
    const capturedPointerId = state.pointerId;
    if (capturedPointerId !== null) {
      try {
        if (state.svg.hasPointerCapture(capturedPointerId)) {
          state.svg.releasePointerCapture(capturedPointerId);
        }
      } catch {
        // Pointer capture may already have ended between the event and reset.
      }
    }
    state.pointerId = null;
    state.lastPointerX = 0;
    state.lastPointerY = 0;
    state.dragMoved = false;
    state.svg.classList.remove("is-dragging");
  }

  function resetViewer() {
    if (!viewerState) {
      return;
    }
    const state = viewerState;
    hoveredChunkId = null;
    focusedChunkId = null;
    pinnedChunkId = null;
    const activeElement = document.activeElement;
    if (activeElement && state.svg.contains(activeElement) && typeof activeElement.blur === "function") {
      activeElement.blur();
    }
    resetPointerInteraction(state);
    state.rotationX = state.initialCamera.rotationX;
    state.rotationY = state.initialCamera.rotationY;
    state.zoom = state.initialCamera.zoom;
    updateZoomControls(state);
    hideConnectionLine(state.connectionLine);
    clearPointTooltip();
    updateLinkedInteraction(false);
    renderViewerScene();
  }

  function renderVisualization(visualization, searchMode) {
    visualizationLabel.textContent = visualization.label;
    visualizationTechnicalLabel.textContent = visualization.technical_label;
    visualizationNote.textContent = visualization.note;
    visualizationCanvas.replaceChildren(visualizationTooltip);
    visualizationTooltip.hidden = true;
    visualizationTooltip.replaceChildren();
    visualizationPointByChunkId.clear();
    viewerState = null;
    hoveredChunkId = null;
    focusedChunkId = null;
    pinnedChunkId = null;

    const isSemantic = searchMode === "semantic";
    visualizationSection.hidden = !isSemantic;
    if (!isSemantic) {
      visualizationNote.hidden = true;
      visualizationStatus.hidden = true;
      visualizationPanel.hidden = true;
      return;
    }

    const allCoordinates = visualization.query
      ? [...visualization.points, visualization.query]
      : [];
    const coordinatesAreFinite = allCoordinates.every((point) =>
      [point.x, point.y, point.z].every((value) => Number.isFinite(Number(value))));
    const canRender = (visualization.status === "ready" || visualization.status === "partial")
      && visualization.query
      && visualization.points.length > 0
      && coordinatesAreFinite;
    visualizationNote.hidden = !canRender;
    visualizationStatus.hidden = canRender;
    visualizationPanel.hidden = !canRender;
    if (!canRender) {
      visualizationStatus.textContent = visualization.note;
      return;
    }

    const width = 960;
    const height = 560;
    const plotInset = 1;
    const plotBounds = Object.freeze({
      x: plotInset,
      y: plotInset,
      width: width - plotInset * 2,
      height: height - plotInset * 2,
    });
    const normalizedScene = normalizeWorldScene(allCoordinates);
    const fittedSceneScale = computeFittedSceneScale(plotBounds.width, plotBounds.height);
    const svg = svgElement("svg", "embedding-map");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("role", "group");
    svg.setAttribute("aria-label", "Interactive three-dimensional projection of indexed chunks and the Query");

    const background = svgElement("rect", "embedding-map-background");
    background.setAttribute("x", String(plotBounds.x));
    background.setAttribute("y", String(plotBounds.y));
    background.setAttribute("width", String(plotBounds.width));
    background.setAttribute("height", String(plotBounds.height));
    background.setAttribute("rx", "12");
    svg.appendChild(background);

    const connectionLine = svgElement("line", "query-connection");
    hideConnectionLine(connectionLine);
    svg.appendChild(connectionLine);
    const pointLayer = svgElement("g", "embedding-point-layer");
    const queryLayer = svgElement("g", "embedding-query-layer");
    svg.append(pointLayer, queryLayer);

    const initialCamera = Object.freeze({
      rotationX: defaultCamera.rotationX,
      rotationY: defaultCamera.rotationY,
      zoom: defaultCamera.zoom,
    });
    const state = {
      width,
      height,
      svg,
      pointLayer,
      connectionLine,
      plotBounds,
      points: [],
      query: null,
      initialCamera,
      fittedSceneScale,
      rotationX: initialCamera.rotationX,
      rotationY: initialCamera.rotationY,
      zoom: initialCamera.zoom,
      pointerId: null,
      lastPointerX: 0,
      lastPointerY: 0,
      dragMoved: false,
    };

    visualization.points.forEach((point, index) => {
      const group = svgElement(
        "g",
        point.retrieved ? "embedding-point retrieved-point" : "embedding-point indexed-point",
      );
      group.dataset.chunkId = point.id;
      const marker = svgElement("circle", "embedding-point-marker");
      marker.setAttribute("r", point.retrieved ? "16" : "5.5");
      group.appendChild(marker);
      const nativeTooltip = svgElement("title", "");
      nativeTooltip.textContent = point.retrieved
        ? `Retrieved rank ${point.rank}: ${point.source}`
        : `Indexed chunk: ${point.source}`;
      group.appendChild(nativeTooltip);

      if (point.retrieved) {
        group.setAttribute("tabindex", "0");
        group.setAttribute("role", "button");
        group.setAttribute("aria-label", `Retrieved rank ${point.rank}, ${point.source}. Press Enter to pin this point.`);
        group.setAttribute("aria-pressed", "false");
        const rank = svgElement("text", "embedding-rank");
        rank.setAttribute("text-anchor", "middle");
        rank.setAttribute("dominant-baseline", "central");
        rank.textContent = String(point.rank);
        group.appendChild(rank);
        visualizationPointByChunkId.set(point.id, group);
        group.addEventListener("mouseenter", () => setHoveredChunk(point.id));
        group.addEventListener("mouseleave", () => clearHoveredChunk(point.id));
        group.addEventListener("focus", () => setFocusedChunk(point.id));
        group.addEventListener("blur", () => clearFocusedChunk(point.id));
        group.addEventListener("click", (event) => {
          event.stopPropagation();
          if (!state.dragMoved) {
            togglePinnedChunk(point.id);
          }
          state.dragMoved = false;
        });
        group.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            togglePinnedChunk(point.id);
          }
        });
      }
      const entry = {
        data: point,
        world: normalizedScene.points[index],
        visual: { group },
      };
      state.points.push(entry);
      pointLayer.appendChild(group);
    });

    const queryGroup = svgElement("g", "embedding-query-point");
    queryGroup.setAttribute("role", "img");
    queryGroup.setAttribute("aria-label", "Query");
    const queryMarker = svgElement("polygon", "embedding-query-marker");
    queryMarker.setAttribute("points", "0,-21 21,0 0,21 -21,0");
    const queryLabel = svgElement("text", "embedding-query-label");
    queryLabel.setAttribute("text-anchor", "middle");
    queryLabel.setAttribute("dominant-baseline", "central");
    queryLabel.textContent = "Q";
    const queryTooltip = svgElement("title", "");
    queryTooltip.textContent = "Query";
    queryGroup.append(queryMarker, queryLabel, queryTooltip);
    queryLayer.appendChild(queryGroup);
    state.query = {
      data: visualization.query,
      world: normalizedScene.points[normalizedScene.points.length - 1],
      visual: { group: queryGroup },
    };

    // Camera rotation belongs in the browser so visitors can play with the
    // safe three-coordinate projection; no raw embedding or 3D library is needed.
    viewerState = state;
    updateZoomControls(state);
    visualizationCanvas.insertBefore(svg, visualizationTooltip);
    installCameraInteraction(state);
    renderViewerScene();
    updateLinkedInteraction();
  }

  function renderPrompt(prompt) {
    promptLabel.textContent = prompt.label;
    promptTechnicalLabel.textContent = prompt.technical_label;
    promptNote.textContent = prompt.note;
    promptMessages.replaceChildren();

    prompt.messages.forEach((message) => {
      const card = document.createElement("article");
      card.className = "card prompt-message";
      const header = document.createElement("div");
      header.className = "card-header";
      const role = document.createElement("span");
      role.className = "badge text-bg-primary";
      role.textContent = message.role.toUpperCase();
      header.appendChild(role);

      const body = document.createElement("div");
      body.className = "card-body";
      const messageContent = document.createElement("pre");
      messageContent.className = "prompt-message-content mb-0";
      // Model-bound prompt text is untrusted display data, never executable markup.
      messageContent.textContent = message.content;
      body.appendChild(messageContent);

      card.append(header, body);
      promptMessages.appendChild(card);
    });
  }

  function renderQueryResponse(data) {
    const hasEvidence = data.search.results.length > 0;
    answerKicker.textContent = hasEvidence ? "Answer" : "Grounding guard";
    answerHeading.textContent = hasEvidence ? "Generated answer" : "Grounded answer status";
    answerContent.textContent = data.answer;
    searchLabel.textContent = data.search.label;
    searchTechnicalLabel.textContent = data.search.technical_label;
    scoreLabel.textContent = data.search.score_label;
    renderSearchResults(data.search.results, data.search.mode);
    renderVisualization(data.visualization, data.search.mode);
    renderPrompt(data.prompt);
    configureComparisonShortcut(data.search.mode);
    queryError.hidden = true;
    retrievalChangeNote.hidden = true;
    retrievalChangeNote.textContent = "";
    queryResults.hidden = false;
  }

  async function submitQuery(event) {
    event.preventDefault();
    const question = questionInput.value.trim();
    const selectedMode = Array.from(retrievalModeInputs).find((input) => input.checked);
    const retrievalMode = selectedMode ? selectedMode.value : "semantic";
    queryError.hidden = true;
    retrievalChangeNote.hidden = true;
    retrievalChangeNote.textContent = "";
    if (!question) {
      showQueryError("Please enter a question.");
      questionInput.focus();
      return;
    }

    queryResults.hidden = true;
    setQueryLoading(true);
    try {
      const response = await fetch("./api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ question, retrieval_mode: retrievalMode }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.message || "The RAG query could not be completed.");
      }
      renderQueryResponse(data);
    } catch (error) {
      showQueryError(error instanceof Error ? error.message : "The RAG query could not be completed.");
    } finally {
      setQueryLoading(false);
    }
  }

  refreshButton.addEventListener("click", loadIndex);
  queryForm.addEventListener("submit", submitQuery);
  retrievalModeInputs.forEach((input) => {
    input.addEventListener("change", () => clearStaleQueryResults());
  });
  compareRetrievalButton.addEventListener("click", () => {
    const targetMode = compareRetrievalButton.dataset.retrievalMode;
    const targetInput = Array.from(retrievalModeInputs).find(
      (input) => input.value === targetMode,
    );
    if (!targetInput || !questionInput.value.trim()) {
      return;
    }
    targetInput.checked = true;
    clearStaleQueryResults(false);
    queryForm.requestSubmit();
  });
  zoomOutButton.addEventListener("click", () => zoomViewer(1 / 1.12));
  zoomInButton.addEventListener("click", () => zoomViewer(1.12));
  resetViewButton.addEventListener("click", resetViewer);
  exampleButtons.forEach((button) => {
    button.addEventListener("click", () => {
      questionInput.value = button.dataset.question || "";
      questionInput.focus();
    });
  });
  loadIndex();
})();
