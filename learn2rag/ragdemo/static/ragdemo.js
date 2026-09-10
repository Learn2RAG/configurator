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
  const visualizationInspector = document.querySelector("#visualization-inspector");
  const visualizationInspectorTitle = document.querySelector("#visualization-inspector-title");
  const visualizationInspectorContent = document.querySelector("#visualization-inspector-content");
  const visualizationInspectorClose = document.querySelector("#visualization-inspector-close");
  const zoomOutButton = document.querySelector("#visualization-zoom-out");
  const zoomLevel = document.querySelector("#visualization-zoom-level");
  const zoomInButton = document.querySelector("#visualization-zoom-in");
  const resetViewButton = document.querySelector("#visualization-reset");
  const expandViewButton = document.querySelector("#visualization-expand");
  const restoreViewButton = document.querySelector("#visualization-restore");
  const closeExpandedViewButton = document.querySelector("#visualization-expanded-close");
  const promptLabel = document.querySelector("#prompt-label");
  const promptTechnicalLabel = document.querySelector("#prompt-technical-label");
  const promptNote = document.querySelector("#prompt-note");
  const promptMessages = document.querySelector("#prompt-messages");
  const exampleContainer = document.querySelector("#example-questions");
  const examplesStatus = document.querySelector("#examples-status");
  const refreshExamplesButton = document.querySelector("#refresh-examples");
  let displayedExamples = [];
  const documentChunksCache = new Map();
  const searchCardByChunkId = new Map();
  const searchResultByChunkId = new Map();
  const visualizationPointByChunkId = new Map();
  const svgNamespace = "http://www.w3.org/2000/svg";
  const defaultCamera = Object.freeze({
    rotation: Object.freeze(quaternionFromEuler(-0.38, 0.58, 0)),
    zoom: 1,
  });
  const minimumZoom = 0.55;
  const maximumZoom = 2.6;
  const wheelZoomFactor = 1.09;
  const scenePadding = 12;
  const redundantVisualizationNote =
    "This is a 3D PCA projection. Retrieval itself uses the full-dimensional dense embedding space.";
  // Screen-space labels and markers share this edge reserve; it is not added
  // to the rotation-safe world-space radius.
  const maximumMarkerExtent = 30;
  let viewerState = null;
  let hoveredChunkId = null;
  let focusedChunkId = null;
  let pinnedChunkId = null;
  let pinnedQuery = false;

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

  function shuffleExamples(values) {
    const shuffled = [...values];
    for (let index = shuffled.length - 1; index > 0; index -= 1) {
      const other = Math.floor(Math.random() * (index + 1));
      [shuffled[index], shuffled[other]] = [shuffled[other], shuffled[index]];
    }
    return shuffled;
  }

  function selectExamples(config) {
    const sets = shuffleExamples(config.question_sets).map((questionSet) => ({
      questions: shuffleExamples(questionSet.questions),
    }));
    const selected = [];
    sets.forEach((questionSet, setIndex) => {
      if (selected.length < config.display_count && questionSet.questions.length > 0) {
        selected.push({ question: questionSet.questions.shift(), setIndex });
      }
    });
    const remaining = shuffleExamples(sets.flatMap((questionSet, setIndex) => (
      questionSet.questions.map((question) => ({ question, setIndex }))
    )));
    selected.push(...remaining.slice(0, config.display_count - selected.length));

    const selectedQuestions = selected.map((entry) => entry.question);
    const sameSubset = selectedQuestions.length === displayedExamples.length
      && selectedQuestions.every((question) => displayedExamples.includes(question));
    if (sameSubset) {
      const selectedSet = new Set(selectedQuestions);
      const replacement = remaining.find((entry) => !selectedSet.has(entry.question));
      if (replacement) {
        const sameSetIndex = selected.findIndex((entry) => entry.setIndex === replacement.setIndex);
        selected[sameSetIndex >= 0 ? sameSetIndex : selected.length - 1] = replacement;
      }
    }
    return selected.map((entry) => entry.question);
  }

  function renderExamples(selected) {
    displayedExamples = selected;
    exampleContainer.replaceChildren();
    selected.forEach((question) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "btn btn-outline-secondary example-question";
      button.textContent = question;
      button.disabled = questionInput.disabled;
      button.addEventListener("click", () => {
        questionInput.value = question;
        questionInput.focus();
      });
      exampleContainer.appendChild(button);
    });
  }

  async function loadExamples() {
    refreshExamplesButton.disabled = true;
    examplesStatus.textContent = "Loading examples…";
    examplesStatus.hidden = false;
    try {
      const response = await fetch("./api/examples", { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error("Examples unavailable");
      const nextConfig = await response.json();
      const selected = selectExamples(nextConfig);
      renderExamples(selected);
      examplesStatus.textContent = "";
      examplesStatus.hidden = true;
    } catch (error) {
      examplesStatus.textContent = "Example questions are temporarily unavailable.";
      examplesStatus.hidden = false;
    } finally {
      refreshExamplesButton.disabled = false;
    }
  }

  function renderDocumentChunks(container, data) {
    container.replaceChildren();
    const orderNote = document.createElement("p");
    orderNote.className = "small text-body-secondary";
    orderNote.textContent = "Chunk numbers indicate display order, which may differ from the original document order.";
    container.appendChild(orderNote);
    data.chunks.forEach((chunk) => {
      const item = document.createElement("article");
      item.className = "indexed-chunk";
      item.dataset.chunkId = chunk.id;
      const heading = document.createElement("h4");
      heading.className = "h6";
      heading.textContent = `Chunk ${chunk.display_order}`;
      const text = document.createElement("p");
      text.className = "indexed-chunk-content mb-0";
      text.textContent = chunk.content;
      item.append(heading, text);
      if (chunk.truncated) {
        const note = document.createElement("p");
        note.className = "small text-body-secondary mt-2 mb-0";
        note.textContent = "This chunk's text has been shortened for display.";
        item.appendChild(note);
      }
      container.appendChild(item);
    });
    if (data.truncated) {
      const note = document.createElement("p");
      note.className = "small text-body-secondary mb-0";
      note.textContent = "This is a limited preview. Some chunks or text may be omitted.";
      container.appendChild(note);
    }
  }

  function addChunkDisclosure(cardBody, indexedDocument) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "btn btn-sm btn-outline-secondary mt-3";
    button.textContent = "Show chunks";
    button.setAttribute("aria-expanded", "false");
    button.setAttribute("aria-label", `Show chunks for ${indexedDocument.name}`);
    const container = document.createElement("div");
    container.id = `document-chunks-${indexedDocument.id}`;
    container.className = "document-chunks mt-3";
    container.hidden = true;
    container.setAttribute("role", "region");
    container.setAttribute("aria-label", `Chunks from ${indexedDocument.name}`);
    button.setAttribute("aria-controls", container.id);
    let loaded = false;
    let loading = false;
    button.addEventListener("click", async () => {
      if (loading) return;
      const expanding = container.hidden;
      container.hidden = !expanding;
      button.setAttribute("aria-expanded", String(expanding));
      button.textContent = expanding ? "Hide chunks" : "Show chunks";
      button.setAttribute("aria-label", `${button.textContent} for ${indexedDocument.name}`);
      if (!expanding || loaded) return;
      loading = true;
      button.disabled = true;
      container.setAttribute("aria-busy", "true");
      container.textContent = "Loading chunks…";
      try {
        let data = documentChunksCache.get(indexedDocument.id);
        if (!data) {
          const response = await fetch(`./api/index/${encodeURIComponent(indexedDocument.id)}/chunks`, {
            headers: { Accept: "application/json" },
          });
          if (!response.ok) throw new Error("Chunks unavailable");
          data = await response.json();
          documentChunksCache.set(indexedDocument.id, data);
        }
        renderDocumentChunks(container, data);
        loaded = true;
      } catch (error) {
        container.textContent = "These chunks could not be loaded. Hide and show chunks to try again.";
      } finally {
        loading = false;
        button.disabled = false;
        container.setAttribute("aria-busy", "false");
      }
    });
    cardBody.append(button, container);
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
      addChunkDisclosure(cardBody, indexedDocument);
      card.appendChild(cardBody);
      column.appendChild(card);
      documentList.appendChild(column);
    });

    statusBox.hidden = true;
    content.hidden = false;
  }

  async function loadIndex(clearChunkCache = false) {
    if (clearChunkCache) {
      documentChunksCache.clear();
    }
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
    exampleContainer.querySelectorAll(".example-question").forEach((button) => { button.disabled = isLoading; });
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
    return pinnedChunkId || focusedChunkId || hoveredChunkId;
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
    pinnedQuery = false;
    updateLinkedInteraction();
  }

  function togglePinnedQuery() {
    pinnedQuery = !pinnedQuery;
    pinnedChunkId = null;
    updateLinkedInteraction();
  }

  function clearPinnedSelection() {
    pinnedChunkId = null;
    pinnedQuery = false;
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
    renderPinnedInspector();
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

  function quaternionNormalize(value) {
    const length = Math.hypot(value.x, value.y, value.z, value.w) || 1;
    return {
      x: value.x / length,
      y: value.y / length,
      z: value.z / length,
      w: value.w / length,
    };
  }

  function quaternionMultiply(first, second) {
    return quaternionNormalize({
      x: first.w * second.x + first.x * second.w + first.y * second.z - first.z * second.y,
      y: first.w * second.y - first.x * second.z + first.y * second.w + first.z * second.x,
      z: first.w * second.z + first.x * second.y - first.y * second.x + first.z * second.w,
      w: first.w * second.w - first.x * second.x - first.y * second.y - first.z * second.z,
    });
  }

  function quaternionFromAxisAngle(axis, angle) {
    const halfAngle = angle / 2;
    const sine = Math.sin(halfAngle);
    return quaternionNormalize({
      x: axis.x * sine,
      y: axis.y * sine,
      z: axis.z * sine,
      w: Math.cos(halfAngle),
    });
  }

  function quaternionFromEuler(rotationX, rotationY, rotationZ) {
    const aroundX = quaternionFromAxisAngle({ x: 1, y: 0, z: 0 }, rotationX);
    const aroundY = quaternionFromAxisAngle({ x: 0, y: 1, z: 0 }, rotationY);
    const aroundZ = quaternionFromAxisAngle({ x: 0, y: 0, z: 1 }, rotationZ);
    return quaternionMultiply(aroundZ, quaternionMultiply(aroundY, aroundX));
  }

  function rotateWorldPoint(world, rotation) {
    const vector = { x: rotation.x, y: rotation.y, z: rotation.z };
    const twiceCross = {
      x: 2 * (vector.y * world.z - vector.z * world.y),
      y: 2 * (vector.z * world.x - vector.x * world.z),
      z: 2 * (vector.x * world.y - vector.y * world.x),
    };
    return {
      x: world.x + rotation.w * twiceCross.x
        + vector.y * twiceCross.z - vector.z * twiceCross.y,
      y: world.y + rotation.w * twiceCross.y
        + vector.z * twiceCross.x - vector.x * twiceCross.z,
      z: world.z + rotation.w * twiceCross.z
        + vector.x * twiceCross.y - vector.y * twiceCross.x,
    };
  }

  function quaternionBetween(first, second) {
    const dot = first.x * second.x + first.y * second.y + first.z * second.z;
    if (dot < -0.9999) {
      const fallbackAxis = Math.abs(first.x) < 0.8
        ? { x: 0, y: -first.z, z: first.y }
        : { x: -first.y, y: first.x, z: 0 };
      return quaternionFromAxisAngle(fallbackAxis, Math.PI);
    }
    return quaternionNormalize({
      x: first.y * second.z - first.z * second.y,
      y: first.z * second.x - first.x * second.z,
      z: first.x * second.y - first.y * second.x,
      w: 1 + dot,
    });
  }

  function trackballVector(event, svg) {
    const bounds = svg.getBoundingClientRect();
    const x = (2 * (event.clientX - bounds.left)) / Math.max(bounds.width, 1) - 1;
    const y = 1 - (2 * (event.clientY - bounds.top)) / Math.max(bounds.height, 1);
    const distanceSquared = x * x + y * y;
    if (distanceSquared > 1) {
      const inverseLength = 1 / Math.sqrt(distanceSquared);
      return { x: x * inverseLength, y: y * inverseLength, z: 0 };
    }
    return { x, y, z: Math.sqrt(1 - distanceSquared) };
  }

  function niceTickStep(minimum, maximum, targetCount = 5) {
    const roughStep = Math.max((maximum - minimum) / targetCount, Number.EPSILON);
    const power = 10 ** Math.floor(Math.log10(roughStep));
    const normalized = roughStep / power;
    const multiplier = normalized <= 1.5 ? 1 : normalized <= 3 ? 2 : normalized <= 7 ? 5 : 10;
    return multiplier * power;
  }

  function majorTickValues(minimum, maximum) {
    let step = niceTickStep(minimum, maximum);
    let values = [];
    for (let attempt = 0; attempt < 3; attempt += 1) {
      const first = Math.ceil((minimum - step * 0.000001) / step) * step;
      values = [];
      for (let value = first; value <= maximum + step * 0.000001; value += step) {
        values.push(Math.abs(value) < step * 0.000001 ? 0 : value);
      }
      if (values.length <= 6) {
        break;
      }
      step = niceTickStep(0, step * 7, 5);
    }
    return { step, values };
  }

  function buildSceneGeometry(coordinates) {
    const numericCoordinates = coordinates.map((point) => ({
      x: Number(point.x),
      y: Number(point.y),
      z: Number(point.z),
    }));
    const dimensions = ["x", "y", "z"];
    const dataBounds = Object.fromEntries(dimensions.map((dimension) => [dimension, {
      minimum: Math.min(0, ...numericCoordinates.map((point) => point[dimension])),
      maximum: Math.max(0, ...numericCoordinates.map((point) => point[dimension])),
    }]));
    const referenceExtent = Math.max(
      0.000001,
      ...dimensions.map(
        (dimension) => dataBounds[dimension].maximum - dataBounds[dimension].minimum,
      ),
    );
    const dataCenter = Object.fromEntries(dimensions.map((dimension) => [
      dimension,
      (dataBounds[dimension].minimum + dataBounds[dimension].maximum) / 2,
    ]));
    const dataAnchors = numericCoordinates.concat({ x: 0, y: 0, z: 0 });
    const normalizationRadius = Math.max(
      ...dataAnchors.map((point) => Math.hypot(
        point.x - dataCenter.x,
        point.y - dataCenter.y,
        point.z - dataCenter.z,
      )),
      0.000001,
    );
    const toWorld = (point) => ({
      x: Number(point.x) / normalizationRadius,
      y: Number(point.y) / normalizationRadius,
      z: Number(point.z) / normalizationRadius,
    });
    // Presentation geometry wraps the data but never participates in the fit.
    // Each dimension keeps its true asymmetric PCA minimum and maximum.
    const coordinateRanges = Object.fromEntries(dimensions.map((dimension) => {
      const span = dataBounds[dimension].maximum - dataBounds[dimension].minimum;
      const decorationPadding = Math.max(span * 0.035, referenceExtent * 0.0125);
      return [dimension, {
        minimum: dataBounds[dimension].minimum - decorationPadding,
        maximum: dataBounds[dimension].maximum + decorationPadding,
      }];
    }));
    const axisRanges = Object.fromEntries(dimensions.map((dimension) => [dimension, {
      minimum: coordinateRanges[dimension].minimum / normalizationRadius,
      maximum: coordinateRanges[dimension].maximum / normalizationRadius,
    }]));
    const cornerCoordinates = [];
    [coordinateRanges.x.minimum, coordinateRanges.x.maximum].forEach((x) => {
      [coordinateRanges.y.minimum, coordinateRanges.y.maximum].forEach((y) => {
        [coordinateRanges.z.minimum, coordinateRanges.z.maximum].forEach((z) => {
          cornerCoordinates.push({ x, y, z });
        });
      });
    });
    const cuboidEdgeIndices = [
      [0, 4], [1, 5], [2, 6], [3, 7],
      [0, 2], [1, 3], [4, 6], [5, 7],
      [0, 1], [2, 3], [4, 5], [6, 7],
    ];
    return {
      points: numericCoordinates.map(toWorld),
      origin: Object.freeze({ x: 0, y: 0, z: 0 }),
      sceneCenter: toWorld(dataCenter),
      axisRanges,
      coordinateRanges,
      cuboid: {
        corners: cornerCoordinates.map((point, index) => ({
          index,
          coordinate: point,
          world: toWorld(point),
        })),
        edgeIndices: cuboidEdgeIndices,
      },
      ticks: Object.fromEntries(dimensions.map((dimension) => {
        const tickSet = majorTickValues(
          coordinateRanges[dimension].minimum,
          coordinateRanges[dimension].maximum,
        );
        return [dimension, {
          step: tickSet.step,
          values: tickSet.values.map((value) => ({
            value,
            world: value / normalizationRadius,
          })),
        }];
      })),
    };
  }

  function computeFittedSceneScale(sceneWidth, sceneHeight) {
    // Data points and O are normalized to a unit sphere around sceneCenter.
    // Decorative axes, grids, ticks, and the cuboid never reduce this scale.
    const edgePadding = Math.max(scenePadding, maximumMarkerExtent);
    const safeWidth = Math.max(2, sceneWidth - 2 * edgePadding);
    const safeHeight = Math.max(2, sceneHeight - 2 * edgePadding);
    return Math.min(safeWidth, safeHeight) / 2;
  }

  function safePointPreview(point) {
    const searchResult = searchResultByChunkId.get(point.id);
    const rawPreview = point.preview || (searchResult ? searchResult.content : "");
    const preview = String(rawPreview || "").replace(/\s+/g, " ").trim();
    return preview;
  }

  function formatCoordinate(value) {
    const number = Number(value);
    const magnitude = Math.abs(number);
    if (magnitude >= 10000 || (magnitude > 0 && magnitude < 0.001)) {
      return number.toExponential(2);
    }
    return number.toFixed(3).replace(/\.000$/, ".0");
  }

  function formatTickValue(value, step) {
    const rounded = Math.abs(value) < step * 0.000001 ? 0 : value;
    if (step >= 10000 || step < 0.0001) {
      return rounded.toExponential(1).replace(/\.0e/, "e");
    }
    const decimals = clamp(Math.max(0, -Math.floor(Math.log10(step))), 0, 4);
    return rounded.toFixed(decimals).replace(/(\.\d*?[1-9])0+$|\.0+$/, "$1");
  }

  function appendCoordinateReadout(container, point) {
    const readout = document.createElement("dl");
    readout.className = "embedding-coordinate-readout";
    ["X", "Y", "Z"].forEach((label) => {
      const coordinate = document.createElement("div");
      const term = document.createElement("dt");
      term.textContent = label;
      const value = document.createElement("dd");
      value.textContent = formatCoordinate(point[label.toLowerCase()]);
      coordinate.append(term, value);
      readout.appendChild(coordinate);
    });
    container.appendChild(readout);
  }

  function appendHoverCoordinates(container, point) {
    const coordinates = document.createElement("span");
    coordinates.className = "embedding-hover-coordinates d-block";
    coordinates.textContent = ["x", "y", "z"]
      .map((dimension) => `${dimension.toUpperCase()} ${formatCoordinate(point[dimension])}`)
      .join(" · ");
    container.appendChild(coordinates);
  }

  function renderPinnedInspector() {
    visualizationInspectorContent.replaceChildren();
    if (!viewerState || (!pinnedQuery && !pinnedChunkId)) {
      visualizationInspector.hidden = true;
      visualizationInspectorTitle.textContent = "Inspector";
      return;
    }

    if (pinnedQuery) {
      visualizationInspectorTitle.textContent = "Query";
      appendCoordinateReadout(visualizationInspectorContent, viewerState.query.data);
      visualizationInspector.hidden = false;
      return;
    }

    const selected = viewerState.points.find((entry) => entry.data.id === pinnedChunkId);
    if (!selected) {
      visualizationInspector.hidden = true;
      return;
    }
    const point = selected.data;
    visualizationInspectorTitle.textContent = point.retrieved
      ? `#${point.rank}`
      : "Indexed chunk";
    const source = document.createElement("p");
    source.className = "embedding-inspector-source";
    source.textContent = point.source || "Unknown source";
    visualizationInspectorContent.appendChild(source);
    const previewText = safePointPreview(point);
    if (previewText) {
      const preview = document.createElement("p");
      preview.className = "embedding-inspector-preview";
      preview.textContent = previewText;
      visualizationInspectorContent.appendChild(preview);
    }
    appendCoordinateReadout(visualizationInspectorContent, point);
    visualizationInspector.hidden = false;
  }

  function projectWorldPoint(world, state) {
    const centered = {
      x: world.x - state.sceneCenterWorld.x,
      y: world.y - state.sceneCenterWorld.y,
      z: world.z - state.sceneCenterWorld.z,
    };
    const rotated = rotateWorldPoint(centered, state.rotation);
    const sceneScale = state.fittedSceneScale * state.zoom;
    return {
      x: state.sceneBounds.x + state.sceneBounds.width / 2 + rotated.x * sceneScale,
      y: state.sceneBounds.y + state.sceneBounds.height / 2 - rotated.y * sceneScale,
      depth: rotated.z,
      depthRatio: clamp((rotated.z + 1) / 2, 0, 1),
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

  function renderChunkHoverLabel(point, projected, state) {
    visualizationTooltip.replaceChildren();
    const heading = document.createElement("strong");
    heading.className = "d-block";
    heading.textContent = point.retrieved
      ? `#${point.rank} · ${point.source}`
      : point.source || "Indexed chunk";
    visualizationTooltip.appendChild(heading);
    appendHoverCoordinates(visualizationTooltip, point);
    visualizationTooltip.hidden = false;
    positionPointTooltip(projected, state);
  }

  function renderQueryHoverLabel(query, projected, state) {
    visualizationTooltip.replaceChildren();
    const heading = document.createElement("strong");
    heading.className = "d-block";
    heading.textContent = "Query";
    visualizationTooltip.appendChild(heading);
    appendHoverCoordinates(visualizationTooltip, query);
    visualizationTooltip.hidden = false;
    positionPointTooltip(projected, state);
  }

  function clearPointTooltip() {
    visualizationTooltip.hidden = true;
    visualizationTooltip.replaceChildren();
    visualizationTooltip.style.removeProperty("left");
    visualizationTooltip.style.removeProperty("top");
  }

  function setProjectedLine(line, start, end) {
    line.setAttribute("x1", String(start.x));
    line.setAttribute("y1", String(start.y));
    line.setAttribute("x2", String(end.x));
    line.setAttribute("y2", String(end.y));
  }

  function shortenProjectedLineEnd(start, end, distance) {
    const length = Math.hypot(end.x - start.x, end.y - start.y);
    if (length <= distance || length === 0) {
      return end;
    }
    const ratio = (length - distance) / length;
    return {
      x: start.x + (end.x - start.x) * ratio,
      y: start.y + (end.y - start.y) * ratio,
    };
  }

  function positionSpatialLabel(label, projected, offsetX = 10, offsetY = -10) {
    label.setAttribute("x", String(projected.x + offsetX));
    label.setAttribute("y", String(projected.y + offsetY));
  }

  function renderViewerScene() {
    const state = viewerState;
    if (!state) {
      return;
    }

    const projectedOrigin = projectWorldPoint(state.origin.world, state);
    const projectedQuery = projectWorldPoint(state.query.world, state);
    const projectedPoints = state.points.map((entry) => ({
      entry,
      projected: projectWorldPoint(entry.world, state),
    }));
    const projectedById = new Map(
      projectedPoints.map(({ entry, projected }) => [entry.data.id, projected]),
    );

    const projectedCuboidCorners = state.cuboid.corners.map((corner) => ({
      corner,
      projected: projectWorldPoint(corner.world, state),
    }));
    state.cuboid.edges.forEach((edge) => {
      const projectedStart = projectedCuboidCorners[edge.startIndex].projected;
      const projectedEnd = projectedCuboidCorners[edge.endIndex].projected;
      setProjectedLine(edge.visual, projectedStart, projectedEnd);
      edge.visual.style.opacity = String(
        0.14 + ((projectedStart.depthRatio + projectedEnd.depthRatio) / 2) * 0.16,
      );
    });
    projectedCuboidCorners.forEach(({ corner, projected }) => {
      corner.visual.setAttribute("transform", `translate(${projected.x} ${projected.y})`);
      corner.visual.style.opacity = String(0.18 + projected.depthRatio * 0.16);
    });

    state.gridLines.forEach((gridLine) => {
      const projectedStart = projectWorldPoint(gridLine.startWorld, state);
      const projectedEnd = projectWorldPoint(gridLine.endWorld, state);
      setProjectedLine(gridLine.visual, projectedStart, projectedEnd);
      gridLine.visual.style.opacity = String(
        gridLine.baseOpacity
          + ((projectedStart.depthRatio + projectedEnd.depthRatio) / 2) * gridLine.depthOpacity,
      );
    });

    state.axes.forEach((axis) => {
      const projectedNegative = projectWorldPoint(axis.negativeWorld, state);
      const projectedPositive = projectWorldPoint(axis.positiveWorld, state);
      setProjectedLine(axis.visual.negativeLine, projectedOrigin, projectedNegative);
      setProjectedLine(axis.visual.positiveLine, projectedOrigin, projectedPositive);
      positionSpatialLabel(axis.visual.label, projectedPositive);
      axis.ticks.forEach((tick) => {
        const projectedStart = projectWorldPoint(tick.startWorld, state);
        const projectedEnd = projectWorldPoint(tick.endWorld, state);
        const projectedCenter = projectWorldPoint(tick.centerWorld, state);
        setProjectedLine(tick.visual.line, projectedStart, projectedEnd);
        positionSpatialLabel(tick.visual.label, projectedCenter, 6, -6);
      });
      const axisDepthRatio = clamp(
        (projectedNegative.depthRatio + projectedPositive.depthRatio) / 2,
        0,
        1,
      );
      axis.visual.group.style.opacity = String(0.48 + axisDepthRatio * 0.38);
    });

    state.origin.visual.group.setAttribute(
      "transform",
      `translate(${projectedOrigin.x} ${projectedOrigin.y})`,
    );
    const queryScale = 0.9 + projectedQuery.depthRatio * 0.25;
    const projectedVectorEnd = shortenProjectedLineEnd(
      projectedOrigin,
      projectedQuery,
      24 * queryScale,
    );
    setProjectedLine(state.query.visual.vector, projectedOrigin, projectedVectorEnd);
    state.query.visual.vector.style.opacity = String(0.8 + projectedQuery.depthRatio * 0.2);

    const activeId = activeChunkId();
    state.connections.forEach((connection, chunkId) => {
      const projectedChunk = projectedById.get(chunkId);
      if (!projectedChunk) {
        return;
      }
      setProjectedLine(connection, projectedQuery, projectedChunk);
      connection.classList.toggle("is-active", chunkId === activeId);
      connection.style.opacity = chunkId === activeId
        ? "1"
        : String(0.32 + ((projectedQuery.depthRatio + projectedChunk.depthRatio) / 2) * 0.28);
    });

    // SVG has no depth buffer, so far-to-near DOM ordering, scale, and opacity
    // provide consistent depth cues without adding perspective distortion.
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

    const activePoint = projectedPoints.find(({ entry }) => entry.data.id === activeId);
    if (activePoint) {
      state.pointLayer.appendChild(activePoint.entry.visual.group);
    }

    state.query.visual.group.setAttribute(
      "transform",
      `translate(${projectedQuery.x} ${projectedQuery.y}) scale(${queryScale})`,
    );

    state.query.visual.group.classList.toggle(
      "is-inspected",
      state.queryHovered || state.queryFocused || pinnedQuery,
    );
    state.query.visual.group.classList.toggle("is-pinned", pinnedQuery);
    state.query.visual.group.setAttribute("aria-pressed", String(pinnedQuery));
    if (state.queryHovered || state.queryFocused) {
      renderQueryHoverLabel(state.query.data, projectedQuery, state);
    } else {
      const tooltipId = activeId || state.hoveredScenePointId;
      const tooltipPoint = state.points.find((entry) => entry.data.id === tooltipId);
      const projectedTooltipPoint = tooltipId ? projectedById.get(tooltipId) : null;
      if (tooltipPoint && projectedTooltipPoint) {
        renderChunkHoverLabel(tooltipPoint.data, projectedTooltipPoint, state);
      } else {
        clearPointTooltip();
      }
    }
  }

  function installCameraInteraction(state) {
    const { svg } = state;
    svg.addEventListener("wheel", (event) => {
      if (event.deltaY === 0) {
        return;
      }
      event.preventDefault();
      zoomViewer(event.deltaY < 0 ? wheelZoomFactor : 1 / wheelZoomFactor);
    }, { passive: false });
    svg.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) {
        return;
      }
      state.pointerId = event.pointerId;
      state.lastTrackballVector = trackballVector(event, svg);
      state.dragMoved = false;
      svg.setPointerCapture(event.pointerId);
      svg.classList.add("is-dragging");
      event.preventDefault();
    });
    svg.addEventListener("pointermove", (event) => {
      if (state.pointerId !== event.pointerId) {
        return;
      }
      const nextTrackballVector = trackballVector(event, svg);
      const movement = Math.hypot(
        nextTrackballVector.x - state.lastTrackballVector.x,
        nextTrackballVector.y - state.lastTrackballVector.y,
        nextTrackballVector.z - state.lastTrackballVector.z,
      );
      if (movement > 0.002) {
        state.dragMoved = true;
      }
      const deltaRotation = quaternionBetween(state.lastTrackballVector, nextTrackballVector);
      state.rotation = quaternionMultiply(deltaRotation, state.rotation);
      state.lastTrackballVector = nextTrackballVector;
      renderViewerScene();
      event.preventDefault();
    });

    const endDrag = (event) => {
      if (state.pointerId !== event.pointerId) {
        return;
      }
      try {
        if (svg.hasPointerCapture(event.pointerId)) {
          svg.releasePointerCapture(event.pointerId);
        }
      } catch {
        // Pointer capture can end before cancellation or a lost-capture event.
      }
      state.pointerId = null;
      state.lastTrackballVector = null;
      svg.classList.remove("is-dragging");
    };
    svg.addEventListener("pointerup", endDrag);
    svg.addEventListener("pointercancel", endDrag);
    svg.addEventListener("lostpointercapture", endDrag);
    svg.addEventListener("click", () => {
      if (!state.dragMoved) {
        const activeElement = document.activeElement;
        if (activeElement && svg.contains(activeElement) && typeof activeElement.blur === "function") {
          activeElement.blur();
        }
        focusedChunkId = null;
        pinnedChunkId = null;
        pinnedQuery = false;
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
    state.lastTrackballVector = null;
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
    pinnedQuery = false;
    state.hoveredScenePointId = null;
    state.queryHovered = false;
    state.queryFocused = false;
    const activeElement = document.activeElement;
    if (activeElement && state.svg.contains(activeElement) && typeof activeElement.blur === "function") {
      activeElement.blur();
    }
    resetPointerInteraction(state);
    state.rotation = { ...state.initialCamera.rotation };
    state.zoom = state.initialCamera.zoom;
    updateZoomControls(state);
    clearPointTooltip();
    renderPinnedInspector();
    updateLinkedInteraction(false);
    renderViewerScene();
  }

  function resizeViewerToCanvas(state) {
    if (viewerState !== state) {
      return;
    }
    const bounds = visualizationCanvas.getBoundingClientRect();
    const width = Math.max(320, Math.round(bounds.width || state.width));
    const height = Math.max(360, Math.round(bounds.height || state.height));
    if (width === state.width && height === state.height) {
      return;
    }
    state.width = width;
    state.height = height;
    state.sceneBounds = Object.freeze({ x: 0, y: 0, width, height });
    state.fittedSceneScale = computeFittedSceneScale(width, height);
    state.svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    state.background.setAttribute("width", String(width));
    state.background.setAttribute("height", String(height));
    renderViewerScene();
  }

  function setPresentationMode(expanded, resize = true) {
    visualizationPanel.classList.toggle("is-expanded", expanded);
    visualizationPanel.setAttribute("aria-expanded", String(expanded));
    expandViewButton.hidden = expanded;
    restoreViewButton.hidden = !expanded;
    closeExpandedViewButton.hidden = !expanded;
    if (!resize || !viewerState) {
      return;
    }
    resizeViewerToCanvas(viewerState);
    if (typeof requestAnimationFrame === "function") {
      requestAnimationFrame(() => {
        if (viewerState) {
          resizeViewerToCanvas(viewerState);
        }
      });
    }
  }

  function visualizationDisplayNote(note) {
    const message = String(note || "").trim();
    if (message === redundantVisualizationNote) {
      return "";
    }
    if (message.startsWith(`${redundantVisualizationNote} `)) {
      return message.slice(redundantVisualizationNote.length).trim();
    }
    return message;
  }

  function renderVisualization(visualization, searchMode) {
    visualizationLabel.textContent = visualization.label;
    visualizationTechnicalLabel.textContent = visualization.technical_label;
    const displayNote = visualizationDisplayNote(visualization.note);
    visualizationNote.textContent = displayNote;
    setPresentationMode(false, false);
    if (viewerState && viewerState.resizeObserver) {
      viewerState.resizeObserver.disconnect();
    }
    visualizationCanvas.replaceChildren(visualizationTooltip, visualizationInspector);
    visualizationTooltip.hidden = true;
    visualizationTooltip.replaceChildren();
    visualizationInspector.hidden = true;
    visualizationInspectorContent.replaceChildren();
    visualizationPointByChunkId.clear();
    viewerState = null;
    hoveredChunkId = null;
    focusedChunkId = null;
    pinnedChunkId = null;
    pinnedQuery = false;

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
    visualizationNote.hidden = !canRender || !displayNote;
    visualizationStatus.hidden = canRender;
    visualizationPanel.hidden = !canRender;
    if (!canRender) {
      visualizationStatus.textContent = visualization.note;
      return;
    }

    const canvasBounds = visualizationCanvas.getBoundingClientRect();
    const width = Math.max(320, Math.round(canvasBounds.width || 960));
    const height = Math.max(360, Math.round(canvasBounds.height || 600));
    const sceneBounds = Object.freeze({
      x: 0,
      y: 0,
      width,
      height,
    });
    const sceneGeometry = buildSceneGeometry(allCoordinates);
    const fittedSceneScale = computeFittedSceneScale(sceneBounds.width, sceneBounds.height);
    const svg = svgElement("svg", "embedding-map");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("role", "group");
    svg.setAttribute("aria-label", "Interactive three-dimensional projection of indexed chunks and the Query");

    const background = svgElement("rect", "embedding-map-background");
    background.setAttribute("x", "0");
    background.setAttribute("y", "0");
    background.setAttribute("width", String(width));
    background.setAttribute("height", String(height));
    svg.appendChild(background);

    const definitions = svgElement("defs", "");
    const axisArrow = svgElement("marker", "");
    axisArrow.id = "embedding-axis-arrowhead";
    axisArrow.setAttribute("viewBox", "0 0 10 10");
    axisArrow.setAttribute("refX", "9");
    axisArrow.setAttribute("refY", "5");
    axisArrow.setAttribute("markerWidth", "7");
    axisArrow.setAttribute("markerHeight", "7");
    axisArrow.setAttribute("orient", "auto-start-reverse");
    const axisArrowPath = svgElement("path", "embedding-axis-arrow");
    axisArrowPath.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
    axisArrow.appendChild(axisArrowPath);
    const queryArrow = svgElement("marker", "");
    queryArrow.id = "query-vector-arrowhead";
    queryArrow.setAttribute("viewBox", "0 0 10 10");
    queryArrow.setAttribute("refX", "8");
    queryArrow.setAttribute("refY", "5");
    queryArrow.setAttribute("markerWidth", "9");
    queryArrow.setAttribute("markerHeight", "9");
    queryArrow.setAttribute("orient", "auto-start-reverse");
    const queryArrowPath = svgElement("path", "query-vector-arrow");
    queryArrowPath.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
    queryArrow.appendChild(queryArrowPath);
    definitions.append(axisArrow, queryArrow);
    svg.appendChild(definitions);

    const cuboidLayer = svgElement("g", "embedding-cuboid");
    ["x", "y", "z"].forEach((dimension) => {
      cuboidLayer.dataset[`${dimension}Minimum`] = String(
        sceneGeometry.coordinateRanges[dimension].minimum,
      );
      cuboidLayer.dataset[`${dimension}Maximum`] = String(
        sceneGeometry.coordinateRanges[dimension].maximum,
      );
    });
    const gridLayer = svgElement("g", "embedding-grid-layer");
    const axisLayer = svgElement("g", "embedding-axis-layer");
    const connectionLayer = svgElement("g", "embedding-connection-layer");
    const queryVectorLayer = svgElement("g", "embedding-query-vector-layer");
    const pointLayer = svgElement("g", "embedding-point-layer");
    const originLayer = svgElement("g", "embedding-origin-layer");
    const queryLayer = svgElement("g", "embedding-query-layer");
    svg.append(
      cuboidLayer,
      gridLayer,
      axisLayer,
      connectionLayer,
      queryVectorLayer,
      pointLayer,
      originLayer,
      queryLayer,
    );

    const initialCamera = Object.freeze({
      rotation: defaultCamera.rotation,
      zoom: defaultCamera.zoom,
    });
    const state = {
      width,
      height,
      svg,
      background,
      cuboidLayer,
      gridLayer,
      axisLayer,
      connectionLayer,
      queryVectorLayer,
      pointLayer,
      originLayer,
      sceneBounds,
      sceneCenterWorld: sceneGeometry.sceneCenter,
      cuboid: null,
      gridLines: [],
      axes: [],
      connections: new Map(),
      points: [],
      origin: null,
      query: null,
      initialCamera,
      fittedSceneScale,
      rotation: { ...initialCamera.rotation },
      zoom: initialCamera.zoom,
      pointerId: null,
      lastTrackballVector: null,
      dragMoved: false,
      hoveredScenePointId: null,
      queryHovered: false,
      queryFocused: false,
      resizeObserver: null,
    };

    const cuboidCorners = sceneGeometry.cuboid.corners.map((corner) => {
      const visual = svgElement("circle", "embedding-cuboid-corner");
      visual.setAttribute("r", "1.7");
      visual.dataset.cornerIndex = String(corner.index);
      visual.dataset.x = String(corner.coordinate.x);
      visual.dataset.y = String(corner.coordinate.y);
      visual.dataset.z = String(corner.coordinate.z);
      cuboidLayer.appendChild(visual);
      return { ...corner, visual };
    });
    const cuboidEdges = sceneGeometry.cuboid.edgeIndices.map(([startIndex, endIndex]) => {
      const visual = svgElement("line", "embedding-cuboid-edge");
      visual.dataset.startCorner = String(startIndex);
      visual.dataset.endCorner = String(endIndex);
      cuboidLayer.appendChild(visual);
      return { startIndex, endIndex, visual };
    });
    state.cuboid = { corners: cuboidCorners, edges: cuboidEdges };

    const addGridLine = (startWorld, endWorld, plane, isOriginLine) => {
      const line = svgElement(
        "line",
        `embedding-grid-line${isOriginLine ? " embedding-grid-origin-line" : ""}`,
      );
      line.dataset.plane = plane;
      gridLayer.appendChild(line);
      state.gridLines.push({
        startWorld,
        endWorld,
        plane,
        baseOpacity: plane === "xy" ? 0.09 : 0.035,
        depthOpacity: plane === "xy" ? 0.14 : 0.075,
        visual: line,
      });
    };
    sceneGeometry.ticks.x.values.forEach((tick) => {
      addGridLine(
        { x: tick.world, y: sceneGeometry.axisRanges.y.minimum, z: 0 },
        { x: tick.world, y: sceneGeometry.axisRanges.y.maximum, z: 0 },
        "xy",
        tick.value === 0,
      );
      addGridLine(
        { x: tick.world, y: 0, z: sceneGeometry.axisRanges.z.minimum },
        { x: tick.world, y: 0, z: sceneGeometry.axisRanges.z.maximum },
        "xz",
        tick.value === 0,
      );
    });
    sceneGeometry.ticks.y.values.forEach((tick) => {
      addGridLine(
        { x: sceneGeometry.axisRanges.x.minimum, y: tick.world, z: 0 },
        { x: sceneGeometry.axisRanges.x.maximum, y: tick.world, z: 0 },
        "xy",
        tick.value === 0,
      );
      addGridLine(
        { x: 0, y: tick.world, z: sceneGeometry.axisRanges.z.minimum },
        { x: 0, y: tick.world, z: sceneGeometry.axisRanges.z.maximum },
        "yz",
        tick.value === 0,
      );
    });
    sceneGeometry.ticks.z.values.forEach((tick) => {
      addGridLine(
        { x: sceneGeometry.axisRanges.x.minimum, y: 0, z: tick.world },
        { x: sceneGeometry.axisRanges.x.maximum, y: 0, z: tick.world },
        "xz",
        tick.value === 0,
      );
      addGridLine(
        { x: 0, y: sceneGeometry.axisRanges.y.minimum, z: tick.world },
        { x: 0, y: sceneGeometry.axisRanges.y.maximum, z: tick.world },
        "yz",
        tick.value === 0,
      );
    });

    ["x", "y", "z"].forEach((dimension) => {
      const group = svgElement("g", `embedding-axis embedding-axis-${dimension}`);
      const negativeLine = svgElement("line", "embedding-axis-line embedding-axis-negative");
      const positiveLine = svgElement("line", "embedding-axis-line embedding-axis-positive");
      positiveLine.setAttribute("marker-end", "url(#embedding-axis-arrowhead)");
      const label = svgElement("text", "embedding-axis-label");
      label.textContent = dimension.toUpperCase();
      const negativeWorld = { x: 0, y: 0, z: 0 };
      const positiveWorld = { x: 0, y: 0, z: 0 };
      negativeWorld[dimension] = sceneGeometry.axisRanges[dimension].minimum;
      positiveWorld[dimension] = sceneGeometry.axisRanges[dimension].maximum;
      const ticks = sceneGeometry.ticks[dimension].values
        .filter((tick) => tick.value !== 0)
        .map((tick) => {
          const centerWorld = { x: 0, y: 0, z: 0 };
          centerWorld[dimension] = tick.world;
          const startWorld = { ...centerWorld };
          const endWorld = { ...centerWorld };
          const tickDimension = dimension === "x" ? "y" : "x";
          startWorld[tickDimension] = -0.012;
          endWorld[tickDimension] = 0.012;
          const tickLine = svgElement("line", "embedding-axis-tick");
          const tickLabel = svgElement("text", "embedding-axis-tick-label");
          tickLabel.textContent = formatTickValue(tick.value, sceneGeometry.ticks[dimension].step);
          group.append(tickLine, tickLabel);
          return {
            startWorld,
            endWorld,
            centerWorld,
            value: tick.value,
            visual: { line: tickLine, label: tickLabel },
          };
        });
      group.append(negativeLine, positiveLine, label);
      axisLayer.appendChild(group);
      state.axes.push({
        dimension,
        negativeWorld,
        positiveWorld,
        ticks,
        visual: { group, negativeLine, positiveLine, label },
      });
    });

    const originGroup = svgElement("g", "embedding-origin");
    const originMarker = svgElement("circle", "embedding-origin-marker");
    originMarker.setAttribute("r", "5");
    const originLabel = svgElement("text", "embedding-origin-label");
    originLabel.setAttribute("x", "11");
    originLabel.setAttribute("y", "-10");
    originLabel.textContent = "O";
    originGroup.append(originMarker, originLabel);
    originLayer.appendChild(originGroup);
    state.origin = {
      world: sceneGeometry.origin,
      visual: { group: originGroup },
    };

    const queryVector = svgElement("line", "query-vector");
    queryVector.setAttribute("marker-end", "url(#query-vector-arrowhead)");
    queryVectorLayer.appendChild(queryVector);

    visualization.points.forEach((point, index) => {
      const group = svgElement(
        "g",
        point.retrieved ? "embedding-point retrieved-point" : "embedding-point indexed-point",
      );
      group.dataset.chunkId = point.id;
      const marker = svgElement("circle", "embedding-point-marker");
      marker.setAttribute("r", point.retrieved ? "14" : "5");
      group.appendChild(marker);
      const nativeTooltip = svgElement("title", "");
      nativeTooltip.textContent = point.retrieved
        ? `Retrieved rank ${point.rank}: ${point.source}`
        : `Indexed chunk: ${point.source}`;
      group.appendChild(nativeTooltip);

      group.addEventListener("mouseenter", () => {
        state.hoveredScenePointId = point.id;
        if (point.retrieved) {
          setHoveredChunk(point.id);
        } else {
          renderViewerScene();
        }
      });
      group.addEventListener("mouseleave", () => {
        if (state.hoveredScenePointId === point.id) {
          state.hoveredScenePointId = null;
        }
        if (point.retrieved) {
          clearHoveredChunk(point.id);
        } else {
          renderViewerScene();
        }
      });

      group.setAttribute("tabindex", "0");
      group.setAttribute("role", "button");
      group.setAttribute(
        "aria-label",
        point.retrieved
          ? `Retrieved rank ${point.rank}, ${point.source}. Press Enter to pin this point.`
          : `Indexed chunk, ${point.source}. Press Enter to pin this point.`,
      );
      group.setAttribute("aria-pressed", "false");
      visualizationPointByChunkId.set(point.id, group);
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

      if (point.retrieved) {
        const rank = svgElement("text", "embedding-rank");
        rank.setAttribute("text-anchor", "middle");
        rank.setAttribute("dominant-baseline", "central");
        rank.textContent = `#${point.rank}`;
        group.appendChild(rank);
        const connection = svgElement("line", "query-retrieval-connection");
        connection.dataset.chunkId = point.id;
        connectionLayer.appendChild(connection);
        state.connections.set(point.id, connection);
      }
      const entry = {
        data: point,
        world: sceneGeometry.points[index],
        visual: { group },
      };
      state.points.push(entry);
      pointLayer.appendChild(group);
    });

    const queryGroup = svgElement("g", "embedding-query-point");
    queryGroup.setAttribute("tabindex", "0");
    queryGroup.setAttribute("role", "button");
    queryGroup.setAttribute("aria-pressed", "false");
    queryGroup.setAttribute("aria-label", "Query vector endpoint. Press Enter to pin its PCA coordinates.");
    const queryMarker = svgElement("polygon", "embedding-query-marker");
    queryMarker.setAttribute("points", "0,-21 21,0 0,21 -21,0");
    const queryLabel = svgElement("text", "embedding-query-label");
    queryLabel.setAttribute("text-anchor", "middle");
    queryLabel.setAttribute("dominant-baseline", "central");
    queryLabel.textContent = "Q";
    const queryTooltip = svgElement("title", "");
    queryTooltip.textContent = `Query — X ${formatCoordinate(visualization.query.x)}, Y ${formatCoordinate(visualization.query.y)}, Z ${formatCoordinate(visualization.query.z)}`;
    queryGroup.append(queryMarker, queryLabel, queryTooltip);
    queryLayer.appendChild(queryGroup);
    state.query = {
      data: visualization.query,
      world: sceneGeometry.points[visualization.points.length],
      visual: { group: queryGroup, vector: queryVector },
    };

    queryGroup.addEventListener("mouseenter", () => {
      state.queryHovered = true;
      renderViewerScene();
    });
    queryGroup.addEventListener("mouseleave", () => {
      state.queryHovered = false;
      renderViewerScene();
    });
    queryGroup.addEventListener("focus", () => {
      state.queryFocused = true;
      renderViewerScene();
    });
    queryGroup.addEventListener("blur", () => {
      state.queryFocused = false;
      renderViewerScene();
    });
    queryGroup.addEventListener("click", (event) => {
      event.stopPropagation();
      if (!state.dragMoved) {
        togglePinnedQuery();
      }
      state.dragMoved = false;
    });
    queryGroup.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        togglePinnedQuery();
      }
    });

    // These lines show retrieval participation only. Retrieval scores and
    // nearest-neighbour search remain full-dimensional and are not computed here.
    viewerState = state;
    updateZoomControls(state);
    visualizationCanvas.insertBefore(svg, visualizationTooltip);
    installCameraInteraction(state);
    renderViewerScene();
    updateLinkedInteraction();
    if (typeof ResizeObserver === "function") {
      state.resizeObserver = new ResizeObserver(() => resizeViewerToCanvas(state));
      state.resizeObserver.observe(visualizationCanvas);
    }
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

  refreshButton.addEventListener("click", () => loadIndex(true));
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
  expandViewButton.addEventListener("click", () => setPresentationMode(true));
  restoreViewButton.addEventListener("click", () => setPresentationMode(false));
  closeExpandedViewButton.addEventListener("click", () => setPresentationMode(false));
  visualizationInspectorClose.addEventListener("click", clearPinnedSelection);
  refreshExamplesButton.addEventListener("click", loadExamples);
  loadExamples();
  loadIndex();
})();
