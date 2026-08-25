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
  const askButton = document.querySelector("#ask-button");
  const queryError = document.querySelector("#query-error");
  const queryLoading = document.querySelector("#query-loading");
  const queryResults = document.querySelector("#query-results");
  const answerContent = document.querySelector("#answer-content");
  const searchLabel = document.querySelector("#search-label");
  const searchTechnicalLabel = document.querySelector("#search-technical-label");
  const scoreLabel = document.querySelector("#score-label");
  const searchResults = document.querySelector("#search-results");
  const promptLabel = document.querySelector("#prompt-label");
  const promptTechnicalLabel = document.querySelector("#prompt-technical-label");
  const promptNote = document.querySelector("#prompt-note");
  const promptMessages = document.querySelector("#prompt-messages");
  const exampleButtons = document.querySelectorAll(".example-question");

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

  function renderSearchResults(results) {
    searchResults.replaceChildren();
    if (results.length === 0) {
      const emptyState = document.createElement("p");
      emptyState.className = "alert alert-secondary mb-0";
      emptyState.textContent = "No matching chunks were retrieved.";
      searchResults.appendChild(emptyState);
      return;
    }

    results.forEach((result) => {
      const card = document.createElement("article");
      card.className = "card retrieval-card";

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
      chunk.textContent = result.content;
      body.appendChild(chunk);

      card.append(header, body);
      searchResults.appendChild(card);
    });
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
    answerContent.textContent = data.answer;
    searchLabel.textContent = data.search.label;
    searchTechnicalLabel.textContent = data.search.technical_label;
    scoreLabel.textContent = data.search.score_label;
    renderSearchResults(data.search.results);
    renderPrompt(data.prompt);
    queryError.hidden = true;
    queryResults.hidden = false;
  }

  async function submitQuery(event) {
    event.preventDefault();
    const question = questionInput.value.trim();
    queryError.hidden = true;
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
        body: JSON.stringify({ question }),
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
  exampleButtons.forEach((button) => {
    button.addEventListener("click", () => {
      questionInput.value = button.dataset.question || "";
      questionInput.focus();
    });
  });
  loadIndex();
})();
