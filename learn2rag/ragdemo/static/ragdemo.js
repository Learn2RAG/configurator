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

  refreshButton.addEventListener("click", loadIndex);
  loadIndex();
})();
