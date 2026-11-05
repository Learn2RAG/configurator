(() => {
  "use strict";

  const languageStorageKey = "learn2rag-ragdemo-language";
  const translations = Object.freeze({
    en: Object.freeze({
      page_title: "Learn2RAG · Interactive RAG Demo",
      navigation_label: "RAG Demo navigation",
      demo_tagline: "Interactive RAG Demo",
      demo_sections: "Demo sections",
      index: "Index",
      ask_rag: "Ask RAG",
      language_selector: "Interface language",
      use_english: "Use English",
      use_german: "Use German",
      indexed_documents: "Indexed documents",
      indexed_documents_intro: "These documents have already been processed and stored in the RAG index.",
      refresh: "Refresh",
      loading_index: "Loading indexed documents…",
      index_summary: "Index summary",
      documents: "Documents",
      chunks: "Chunks",
      partial_index_note: "The safety scan limit was reached, so these counts show a partial index snapshot.",
      ask_intro: "Ask once, then compare four already-prepared answers for the same question.",
      question: "Question",
      question_placeholder: "What would you like to know?",
      retrieval_method: "Retrieval method",
      semantic_vectors: "Semantic vectors",
      semantic_description: "Search by meaning, not only exact words.",
      keyword_search: "Keyword search",
      keyword_description: "Match the words you typed.",
      hybrid_search: "Hybrid search",
      hybrid_description: "Combine semantic and sparse lexical retrieval.",
      question_prepares_all: "One question prepares Semantic, Keyword, Hybrid and Generative AI results.",
      indexed_documents_only: "Questions are answered using the currently indexed documents.",
      example_questions: "Example questions",
      try_example: "Try an example",
      loading_examples: "Loading examples…",
      suggest_examples: "Suggest more examples",
      query_loading: "Preparing Semantic, Keyword, Hybrid and Generative AI results…",
      compare_loaded_results: "Compare results",
      result_mode: "Result mode",
      semantic_mode: "Semantic",
      keyword_mode: "Keyword",
      hybrid_mode: "Hybrid",
      generative_ai: "Generative AI",
      generative_card_description: "Answer without retrieved documents.",
      semantic_card_description: "Meaning-based dense retrieval.",
      keyword_card_description: "Keyword-based BM25 retrieval.",
      hybrid_card_description: "Combines dense and sparse retrieval.",
      semantic_mode_description: "Answer grounded in dense semantic retrieval.",
      keyword_mode_description: "Answer grounded in BM25 keyword retrieval.",
      hybrid_mode_description: "Answer grounded in native dense and sparse fusion.",
      generative_mode_description: "Answer directly from the language model without retrieved documents.",
      receive_question: "Receive question",
      no_retrieval: "No retrieval",
      build_direct_prompt: "Build direct prompt",
      no_indexed_documents: "No indexed documents used.",
      embed_question: "Embed question",
      retrieve_chunks: "Retrieve top-ranked chunks",
      add_chunks_prompt: "Add chunks to prompt",
      generate_answer: "Generate answer",
      explore_embedding_space: "Explore the embedding space",
      projection: "Projection:",
      projection_selector: "Projection method",
      pca_projection: "3D PCA projection of dense embeddings",
      tsne_projection: "3D t-SNE projection of dense embeddings",
      umap_projection: "3D UMAP projection of dense embeddings",
      visualization_intro: "All indexed chunks form the surrounding embedding space. The Query vector and its links identify the chunks selected for the RAG context.",
      pca_caveat: "Spatial proximity is approximate because PCA compresses the original embedding space. Semantic similarity is calculated using the original full-dimensional vectors.",
      tsne_caveat: "t-SNE emphasizes local neighbourhood structure. Distances in this visualization are approximate and do not determine retrieval. Semantic similarity is calculated using the original full-dimensional vectors.",
      umap_caveat: "UMAP emphasizes neighbourhood structure while retaining broader embedding structure. Distances in this view are approximate and do not determine retrieval. Semantic similarity uses the original full-dimensional vectors.",
      umap_pca_fallback: "The sample is too small for a stable UMAP graph, so PCA coordinates are shown.",
      embedding_graph: "Interactive 3D embedding graph",
      visualization_help: "Drag freely to rotate · Use the wheel or +/− to zoom · Hover for a label · Click to inspect",
      embedding_controls: "3D embedding view controls",
      zoom_out_label: "Zoom out of embedding space",
      zoom_out: "Zoom out",
      zoom_in_label: "Zoom into embedding space",
      zoom_in: "Zoom in",
      current_zoom: "Current embedding space zoom: {percentage}%",
      reset_view: "Reset view",
      reset_view_label: "Reset embedding space view",
      graph_display_mode: "Embedding graph display mode",
      expand: "Expand",
      expand_graph: "Expand embedding graph",
      restore: "Restore",
      restore_graph: "Restore embedded graph",
      close: "Close",
      close_graph: "Close expanded graph",
      chunk_details: "Chunk details",
      close_chunk_details: "Close chunk details",
      embedding_legend: "Embedding map legend",
      query_vector: "Query vector",
      retrieved_chunks: "Retrieved chunks",
      other_chunks: "Other indexed chunks",
      query_retrieved_link: "Query → retrieved link",
      search: "Search",
      semantic_vector_search: "Semantic vector search",
      dense_similarity: "Dense similarity",
      prompt: "Prompt",
      prompt_sent: "Prompt sent to the model",
      chat_messages: "Application-level chat messages",
      answer: "Answer",
      generated_answer: "Generated answer",
      grounding_guard: "Grounding guard",
      grounded_answer_status: "Grounded answer status",
      learn2rag_website: "Learn2RAG website",
      visit_bmftr: "Visit the BMFTR website",
      bmftr_logo_alt: "BMFTR funding logo",
      examples_unavailable: "Example questions are temporarily unavailable.",
      chunk_order_note: "Chunk numbers indicate display order, which may differ from the original document order.",
      chunk_number: "Chunk {number}",
      chunk_shortened: "This chunk's text has been shortened for display.",
      preview_limited: "This is a limited preview. Some chunks or text may be omitted.",
      show_chunks: "Show chunks",
      hide_chunks: "Hide chunks",
      show_chunks_for: "Show chunks for {name}",
      hide_chunks_for: "Hide chunks for {name}",
      chunks_from: "Chunks from {name}",
      loading_chunks: "Loading chunks…",
      chunks_unavailable: "These chunks could not be loaded. Hide and show chunks to try again.",
      one_chunk: "{count} chunk",
      many_chunks: "{count} chunks",
      empty_index: "The index is currently empty. Documents will appear here after they are processed.",
      index_unavailable: "The index could not be loaded.",
      asking: "Asking…",
      no_keyword_chunks: "No matching keyword chunks were found.",
      no_chunks: "No matching chunks were retrieved.",
      retrieved_rank: "Retrieved · Rank #{rank}",
      not_retrieved: "Not retrieved for this question",
      semantic_similarity: "Semantic similarity",
      source: "Source",
      unknown_source: "Unknown source",
      chunk_content: "Chunk content",
      loading_chunk: "Loading chunk…",
      chunk_load_error: "This chunk could not be loaded.",
      pca_coordinates: "PCA coordinates",
      tsne_coordinates: "t-SNE coordinates",
      umap_coordinates: "UMAP coordinates",
      query: "Query",
      indexed_chunk: "Indexed chunk",
      similarity: "Similarity {value}",
      interactive_projection: "Interactive three-dimensional projection of indexed chunks and the Query",
      retrieved_point_title: "Retrieved rank {rank}: {source}; similarity {similarity}",
      indexed_point_title: "Indexed chunk: {source}; similarity {similarity}",
      retrieved_point_label: "Retrieved rank {rank}, {source}, similarity {similarity}. Press Enter to pin this point.",
      indexed_point_label: "Indexed chunk, {source}, similarity {similarity}. Press Enter to pin this point.",
      query_endpoint_label: "Query vector endpoint. Press Enter to pin its projection coordinates.",
      query_coordinates: "Query — X {x}, Y {y}, Z {z}",
      enter_question: "Please enter a question.",
      query_failed: "The RAG query could not be completed.",
      hybrid_unavailable: "Hybrid search is unavailable for the current index.",
      keyword_note: "Keyword search scores the indexed text with demo-local BM25.",
      hybrid_note: "Hybrid ranking combines a dense semantic representation and a sparse lexical representation with native Qdrant fusion. The 3D embedding explorer is shown only for Semantic vectors.",
      bm25_ranking: "BM25 lexical ranking",
      bm25_score: "BM25 score",
      fusion_score: "Fusion score",
      exact_messages_note: "These are the exact application-level chat messages passed by Learn2RAG to the configured model. Provider-specific serialization or chat templates are not shown.",
      partial_visualization: "The bounded display shows a partial index snapshot.",
      dense_only_map: "This embedding map is available for dense retrieval only.",
      map_unavailable: "The embedding map is temporarily unavailable.",
      no_dense_points: "No dense embedding points are available to display.",
      no_usable_dense_points: "No usable dense embedding points are available to display.",
      projection_failed: "The dense embedding points could not be projected safely.",
      semantic_map_only: "The embedding explorer is available for semantic vector search.",
      hybrid_map_only: "Hybrid ranking combines dense semantic and sparse lexical signals. The 3D embedding explorer is shown only for Semantic vectors.",
      role_system: "SYSTEM",
      role_human: "HUMAN",
    }),
    de: Object.freeze({
      page_title: "Learn2RAG · Interaktive RAG-Demo",
      navigation_label: "Navigation der RAG-Demo",
      demo_tagline: "Interaktive RAG-Demo",
      demo_sections: "Demo-Bereiche",
      index: "Index",
      ask_rag: "RAG fragen",
      language_selector: "Sprache der Benutzeroberfläche",
      use_english: "Englisch verwenden",
      use_german: "Deutsch verwenden",
      indexed_documents: "Indexierte Dokumente",
      indexed_documents_intro: "Diese Dokumente wurden bereits verarbeitet und im RAG-Index gespeichert.",
      refresh: "Aktualisieren",
      loading_index: "Indexierte Dokumente werden geladen…",
      index_summary: "Indexübersicht",
      documents: "Dokumente",
      chunks: "Chunks",
      partial_index_note: "Das Sicherheitslimit für die Abfrage wurde erreicht. Die Zahlen zeigen daher nur einen Ausschnitt des Index.",
      ask_intro: "Stellen Sie eine Frage und vergleichen Sie anschließend vier bereits vorbereitete Antworten auf dieselbe Frage.",
      question: "Frage",
      question_placeholder: "Was möchten Sie wissen?",
      retrieval_method: "Abrufmethode",
      semantic_vectors: "Semantische Vektoren",
      semantic_description: "Suche nach Bedeutung, nicht nur nach exakten Wörtern.",
      keyword_search: "Stichwortsuche",
      keyword_description: "Findet die eingegebenen Wörter.",
      hybrid_search: "Hybride Suche",
      hybrid_description: "Kombiniert semantischen und spärlichen lexikalischen Abruf.",
      question_prepares_all: "Eine Frage bereitet Ergebnisse für Semantik, Stichwortsuche, Hybrid und Generative KI vor.",
      indexed_documents_only: "Fragen werden anhand der aktuell indexierten Dokumente beantwortet.",
      example_questions: "Beispielfragen",
      try_example: "Beispiel ausprobieren",
      loading_examples: "Beispiele werden geladen…",
      suggest_examples: "Weitere Beispiele vorschlagen",
      query_loading: "Ergebnisse für Semantik, Stichwortsuche, Hybrid und Generative KI werden vorbereitet…",
      compare_loaded_results: "Ergebnisse vergleichen",
      result_mode: "Ergebnismodus",
      semantic_mode: "Semantisch",
      keyword_mode: "Stichwort",
      hybrid_mode: "Hybrid",
      generative_ai: "Generative KI",
      generative_card_description: "Antwort ohne abgerufene Dokumente.",
      semantic_card_description: "Bedeutungsbasierter dichter Abruf.",
      keyword_card_description: "Stichwortbasierter BM25-Abruf.",
      hybrid_card_description: "Kombiniert dichten und spärlichen Abruf.",
      semantic_mode_description: "Antwort auf Grundlage dichter semantischer Suche.",
      keyword_mode_description: "Antwort auf Grundlage der BM25-Stichwortsuche.",
      hybrid_mode_description: "Antwort auf Grundlage nativer dichter und spärlicher Fusion.",
      generative_mode_description: "Direkte Antwort des Sprachmodells ohne abgerufene Dokumente.",
      receive_question: "Frage empfangen",
      no_retrieval: "Kein Abruf",
      build_direct_prompt: "Direkten Prompt erstellen",
      no_indexed_documents: "Keine indexierten Dokumente verwendet.",
      embed_question: "Frage einbetten",
      retrieve_chunks: "Bestplatzierte Chunks abrufen",
      add_chunks_prompt: "Chunks zum Prompt hinzufügen",
      generate_answer: "Antwort erzeugen",
      explore_embedding_space: "Embedding-Raum erkunden",
      projection: "Projektion:",
      projection_selector: "Projektionsmethode",
      pca_projection: "3D-PCA-Projektion dichter Embeddings",
      tsne_projection: "3D-t-SNE-Projektion dichter Embeddings",
      umap_projection: "3D-UMAP-Projektion dichter Embeddings",
      visualization_intro: "Alle indexierten Chunks bilden den umgebenden Embedding-Raum. Der Fragevektor und seine Verbindungen kennzeichnen die für den RAG-Kontext ausgewählten Chunks.",
      pca_caveat: "Die räumliche Nähe ist nur näherungsweise, da PCA den ursprünglichen Embedding-Raum komprimiert. Die semantische Ähnlichkeit wird mit den ursprünglichen, vollständigen Vektoren berechnet.",
      tsne_caveat: "t-SNE betont lokale Nachbarschaftsstrukturen. Abstände in dieser Darstellung sind Näherungen und bestimmen nicht den Abruf. Die semantische Ähnlichkeit wird mit den ursprünglichen, vollständigen Vektoren berechnet.",
      umap_caveat: "UMAP betont Nachbarschaftsstrukturen und bewahrt zugleich übergreifende Embedding-Strukturen. Abstände in dieser Ansicht sind Näherungen und bestimmen nicht den Abruf. Die semantische Ähnlichkeit nutzt die ursprünglichen, vollständigen Vektoren.",
      umap_pca_fallback: "Die Stichprobe ist für einen stabilen UMAP-Graphen zu klein; daher werden PCA-Koordinaten angezeigt.",
      embedding_graph: "Interaktiver 3D-Embedding-Graph",
      visualization_help: "Ziehen zum freien Drehen · Mausrad oder +/− zum Zoomen · Zeigen für Beschriftung · Klicken zum Prüfen",
      embedding_controls: "Steuerelemente der 3D-Embedding-Ansicht",
      zoom_out_label: "Aus dem Embedding-Raum herauszoomen",
      zoom_out: "Verkleinern",
      zoom_in_label: "In den Embedding-Raum hineinzoomen",
      zoom_in: "Vergrößern",
      current_zoom: "Aktueller Zoom des Embedding-Raums: {percentage}%",
      reset_view: "Ansicht zurücksetzen",
      reset_view_label: "Ansicht des Embedding-Raums zurücksetzen",
      graph_display_mode: "Anzeigemodus des Embedding-Graphen",
      expand: "Vergrößern",
      expand_graph: "Embedding-Graph vergrößern",
      restore: "Einbetten",
      restore_graph: "Graph wieder einbetten",
      close: "Schließen",
      close_graph: "Vergrößerten Graphen schließen",
      chunk_details: "Chunk-Details",
      close_chunk_details: "Chunk-Details schließen",
      embedding_legend: "Legende der Embedding-Karte",
      query_vector: "Fragevektor",
      retrieved_chunks: "Abgerufene Chunks",
      other_chunks: "Andere indexierte Chunks",
      query_retrieved_link: "Frage → abgerufener Chunk",
      search: "Suche",
      semantic_vector_search: "Semantische Vektorsuche",
      dense_similarity: "Dichte Ähnlichkeit",
      prompt: "Prompt",
      prompt_sent: "An das Modell gesendeter Prompt",
      chat_messages: "Chat-Nachrichten auf Anwendungsebene",
      answer: "Antwort",
      generated_answer: "Erzeugte Antwort",
      grounding_guard: "Belegsicherung",
      grounded_answer_status: "Status der belegten Antwort",
      learn2rag_website: "Learn2RAG-Webseite",
      visit_bmftr: "BMFTR-Webseite besuchen",
      bmftr_logo_alt: "Förderlogo des BMFTR",
      examples_unavailable: "Beispielfragen sind vorübergehend nicht verfügbar.",
      chunk_order_note: "Chunk-Nummern geben die Anzeigereihenfolge an, die von der Reihenfolge im Originaldokument abweichen kann.",
      chunk_number: "Chunk {number}",
      chunk_shortened: "Der Text dieses Chunks wurde für die Anzeige gekürzt.",
      preview_limited: "Dies ist eine begrenzte Vorschau. Einige Chunks oder Texte können fehlen.",
      show_chunks: "Chunks anzeigen",
      hide_chunks: "Chunks ausblenden",
      show_chunks_for: "Chunks für {name} anzeigen",
      hide_chunks_for: "Chunks für {name} ausblenden",
      chunks_from: "Chunks aus {name}",
      loading_chunks: "Chunks werden geladen…",
      chunks_unavailable: "Diese Chunks konnten nicht geladen werden. Blenden Sie die Chunks aus und wieder ein, um es erneut zu versuchen.",
      one_chunk: "{count} Chunk",
      many_chunks: "{count} Chunks",
      empty_index: "Der Index ist derzeit leer. Dokumente erscheinen hier, nachdem sie verarbeitet wurden.",
      index_unavailable: "Der Index konnte nicht geladen werden.",
      asking: "Anfrage läuft…",
      no_keyword_chunks: "Es wurden keine passenden Stichwort-Chunks gefunden.",
      no_chunks: "Es wurden keine passenden Chunks abgerufen.",
      retrieved_rank: "Abgerufen · Rang #{rank}",
      not_retrieved: "Für diese Frage nicht abgerufen",
      semantic_similarity: "Semantische Ähnlichkeit",
      source: "Quelle",
      unknown_source: "Unbekannte Quelle",
      chunk_content: "Chunk-Inhalt",
      loading_chunk: "Chunk wird geladen…",
      chunk_load_error: "Dieser Chunk konnte nicht geladen werden.",
      pca_coordinates: "PCA-Koordinaten",
      tsne_coordinates: "t-SNE-Koordinaten",
      umap_coordinates: "UMAP-Koordinaten",
      query: "Frage",
      indexed_chunk: "Indexierter Chunk",
      similarity: "Ähnlichkeit {value}",
      interactive_projection: "Interaktive dreidimensionale Projektion indexierter Chunks und der Frage",
      retrieved_point_title: "Abgerufener Rang {rank}: {source}; Ähnlichkeit {similarity}",
      indexed_point_title: "Indexierter Chunk: {source}; Ähnlichkeit {similarity}",
      retrieved_point_label: "Abgerufener Rang {rank}, {source}, Ähnlichkeit {similarity}. Eingabetaste drücken, um diesen Punkt anzuheften.",
      indexed_point_label: "Indexierter Chunk, {source}, Ähnlichkeit {similarity}. Eingabetaste drücken, um diesen Punkt anzuheften.",
      query_endpoint_label: "Endpunkt des Fragevektors. Eingabetaste drücken, um seine Projektionskoordinaten anzuheften.",
      query_coordinates: "Frage — X {x}, Y {y}, Z {z}",
      enter_question: "Bitte geben Sie eine Frage ein.",
      query_failed: "Die RAG-Anfrage konnte nicht abgeschlossen werden.",
      hybrid_unavailable: "Die hybride Suche ist für den aktuellen Index nicht verfügbar.",
      keyword_note: "Die Stichwortsuche bewertet den indexierten Text mit demo-lokalem BM25.",
      hybrid_note: "Das hybride Ranking kombiniert eine dichte semantische und eine spärliche lexikalische Darstellung mit nativer Qdrant-Fusion. Der 3D-Embedding-Explorer wird nur für semantische Vektoren angezeigt.",
      bm25_ranking: "Lexikalisches BM25-Ranking",
      bm25_score: "BM25-Wert",
      fusion_score: "Fusionswert",
      exact_messages_note: "Dies sind die exakten Chat-Nachrichten auf Anwendungsebene, die Learn2RAG an das konfigurierte Modell übergibt. Anbieterspezifische Serialisierung oder Chat-Vorlagen werden nicht angezeigt.",
      partial_visualization: "Die begrenzte Anzeige zeigt einen Ausschnitt des Index.",
      dense_only_map: "Diese Embedding-Karte ist nur für dichten Abruf verfügbar.",
      map_unavailable: "Die Embedding-Karte ist vorübergehend nicht verfügbar.",
      no_dense_points: "Es sind keine dichten Embedding-Punkte zur Anzeige verfügbar.",
      no_usable_dense_points: "Es sind keine verwendbaren dichten Embedding-Punkte zur Anzeige verfügbar.",
      projection_failed: "Die dichten Embedding-Punkte konnten nicht sicher projiziert werden.",
      semantic_map_only: "Der Embedding-Explorer ist für die semantische Vektorsuche verfügbar.",
      hybrid_map_only: "Das hybride Ranking kombiniert dichte semantische und spärliche lexikalische Signale. Der 3D-Embedding-Explorer wird nur für semantische Vektoren angezeigt.",
      role_system: "SYSTEM",
      role_human: "MENSCH",
    }),
  });

  const serverTranslationKeys = Object.freeze({
    "Semantic vector search": "semantic_vector_search",
    "Keyword search": "keyword_search",
    "Hybrid search": "hybrid_search",
    "BM25 lexical ranking": "bm25_ranking",
    "BM25 score": "bm25_score",
    "Fusion score": "fusion_score",
    "Semantic similarity": "semantic_similarity",
    "Explore the embedding space": "explore_embedding_space",
    "3D PCA projection of dense embeddings": "pca_projection",
    "Prompt sent to the model": "prompt_sent",
    "Application-level chat messages": "chat_messages",
    "These are the exact application-level chat messages passed by Learn2RAG to the configured model. Provider-specific serialization or chat templates are not shown.": "exact_messages_note",
    "The bounded display shows a partial index snapshot.": "partial_visualization",
    "This embedding map is available for dense retrieval only.": "dense_only_map",
    "The embedding map is temporarily unavailable.": "map_unavailable",
    "No dense embedding points are available to display.": "no_dense_points",
    "No usable dense embedding points are available to display.": "no_usable_dense_points",
    "The dense embedding points could not be projected safely.": "projection_failed",
    "The embedding explorer is available for semantic vector search.": "semantic_map_only",
    "Hybrid ranking combines dense semantic and sparse lexical signals. The 3D embedding explorer is shown only for Semantic vectors.": "hybrid_map_only",
    "The RAG index is temporarily unavailable. Please try again shortly.": "index_unavailable",
    "The RAG query is temporarily unavailable. Please try again shortly.": "query_failed",
    "Hybrid search is unavailable for the current index.": "hybrid_unavailable",
    "The index could not be loaded.": "index_unavailable",
  });
  const localizedTextBindings = new Map();
  const localizedAttributeBindings = [];
  let currentLanguage = "en";

  function t(key, parameters = {}) {
    const template = translations[currentLanguage][key] || translations.en[key] || key;
    return template.replace(/\{(\w+)\}/g, (match, name) => (
      Object.prototype.hasOwnProperty.call(parameters, name) ? String(parameters[name]) : match
    ));
  }

  function translatedServerValue(value) {
    const text = String(value || "");
    const exactKey = serverTranslationKeys[text];
    if (exactKey) return t(exactKey);
    if (text.startsWith("3D PCA projection of dense embeddings / ")) {
      return `${t("pca_projection")} / ${text.slice("3D PCA projection of dense embeddings / ".length)}`;
    }
    if (currentLanguage === "de") {
      if (text === "Dense") return "Dicht";
      if (text.startsWith("Dense / ")) return `Dicht / ${text.slice("Dense / ".length)}`;
      return text.replace(" dense + sparse", " dicht + spärlich");
    }
    return text;
  }

  function setLocalizedText(element, key, parameters = {}) {
    const render = () => t(key, parameters);
    element.textContent = render();
    localizedTextBindings.set(element, render);
  }

  function setTranslatedText(element, value) {
    const render = () => translatedServerValue(value);
    element.textContent = render();
    localizedTextBindings.set(element, render);
  }

  function setLocalizedAttribute(element, attribute, key, parameters = {}) {
    const binding = { element, attribute, key, parameters };
    element.setAttribute(attribute, t(key, parameters));
    localizedAttributeBindings.push(binding);
  }

  function readStoredLanguage() {
    try {
      const stored = typeof localStorage === "undefined" ? null : localStorage.getItem(languageStorageKey);
      return stored === "de" || stored === "en" ? stored : "en";
    } catch (error) {
      return "en";
    }
  }

  function applyStaticTranslations() {
    document.querySelectorAll("[data-i18n]").forEach((element) => {
      element.textContent = t(element.dataset.i18n);
    });
    ["aria-label", "placeholder", "title", "alt"].forEach((attribute) => {
      const datasetName = `i18n${attribute.split("-").map((part) => (
        part.charAt(0).toUpperCase() + part.slice(1)
      )).join("")}`;
      document.querySelectorAll(`[data-i18n-${attribute}]`).forEach((element) => {
        element.setAttribute(attribute, t(element.dataset[datasetName]));
      });
    });
  }

  function setLanguage(language, persist = true) {
    currentLanguage = language === "de" ? "de" : "en";
    if (document.documentElement) document.documentElement.lang = currentLanguage;
    applyStaticTranslations();
    localizedTextBindings.forEach((render, element) => {
      element.textContent = render();
    });
    localizedAttributeBindings.forEach(({ element, attribute, key, parameters }) => {
      element.setAttribute(attribute, t(key, parameters));
    });
    document.querySelectorAll("[data-language]").forEach((button) => {
      const active = button.dataset.language === currentLanguage;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    if (persist) {
      try {
        if (typeof localStorage !== "undefined") localStorage.setItem(languageStorageKey, currentLanguage);
      } catch (error) {
        // The interface still switches when storage is unavailable.
      }
    }
  }

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
  const comparisonActions = document.querySelector("#comparison-actions");
  const comparisonButtons = document.querySelector("#comparison-buttons");
  const generativeResult = document.querySelector("#generative-result");
  const directAnswerContent = document.querySelector("#direct-answer-content");
  const ragStages = document.querySelector("#rag-stages");
  const embedStage = document.querySelector("#embed-stage");
  const retrieveStage = document.querySelector("#retrieve-stage");
  const promptStage = document.querySelector("#prompt-stage");
  const answerStage = document.querySelector("#answer-stage");
  const embedStageLabel = document.querySelector("#embed-stage-label");
  const retrieveStageLabel = document.querySelector("#retrieve-stage-label");
  const promptStageLabel = document.querySelector("#prompt-stage-label");
  const answerStageLabel = document.querySelector("#answer-stage-label");
  const embeddedQuestion = document.querySelector("#embedded-question");
  const embeddedRetrievalMethod = document.querySelector("#embedded-retrieval-method");
  const nonsemanticRetrievalNote = document.querySelector("#nonsemantic-retrieval-note");
  const directRetrievalNote = document.querySelector("#direct-retrieval-note");
  const searchContent = document.querySelector("#search-content");
  const answerKicker = document.querySelector("#answer-kicker");
  const answerHeading = document.querySelector("#answer-heading");
  const answerContent = document.querySelector("#answer-content");
  const searchLabel = document.querySelector("#search-label");
  const searchTechnicalLabel = document.querySelector("#search-technical-label");
  const scoreLabel = document.querySelector("#score-label");
  const searchResults = document.querySelector("#search-results");
  const visualizationLabel = document.querySelector("#visualization-label");
  const visualizationTechnicalLabel = document.querySelector("#visualization-technical-label");
  const projectionExplanation = document.querySelector("#projection-explanation");
  const pcaProjectionButton = document.querySelector("#projection-pca");
  const tsneProjectionButton = document.querySelector("#projection-tsne");
  const umapProjectionButton = document.querySelector("#projection-umap");
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
  const queryLegendSymbol = document.querySelector("#query-legend-symbol");
  const queryLegendLabel = document.querySelector("#query-legend-label");
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
  const chunkDetailsCache = new Map();
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
  const pointerDragThreshold = 5;
  const scenePadding = 12;
  const redundantVisualizationNote =
    "3D proximity is approximate. Semantic similarity is calculated in the original full-dimensional embedding space used for retrieval.";
  // Screen-space labels and markers share this edge reserve; it is not added
  // to the rotation-safe world-space radius.
  const maximumMarkerExtent = 30;
  let viewerState = null;
  let currentVisualization = null;
  let currentVisualizationSearchMode = null;
  let currentProjectionMethod = "pca";
  let hoveredChunkId = null;
  let focusedChunkId = null;
  let pinnedChunkId = null;
  let pinnedQuery = false;
  let inspectorTrigger = null;
  let chunkInspectorLoadingId = null;
  let chunkInspectorErrorId = null;
  let inspectorRequestVersion = 0;
  let comparisonResultSet = null;
  let activeComparisonMode = null;
  let queryRequestVersion = 0;

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

  function showState(message, isError = false, isTranslationKey = false) {
    if (isTranslationKey) {
      setLocalizedText(statusBox, message);
    } else {
      setTranslatedText(statusBox, message);
    }
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
      button.disabled = questionInput.disabled;
      button.textContent = question;
      button.addEventListener("click", () => {
        questionInput.value = question;
        questionInput.focus();
      });
      exampleContainer.appendChild(button);
    });
  }

  async function loadExamples() {
    refreshExamplesButton.disabled = true;
    setLocalizedText(examplesStatus, "loading_examples");
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
      setLocalizedText(examplesStatus, "examples_unavailable");
      examplesStatus.hidden = false;
    } finally {
      refreshExamplesButton.disabled = false;
    }
  }

  function renderDocumentChunks(container, data) {
    localizedTextBindings.delete(container);
    container.replaceChildren();
    const orderNote = document.createElement("p");
    orderNote.className = "small text-body-secondary";
    setLocalizedText(orderNote, "chunk_order_note");
    container.appendChild(orderNote);
    data.chunks.forEach((chunk) => {
      const item = document.createElement("article");
      item.className = "indexed-chunk";
      item.dataset.chunkId = chunk.id;
      const heading = document.createElement("h4");
      heading.className = "h6";
      setLocalizedText(heading, "chunk_number", { number: chunk.display_order });
      const text = document.createElement("p");
      text.className = "indexed-chunk-content mb-0";
      text.textContent = chunk.content;
      item.append(heading, text);
      if (chunk.truncated) {
        const note = document.createElement("p");
        note.className = "small text-body-secondary mt-2 mb-0";
        setLocalizedText(note, "chunk_shortened");
        item.appendChild(note);
      }
      container.appendChild(item);
    });
    if (data.truncated) {
      const note = document.createElement("p");
      note.className = "small text-body-secondary mb-0";
      setLocalizedText(note, "preview_limited");
      container.appendChild(note);
    }
  }

  function addChunkDisclosure(cardBody, indexedDocument) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "btn btn-sm btn-outline-secondary mt-3";
    setLocalizedText(button, "show_chunks");
    button.setAttribute("aria-expanded", "false");
    setLocalizedAttribute(button, "aria-label", "show_chunks_for", { name: indexedDocument.name });
    const container = document.createElement("div");
    container.id = `document-chunks-${indexedDocument.id}`;
    container.className = "document-chunks mt-3";
    container.hidden = true;
    container.setAttribute("role", "region");
    setLocalizedAttribute(container, "aria-label", "chunks_from", { name: indexedDocument.name });
    button.setAttribute("aria-controls", container.id);
    let loaded = false;
    let loading = false;
    button.addEventListener("click", async () => {
      if (loading) return;
      const expanding = container.hidden;
      container.hidden = !expanding;
      button.setAttribute("aria-expanded", String(expanding));
      setLocalizedText(button, expanding ? "hide_chunks" : "show_chunks");
      setLocalizedAttribute(
        button,
        "aria-label",
        expanding ? "hide_chunks_for" : "show_chunks_for",
        { name: indexedDocument.name },
      );
      if (!expanding || loaded) return;
      loading = true;
      button.disabled = true;
      container.setAttribute("aria-busy", "true");
      setLocalizedText(container, "loading_chunks");
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
        setLocalizedText(container, "chunks_unavailable");
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
      setLocalizedText(emptyState, "empty_index");
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
      setLocalizedText(
        chunks,
        indexedDocument.chunk_count === 1 ? "one_chunk" : "many_chunks",
        { count: indexedDocument.chunk_count },
      );
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
    showState("loading_index", false, true);
    refreshButton.disabled = true;
    try {
      const response = await fetch("./api/index", { headers: { Accept: "application/json" } });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.message || "The index could not be loaded.");
      }
      renderDocuments(data);
    } catch (error) {
      if (error instanceof Error) {
        showState(error.message, true);
      } else {
        showState("index_unavailable", true, true);
      }
    } finally {
      refreshButton.disabled = false;
    }
  }

  function setQueryLoading(isLoading) {
    queryForm.setAttribute("aria-busy", String(isLoading));
    questionInput.disabled = isLoading;
    askButton.disabled = isLoading;
    exampleContainer.querySelectorAll(".example-question").forEach((button) => { button.disabled = isLoading; });
    setLocalizedText(askButton, isLoading ? "asking" : "ask_rag");
    queryLoading.hidden = !isLoading;
  }

  function showQueryError(message) {
    setTranslatedText(queryError, message);
    queryError.hidden = false;
    queryResults.hidden = true;
  }

  function clearComparisonState() {
    comparisonResultSet = null;
    activeComparisonMode = null;
    queryResults.hidden = true;
    comparisonActions.hidden = true;
    comparisonButtons.replaceChildren();
    generativeResult.hidden = true;
    directAnswerContent.replaceChildren();
    ragStages.hidden = false;
  }

  function deepFreeze(value) {
    if (!value || typeof value !== "object" || Object.isFrozen(value)) return value;
    Object.freeze(value);
    Object.values(value).forEach(deepFreeze);
    return value;
  }

  function configureComparisonSelector(results, selectedMode) {
    const labels = {
      semantic: "semantic_mode",
      keyword: "keyword_mode",
      hybrid: "hybrid_mode",
      generative: "generative_ai",
    };
    const descriptions = {
      generative: "generative_card_description",
      semantic: "semantic_card_description",
      keyword: "keyword_card_description",
      hybrid: "hybrid_card_description",
    };
    comparisonButtons.replaceChildren();
    ["generative", "semantic", "keyword", "hybrid"].forEach((mode) => {
      const envelope = results[mode];
      const active = mode === selectedMode;
      const button = document.createElement("button");
      button.type = "button";
      button.className = `comparison-card${active ? " is-selected" : ""}`;
      button.dataset.comparisonMode = mode;
      button.disabled = !envelope || envelope.status !== "ready";
      button.setAttribute("aria-pressed", String(active));
      if (button.disabled && envelope && envelope.message) button.title = envelope.message;
      const title = document.createElement("span");
      title.className = "comparison-card-title";
      setLocalizedText(title, labels[mode]);
      const description = document.createElement("span");
      description.className = "comparison-card-description";
      setLocalizedText(description, descriptions[mode]);
      button.append(title, description);
      comparisonButtons.appendChild(button);
    });
    comparisonActions.hidden = false;
  }

  function setStagePresentation() {
    setLocalizedText(embedStageLabel, "embed_question");
    setLocalizedText(retrieveStageLabel, "retrieve_chunks");
    setLocalizedText(promptStageLabel, "add_chunks_prompt");
    setLocalizedText(answerStageLabel, "generate_answer");
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

  function renderSearchResults(results, mode, scoreLabelText) {
    searchResults.replaceChildren();
    searchCardByChunkId.clear();
    searchResultByChunkId.clear();
    if (results.length === 0) {
      const emptyState = document.createElement("p");
      emptyState.className = "alert alert-secondary mb-0";
      setLocalizedText(emptyState, mode === "keyword" ? "no_keyword_chunks" : "no_chunks");
      searchResults.appendChild(emptyState);
      return [];
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
      const scoreKey = mode === "keyword"
        ? "bm25_score"
        : mode === "hybrid" ? "fusion_score" : "semantic_similarity";
      setLocalizedText(score, scoreKey);
      score.textContent += `: ${Number(result.score).toFixed(3)}`;
      const scoreValue = Number(result.score).toFixed(3);
      localizedTextBindings.set(score, () => `${t(scoreKey)}: ${scoreValue}`);
      header.append(sourceGroup, score);

      const body = document.createElement("div");
      body.className = "card-body";
      const chunk = document.createElement("p");
      chunk.className = "chunk-content mb-0";
      body.appendChild(chunk);
      if (mode === "keyword" && result.matched_terms.length > 0) {
        appendKeywordHighlights(chunk, result.content, result.matched_terms);
      } else {
        chunk.textContent = result.content;
      }

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

  async function openChunkInspector(chunkId, trigger) {
    pinnedChunkId = chunkId;
    pinnedQuery = false;
    inspectorTrigger = trigger || visualizationPointByChunkId.get(chunkId) || null;
    chunkInspectorErrorId = null;
    const requestVersion = ++inspectorRequestVersion;
    if (chunkDetailsCache.has(chunkId)) {
      chunkInspectorLoadingId = null;
      updateLinkedInteraction();
      return;
    }
    chunkInspectorLoadingId = chunkId;
    updateLinkedInteraction();
    try {
      const response = await fetch(`./api/chunks/${encodeURIComponent(chunkId)}`, {
        headers: { Accept: "application/json" },
      });
      if (!response.ok) throw new Error("Chunk unavailable");
      const details = await response.json();
      chunkDetailsCache.set(chunkId, details);
      if (requestVersion === inspectorRequestVersion && pinnedChunkId === chunkId) {
        chunkInspectorLoadingId = null;
        updateLinkedInteraction(false);
      }
    } catch (error) {
      if (requestVersion === inspectorRequestVersion && pinnedChunkId === chunkId) {
        chunkInspectorLoadingId = null;
        chunkInspectorErrorId = chunkId;
        updateLinkedInteraction(false);
      }
    }
  }

  function togglePinnedQuery(trigger) {
    pinnedQuery = !pinnedQuery;
    pinnedChunkId = null;
    inspectorTrigger = trigger || null;
    chunkInspectorLoadingId = null;
    chunkInspectorErrorId = null;
    inspectorRequestVersion += 1;
    updateLinkedInteraction();
  }

  function clearPinnedSelection(restoreFocus = false) {
    const previousTrigger = inspectorTrigger;
    pinnedChunkId = null;
    pinnedQuery = false;
    inspectorTrigger = null;
    chunkInspectorLoadingId = null;
    chunkInspectorErrorId = null;
    inspectorRequestVersion += 1;
    updateLinkedInteraction();
    if (restoreFocus && previousTrigger && typeof previousTrigger.focus === "function") {
      previousTrigger.focus();
    }
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

  function appendInspectorField(container, labelKey, value, className = "", valueKey = null) {
    const field = document.createElement("div");
    field.className = "embedding-inspector-field";
    const heading = document.createElement("h4");
    heading.className = "embedding-inspector-label";
    setLocalizedText(heading, labelKey);
    const content = document.createElement("p");
    content.className = className;
    if (valueKey) {
      setLocalizedText(content, valueKey);
    } else {
      content.textContent = value;
    }
    field.append(heading, content);
    container.appendChild(field);
    return content;
  }

  function renderPinnedInspector() {
    visualizationInspectorContent.replaceChildren();
    if (!viewerState || (!pinnedQuery && !pinnedChunkId)) {
      visualizationInspector.hidden = true;
      setLocalizedText(visualizationInspectorTitle, "chunk_details");
      return;
    }

    if (pinnedQuery) {
      setLocalizedText(visualizationInspectorTitle, "query");
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
    const details = chunkDetailsCache.get(point.id);
    setLocalizedText(visualizationInspectorTitle, "chunk_details");
    const retrievalStatus = document.createElement("p");
    retrievalStatus.className = point.retrieved
      ? "embedding-inspector-status is-retrieved"
      : "embedding-inspector-status";
    setLocalizedText(
      retrievalStatus,
      point.retrieved ? "retrieved_rank" : "not_retrieved",
      point.retrieved ? { rank: point.rank } : {},
    );
    visualizationInspectorContent.appendChild(retrievalStatus);
    appendInspectorField(
      visualizationInspectorContent,
      "semantic_similarity",
      Number(point.similarity).toFixed(3),
      "embedding-inspector-score",
    );
    appendInspectorField(
      visualizationInspectorContent,
      "source",
      (details && details.source) || point.source || t("unknown_source"),
      "embedding-inspector-source",
    );

    if (chunkInspectorLoadingId === point.id) {
      appendInspectorField(
        visualizationInspectorContent,
        "chunk_content",
        "",
        "embedding-inspector-message",
        "loading_chunk",
      );
    } else if (chunkInspectorErrorId === point.id) {
      appendInspectorField(
        visualizationInspectorContent,
        "chunk_content",
        "",
        "embedding-inspector-message",
        "chunk_load_error",
      );
    } else if (details) {
      appendInspectorField(
        visualizationInspectorContent,
        "chunk_content",
        details.content,
        "embedding-inspector-chunk",
      );
      if (details.truncated) {
        const truncatedNote = document.createElement("p");
        truncatedNote.className = "small text-body-secondary";
        setLocalizedText(truncatedNote, "chunk_shortened");
        visualizationInspectorContent.appendChild(truncatedNote);
      }
    }
    const coordinateLabel = document.createElement("h4");
    coordinateLabel.className = "embedding-inspector-label";
    setLocalizedText(
      coordinateLabel,
      currentProjectionMethod === "tsne"
        ? "tsne_coordinates"
        : currentProjectionMethod === "umap" ? "umap_coordinates" : "pca_coordinates",
    );
    visualizationInspectorContent.appendChild(coordinateLabel);
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
      : point.source || t("indexed_chunk");
    visualizationTooltip.appendChild(heading);
    const similarity = document.createElement("span");
    similarity.className = "d-block";
    setLocalizedText(similarity, "similarity", {
      value: Number(point.similarity).toFixed(3),
    });
    visualizationTooltip.appendChild(similarity);
    appendHoverCoordinates(visualizationTooltip, point);
    visualizationTooltip.hidden = false;
    positionPointTooltip(projected, state);
  }

  function renderQueryHoverLabel(query, projected, state) {
    visualizationTooltip.replaceChildren();
    const heading = document.createElement("strong");
    heading.className = "d-block";
    setLocalizedText(heading, "query");
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

    const queryScale = 0.9 + projectedQuery.depthRatio * 0.25;
    if (state.showsLinearOrigin) {
      state.origin.visual.group.setAttribute(
        "transform",
        `translate(${projectedOrigin.x} ${projectedOrigin.y})`,
      );
      const projectedVectorEnd = shortenProjectedLineEnd(
        projectedOrigin,
        projectedQuery,
        24 * queryScale,
      );
      setProjectedLine(state.query.visual.vector, projectedOrigin, projectedVectorEnd);
      state.query.visual.vector.style.opacity = String(0.8 + projectedQuery.depthRatio * 0.2);
    }

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
    const tooltipId = activeId || state.hoveredScenePointId;
    if (state.queryHovered || (state.queryFocused && !tooltipId)) {
      renderQueryHoverLabel(state.query.data, projectedQuery, state);
    } else {
      const tooltipPoint = state.points.find((entry) => entry.data.id === tooltipId);
      const projectedTooltipPoint = tooltipId ? projectedById.get(tooltipId) : null;
      if (tooltipPoint && projectedTooltipPoint) {
        renderChunkHoverLabel(tooltipPoint.data, projectedTooltipPoint, state);
      } else {
        clearPointTooltip();
      }
    }
  }

  function closestSvgGroup(target, boundary, className) {
    let element = target;
    while (element && element !== boundary) {
      if (element.classList && element.classList.contains(className)) {
        return element;
      }
      element = element.parentElement || element.parentNode;
    }
    return null;
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
      state.pointerStartClientX = event.clientX;
      state.pointerStartClientY = event.clientY;
      state.lastTrackballVector = trackballVector(event, svg);
      state.dragMoved = false;
      state.pressedChunkElement = closestSvgGroup(event.target, svg, "embedding-point");
      state.pressedChunkId = state.pressedChunkElement
        ? state.pressedChunkElement.dataset.chunkId
        : null;
      state.pressedQuery = Boolean(
        closestSvgGroup(event.target, svg, "embedding-query-point"),
      );
      svg.classList.add("is-dragging");
      event.preventDefault();
    });
    svg.addEventListener("pointermove", (event) => {
      if (state.pointerId !== event.pointerId) {
        return;
      }
      const screenMovement = Math.hypot(
        event.clientX - state.pointerStartClientX,
        event.clientY - state.pointerStartClientY,
      );
      if (!state.dragMoved && screenMovement > pointerDragThreshold) {
        state.dragMoved = true;
        if (!svg.hasPointerCapture(event.pointerId)) {
          svg.setPointerCapture(event.pointerId);
        }
      }
      if (!state.dragMoved) {
        event.preventDefault();
        return;
      }
      const nextTrackballVector = trackballVector(event, svg);
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
      state.pointerStartClientX = null;
      state.pointerStartClientY = null;
      state.lastTrackballVector = null;
      state.pressedChunkId = null;
      state.pressedChunkElement = null;
      state.pressedQuery = false;
      svg.classList.remove("is-dragging");
    };
    svg.addEventListener("pointerup", async (event) => {
      if (state.pointerId !== event.pointerId) {
        return;
      }
      const shouldInspectChunk = !state.dragMoved && state.pressedChunkId;
      const pressedChunkId = state.pressedChunkId;
      const pressedChunkElement = state.pressedChunkElement;
      const shouldClearSelection = !state.dragMoved
        && !state.pressedChunkId
        && !state.pressedQuery;
      endDrag(event);
      if (shouldInspectChunk) {
        await openChunkInspector(pressedChunkId, pressedChunkElement);
      } else if (shouldClearSelection) {
        const activeElement = document.activeElement;
        if (activeElement && svg.contains(activeElement) && typeof activeElement.blur === "function") {
          activeElement.blur();
        }
        focusedChunkId = null;
        pinnedChunkId = null;
        pinnedQuery = false;
        updateLinkedInteraction();
      }
    });
    svg.addEventListener("pointercancel", endDrag);
    svg.addEventListener("lostpointercapture", endDrag);
  }

  function updateZoomControls(state) {
    const percentage = Math.round(state.zoom * 100);
    zoomLevel.textContent = `${percentage}%`;
    setLocalizedAttribute(zoomLevel, "aria-label", "current_zoom", { percentage });
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
    state.pointerStartClientX = null;
    state.pointerStartClientY = null;
    state.lastTrackballVector = null;
    state.dragMoved = false;
    state.pressedChunkId = null;
    state.pressedChunkElement = null;
    state.pressedQuery = false;
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

  function selectVisualizationProjection(visualization, requestedMethod) {
    if (requestedMethod === "pca") {
      const pcaProjection = Array.isArray(visualization.projections)
        ? visualization.projections.find((projection) => projection.method === "pca")
        : null;
      return pcaProjection
        ? {...visualization, points: mergeProjectionPoints(visualization.points, pcaProjection), query: pcaProjection.query}
        : visualization;
    }
    if (!["tsne", "umap"].includes(requestedMethod) || !Array.isArray(visualization.projections)) return null;
    const projection = visualization.projections.find((candidate) => candidate.method === requestedMethod);
    if (!projection) return null;
    return {
      ...visualization,
      points: mergeProjectionPoints(visualization.points, projection),
      query: projection.query,
    };
  }

  function mergeProjectionPoints(points, projection) {
    const coordinatesById = new Map(projection.points.map((point) => [point.id, point]));
    if (coordinatesById.size !== points.length || points.some((point) => !coordinatesById.has(point.id))) {
      return [];
    }
    return points.map((point) => ({...point, ...coordinatesById.get(point.id)}));
  }

  function setProjectionControls(method, visualization) {
    const availableMethods = new Set(
      Array.isArray(visualization.projections)
        ? visualization.projections.map((projection) => projection.method)
        : ["pca"],
    );
    [
      [pcaProjectionButton, "pca"],
      [tsneProjectionButton, "tsne"],
      [umapProjectionButton, "umap"],
    ].forEach(([button, value]) => {
      const active = value === method;
      button.disabled = !availableMethods.has(value);
      button.classList.toggle("btn-primary", active);
      button.classList.toggle("btn-outline-primary", !active);
      button.setAttribute("aria-pressed", String(active));
    });
    const labelKeys = {pca: "pca_projection", tsne: "tsne_projection", umap: "umap_projection"};
    const explanationKeys = {pca: "pca_caveat", tsne: "tsne_caveat", umap: "umap_caveat"};
    const selectedProjection = Array.isArray(visualization.projections)
      ? visualization.projections.find((projection) => projection.method === method)
      : null;
    const sourceLabel = String(visualization.technical_label || "");
    const separator = sourceLabel.indexOf(" / ");
    const suffix = separator >= 0 ? sourceLabel.slice(separator) : "";
    const renderLabel = () => `${t(labelKeys[method])}${suffix}`;
    visualizationTechnicalLabel.textContent = renderLabel();
    localizedTextBindings.set(visualizationTechnicalLabel, renderLabel);
    const renderExplanation = () => {
      const base = t(explanationKeys[method]);
      return selectedProjection && selectedProjection.fallback === "pca"
        ? `${base} ${t("umap_pca_fallback")}`
        : base;
    };
    projectionExplanation.textContent = renderExplanation();
    localizedTextBindings.set(projectionExplanation, renderExplanation);
    const showsLinearOrigin = method === "pca";
    setLocalizedText(queryLegendLabel, showsLinearOrigin ? "query_vector" : "query");
    queryLegendSymbol.classList.toggle("legend-query-vector", showsLinearOrigin);
    queryLegendSymbol.classList.toggle("legend-query", !showsLinearOrigin);
  }

  function renderVisualization(visualization, searchMode, requestedMethod = "pca") {
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
      currentVisualization = null;
      currentVisualizationSearchMode = searchMode;
      visualizationNote.hidden = true;
      visualizationStatus.hidden = true;
      visualizationPanel.hidden = true;
      return;
    }

    currentVisualization = visualization;
    currentVisualizationSearchMode = searchMode;
    const selectedVisualization = selectVisualizationProjection(visualization, requestedMethod);
    currentProjectionMethod = selectedVisualization ? requestedMethod : "pca";
    visualization = selectedVisualization || selectVisualizationProjection(visualization, "pca") || visualization;
    setTranslatedText(visualizationLabel, visualization.label);
    setProjectionControls(currentProjectionMethod, currentVisualization);
    const displayNote = visualizationDisplayNote(visualization.note);
    setTranslatedText(visualizationNote, displayNote);

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
      setTranslatedText(visualizationStatus, visualization.note);
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
    const showsLinearOrigin = currentProjectionMethod === "pca";
    const fittedSceneScale = computeFittedSceneScale(sceneBounds.width, sceneBounds.height);
    const svg = svgElement("svg", "embedding-map");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("role", "group");
    setLocalizedAttribute(svg, "aria-label", "interactive_projection");

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
      showsLinearOrigin,
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
      pointerStartClientX: null,
      pointerStartClientY: null,
      lastTrackballVector: null,
      dragMoved: false,
      pressedChunkId: null,
      pressedChunkElement: null,
      pressedQuery: false,
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

    ["x", "y", "z"].forEach((dimension, dimensionIndex) => {
      const group = svgElement("g", `embedding-axis embedding-axis-${dimension}`);
      const negativeLine = svgElement("line", "embedding-axis-line embedding-axis-negative");
      const positiveLine = svgElement("line", "embedding-axis-line embedding-axis-positive");
      positiveLine.setAttribute("marker-end", "url(#embedding-axis-arrowhead)");
      const label = svgElement("text", "embedding-axis-label");
      const projectionAxisLabel = currentProjectionMethod === "tsne"
        ? "t-SNE"
        : currentProjectionMethod === "umap" ? "UMAP" : "PCA";
      label.textContent = `${projectionAxisLabel} ${dimensionIndex + 1}`;
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
    if (showsLinearOrigin) originLayer.appendChild(originGroup);
    state.origin = {
      world: sceneGeometry.origin,
      visual: { group: originGroup },
    };

    const queryVector = svgElement("line", "query-vector");
    queryVector.setAttribute("marker-end", "url(#query-vector-arrowhead)");
    if (showsLinearOrigin) queryVectorLayer.appendChild(queryVector);

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
      const pointParameters = {
        rank: point.rank,
        source: point.source,
        similarity: Number(point.similarity).toFixed(3),
      };
      setLocalizedText(
        nativeTooltip,
        point.retrieved ? "retrieved_point_title" : "indexed_point_title",
        pointParameters,
      );
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
      setLocalizedAttribute(
        group,
        "aria-label",
        point.retrieved ? "retrieved_point_label" : "indexed_point_label",
        pointParameters,
      );
      group.setAttribute("aria-pressed", "false");
      visualizationPointByChunkId.set(point.id, group);
      group.addEventListener("focus", () => setFocusedChunk(point.id));
      group.addEventListener("blur", () => clearFocusedChunk(point.id));
      group.addEventListener("click", async (event) => {
        event.stopPropagation();
        if (event.detail === 0) {
          await openChunkInspector(point.id, group);
        }
      });
      group.addEventListener("keydown", async (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          await openChunkInspector(point.id, group);
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
    setLocalizedAttribute(queryGroup, "aria-label", "query_endpoint_label");
    const queryMarker = svgElement("polygon", "embedding-query-marker");
    queryMarker.setAttribute("points", "0,-21 21,0 0,21 -21,0");
    const queryLabel = svgElement("text", "embedding-query-label");
    queryLabel.setAttribute("text-anchor", "middle");
    queryLabel.setAttribute("dominant-baseline", "central");
    queryLabel.textContent = "Q";
    const queryTooltip = svgElement("title", "");
    setLocalizedText(queryTooltip, "query_coordinates", {
      x: formatCoordinate(visualization.query.x),
      y: formatCoordinate(visualization.query.y),
      z: formatCoordinate(visualization.query.z),
    });
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
        togglePinnedQuery(queryGroup);
      }
      state.dragMoved = false;
    });
    queryGroup.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        togglePinnedQuery(queryGroup);
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

  function renderPromptMessages(messages) {
    promptMessages.replaceChildren();
    messages.forEach((message) => {
      const card = document.createElement("article");
      card.className = "card prompt-message";
      const header = document.createElement("div");
      header.className = "card-header";
      const role = document.createElement("span");
      role.className = "badge text-bg-primary";
      setLocalizedText(role, message.role === "system" ? "role_system" : "role_human");
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

  function renderPrompt(prompt) {
    setTranslatedText(promptLabel, prompt.label);
    setTranslatedText(promptTechnicalLabel, prompt.technical_label);
    setTranslatedText(promptNote, prompt.note);
    renderPromptMessages(prompt.messages);
  }

  function renderRetrievalNote(search) {
    const modeNotes = {
      keyword: "keyword_note",
      hybrid: "hybrid_note",
    };
    if (!modeNotes[search.mode]) {
      nonsemanticRetrievalNote.textContent = "";
      localizedTextBindings.delete(nonsemanticRetrievalNote);
      nonsemanticRetrievalNote.hidden = true;
      return;
    }
    const render = () => t(modeNotes[search.mode]);
    nonsemanticRetrievalNote.textContent = render();
    localizedTextBindings.set(nonsemanticRetrievalNote, render);
    nonsemanticRetrievalNote.hidden = false;
  }

  function renderQueryResponse(data) {
    const hasEvidence = data.search.results.length > 0;
    generativeResult.hidden = true;
    ragStages.hidden = false;
    setStagePresentation();
    embeddedQuestion.textContent = data.question;
    setTranslatedText(embeddedRetrievalMethod, data.search.label);
    renderRetrievalNote(data.search);
    searchContent.hidden = false;
    directRetrievalNote.hidden = true;
    setLocalizedText(answerKicker, hasEvidence ? "answer" : "grounding_guard");
    setLocalizedText(answerHeading, hasEvidence ? "generated_answer" : "grounded_answer_status");
    answerContent.replaceChildren();
    const answerText = document.createElement("div");
    answerText.className = "answer-display-text";
    answerText.textContent = data.answer;
    answerContent.appendChild(answerText);
    setTranslatedText(searchLabel, data.search.label);
    setTranslatedText(searchTechnicalLabel, data.search.technical_label);
    setTranslatedText(scoreLabel, data.search.score_label);
    renderSearchResults(
      data.search.results,
      data.search.mode,
      data.search.score_label,
    );
    renderVisualization(data.visualization, data.search.mode);
    renderPrompt(data.prompt);
    embedStage.open = false;
    retrieveStage.open = false;
    promptStage.open = false;
    answerStage.open = true;
    queryError.hidden = true;
    queryResults.hidden = false;
  }

  function renderDirectQueryResponse(data) {
    ragStages.hidden = true;
    generativeResult.hidden = false;
    searchContent.hidden = true;
    searchResults.replaceChildren();
    searchCardByChunkId.clear();
    searchResultByChunkId.clear();
    renderVisualization(null, "generative");
    promptMessages.replaceChildren();
    directAnswerContent.replaceChildren();
    const answerText = document.createElement("div");
    answerText.className = "answer-display-text";
    answerText.textContent = data.answer;
    directAnswerContent.appendChild(answerText);
    queryError.hidden = true;
    queryResults.hidden = false;
  }

  function renderComparisonMode(mode) {
    if (!comparisonResultSet) return;
    const envelope = comparisonResultSet.results[mode];
    if (!envelope || envelope.status !== "ready" || !envelope.result) return;
    activeComparisonMode = mode;
    configureComparisonSelector(comparisonResultSet.results, mode);
    if (mode === "generative") {
      renderDirectQueryResponse(envelope.result);
    } else {
      renderQueryResponse(envelope.result);
    }
  }

  function storeAndRenderComparison(data) {
    comparisonResultSet = deepFreeze(data);
    const initialMode = ["generative", "semantic", "keyword", "hybrid"]
      .find((mode) => data.results[mode] && data.results[mode].status === "ready");
    if (!initialMode) {
      const firstFailure = Object.values(data.results).find((result) => result.message);
      showQueryError(firstFailure ? firstFailure.message : t("query_failed"));
      return;
    }
    renderComparisonMode(initialMode);
  }

  async function submitQuery(event) {
    event.preventDefault();
    const question = questionInput.value.trim();
    const requestVersion = ++queryRequestVersion;
    queryError.hidden = true;
    if (!question) {
      setLocalizedText(queryError, "enter_question");
      queryError.hidden = false;
      queryResults.hidden = true;
      questionInput.focus();
      return;
    }

    clearComparisonState();
    setQueryLoading(true);
    try {
      const response = await fetch("./api/query/all", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({question}),
      });
      const data = await response.json();
      if (requestVersion !== queryRequestVersion) return;
      if (!response.ok) {
        throw new Error(data.message || "The RAG query could not be completed.");
      }
      storeAndRenderComparison(data);
    } catch (error) {
      if (requestVersion !== queryRequestVersion) return;
      if (error instanceof Error) {
        showQueryError(error.message);
      } else {
        setLocalizedText(queryError, "query_failed");
        queryError.hidden = false;
        queryResults.hidden = true;
      }
    } finally {
      if (requestVersion === queryRequestVersion) setQueryLoading(false);
    }
  }

  refreshButton.addEventListener("click", () => loadIndex(true));
  queryForm.addEventListener("submit", submitQuery);
  questionInput.addEventListener("input", () => {
    queryRequestVersion += 1;
    clearComparisonState();
    setQueryLoading(false);
  });
  comparisonButtons.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-comparison-mode]");
    if (!button || button.disabled) return;
    renderComparisonMode(button.dataset.comparisonMode);
  });
  [pcaProjectionButton, tsneProjectionButton, umapProjectionButton].forEach((button) => {
    button.addEventListener("click", () => {
      if (!currentVisualization || currentVisualizationSearchMode !== "semantic") return;
      renderVisualization(
        currentVisualization,
        currentVisualizationSearchMode,
        button.dataset.projection,
      );
    });
  });
  zoomOutButton.addEventListener("click", () => zoomViewer(1 / 1.12));
  zoomInButton.addEventListener("click", () => zoomViewer(1.12));
  resetViewButton.addEventListener("click", resetViewer);
  expandViewButton.addEventListener("click", () => setPresentationMode(true));
  restoreViewButton.addEventListener("click", () => setPresentationMode(false));
  closeExpandedViewButton.addEventListener("click", () => setPresentationMode(false));
  visualizationInspectorClose.addEventListener("click", () => clearPinnedSelection(true));
  visualizationInspector.addEventListener("wheel", (event) => event.stopPropagation());
  visualizationInspector.addEventListener("pointerdown", (event) => event.stopPropagation());
  if (typeof document.addEventListener === "function") {
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !visualizationInspector.hidden) {
        clearPinnedSelection(true);
      }
    });
  }
  refreshExamplesButton.addEventListener("click", loadExamples);
  document.querySelectorAll("[data-language]").forEach((button) => {
    button.addEventListener("click", () => setLanguage(button.dataset.language));
  });
  setLanguage(readStoredLanguage(), false);
  loadExamples();
  loadIndex();
})();
