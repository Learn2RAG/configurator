# Pipeline Configuration Parameters

This document describes all keys present in:

- `learn2rag/pipeline/opt_config.json`
- `learn2rag/pipeline/user_config.json`

## `user_config.json`

| Key | Type | Allowed / expected values | Implemented behavior                                                                                            |
|---|---|---|-----------------------------------------------------------------------------------------------------------------|
| `collection_name` | `string` | Qdrant collection name | Used to create/select the Qdrant collection for ingestion/retrieval. Required.                                  |

### Notes

- `user_config.json` is loaded by `config.py` (or overridden via `PIPELINE_USER_CONFIG`).
- Missing required keys (for example `collection_name` during search/index) will raise `KeyError`.
- LLM used for answer generation and query rewriting is configured in `.env`.

## `opt_config.json`

| Key | Type | Allowed / expected values                                                                                            | Implemented behavior                                                                                                                                                                                                               |
|---|---|----------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `chunk_size` | `int` | Positive integer                                                                                                     | Number of characters per indexed text segment. Used by `RecursiveCharacterTextSplitter` in `ingestion.py`.                                                                                                                         |
| `chunk_overlap` | `int` | Integer, typically `0 <= overlap < chunk_size`                                                                       | Number of shared characters between adjacent chunks. Used by `RecursiveCharacterTextSplitter` in `ingestion.py`.                                                                                                                   |
| `embedding_model` | `string` | Implemented options: `BAAI/bge-m3`, `sentence-transformers/all-mpnet-base-v2`                                        | Embedding model used for ingestion and retrieval. Handled in `embeddings.py`; unknown value falls back to `BAAI/bge-m3` with warning. **Important:** requires a matching dimension in `vector_size[embedding_model]`.<sup>1)</sup> |
| `vector_size` | `object` (`model_name -> int`) | Must contain name and vector dimension for active `embedding_model`                                                  | Model name and length of vector representation from implemented embedding models. Missing model key of the active `embedding_model` causes failure (`KeyError`).                                                                   |
| `search_mode` | `string` | Implemented options: `dense`, `sparse`, `dense_sparse`, `dense_sparse_colbert`, `multi_search`                       | Controls embedding creation and retrieval flow in `search.py`; also affects ingestion behavior in `ingestion.py`.<sup>2)</sup>                                                                                                     |
| `top_k` | `int` | Positive integer | Maximum number of contexts handed to the LLM. It does not limit the candidate pool: the pool is sorted first and the authorization check stops as soon as `top_k` authorized results are found.<sup>6)</sup> |
| `reranking` | `string` | Effective enabled value: `"True"` (string). Any other value disables reranking.                                      | If exactly `"True"`, reranking is executed during retrieval.                                                                                                                                                                       |
| `reranking_mode` | `string` | Implemented options: `reranking_with_flagreranker`, `reranking_with_sentence_transformers`, `reranking_with_colbert` | Selects reranking implementation during retrieval when `reranking == "True"`.<sup>3)</sup>                                                                                                                                         |
| `fusion_mode` | `string` | Implemented options: `RRF` (Reciprocal Rank Fusion), `DBSF` (Distribution-Based Score Fusion) | Used in retrieval with hybrid search (if `search_mode` is `dense_sparse` or `dense_sparse_colbert`) to combine the vector types of one query (the question and each subquery). Results of different queries are always combined with RRF, see `fusion_weights`. |
| `fusion_weights` | `object` | Keys `original` and `rewrite`, numbers `> 0`. A missing key (or the whole object) means `1.0` | Weights of the original question (after history contextualization) and of all rewrites (subqueries and keywords together) in the final RRF fusion. Has no effect without subqueries or keywords.<sup>6)</sup> |
| `rrf_k` | `int` | Positive integer, optional | RRF constant of the fusions across queries. If missing, the Qdrant default is used. |
| `rewrite` | `string` | Effective enabled value: `"True"` (string)                                                                           | If exactly `"True"`, query rewriting is used for retrieval.                                                                                                                                                                        |
| `rewrite_mode` | `string` | Implemented: `subqueries`, `keywords`, `history`, and any `_`-joined combination of them (e.g. `subqueries_keywords`, `history_subqueries_keywords`) | Selects one or more rewriting components, active if `rewrite == "True"`.<sup>4)</sup>                                                                                                                           |
| `n_subqueries` | `int` | Positive integer                                                                                                     | Number of generated subqueries if rewriting with subqueries is performed.                                                                                                                                                          |
| `n_keywords` | `int` | Positive integer                                                                                                     | Number of generated keywords if rewriting with keywords is performed.                                                                                                                                                              |
| `top_k_subqueries` | `int` | Positive integer | Number of candidates each generated subquery adds to the candidate pool. |
| `top_k_keywords` | `int` | Positive integer | Number of candidates each generated keyword query adds to the candidate pool. |
| `prefetch_limit_dense` | `int` | Positive integer | Number of dense candidates of the question for the candidate pool, in every `search_mode` that uses dense vectors (including `dense`). With hybrid search it is also the dense prefetch of each subquery before fusion. |
| `prefetch_limit_sparse` | `int` | Positive integer | Number of sparse candidates of the question for the candidate pool, in every `search_mode` that uses sparse vectors (including `sparse`). With hybrid search it is also the sparse prefetch of each subquery before fusion. |
| `prefetch_limit_colbert` | `int` | Positive integer | Number of ColBERT candidates of the question for the candidate pool if `search_mode` is `dense_sparse_colbert`. Also the ColBERT prefetch of each subquery before fusion. |
| `query_mode` | `string` | Implemented options: `single`, `multi` (special handling). Other values than `multi` follow single-query defaults.   | `multi` activates multi-vector ingestion schema                                                                                                                                                                                    |
| `multi_search` | `list[string]` | List of metadata fields (for example `title`, `summary`, `source_path`)                                              | In multi-query mode, embeddings are concatenated for `content + each listed field` during ingestion/search. Missing metadata fields are replaced with empty strings.                                                               |
| `prompt` | `string` | System prompt template (expects `{context}` placeholder in current design).                    | Used to build system message for answer generation. The recent conversation history is added as separate messages between system message and Current Question, see `history_length`.                                                                                                                                                                                |
| `history_length` | `int` | Non-negative integer. A missing key defaults to `5`.                                                                 | Number of previous answers of the pipeline that are put into the prompt, each together with the question it belongs to. `0` disables the conversation history. Applied in `generate.py`.<sup>5)</sup>                                |

### Notes:

#### 1) `embedding_model`

- `BAAI/bge-m3`: full support for dense/sparse/colbert outputs in `embeddings.py`. See https://huggingface.co/BAAI/bge-m3 for details.
- `sentence-transformers/all-mpnet-base-v2`: dense embedding only in `embeddings.py`. See https://huggingface.co/sentence-transformers/all-mpnet-base-v2 for details.
- Any other model string: warning fallback to default `BAAI/bge-m3` in `embeddings.py`.

#### 2) `search_mode`

- `dense`: implemented for ingestion and retrieval.
- `dense_sparse`: implemented for ingestion and retrieval.
- `dense_sparse_colbert`: implemented for ingestion and retrieval.
- `sparse`: retrieval branch exists in `search.py`, but ingestion does not create sparse-only points as a dedicated mode.
- `multi_search`: retrieval branch expects a dict-like query with keys `content` + all `multi_search` fields.

#### 3) `reranking_mode`

- `reranking_with_flagreranker`: implemented via `FlagReranker` from `FlagEmbedding`. See https://huggingface.co/BAAI/bge-reranker-v2-m3 for details.
- `reranking_with_sentence_transformers`: implemented via `CrossEncoder` from `sentence_transformers`. See https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2 for details.
- `reranking_with_colbert`: implemented to use `colbert` query over candidate search results. Only possible if `BAAI/bge-m3` is used as `embedding_model` and ColBERT vectors exist in used Qdrant collection.

#### 4) `rewrite_mode`

`rewrite_mode` is split on `_` into components; each recognized component activates independently, so any combination (e.g. `history_subqueries`) works without needing its own literal entry.

- `subqueries`: generates and searches additional subqueries with selected `search_mode`.
- `keywords`: generates keywords and searches them with forced sparse mode. Only possible if sparse vectors exist in used Qdrant collection.
- `history`: **opt-in, requires the conversation history to be passed to search** (it is not by clients that omit it, for example the optimization scripts). Before the base search and any subquery/keyword expansion, rewrites the Current Question plus the recent conversation history (same `history_length` window used for generation) into a standalone search query via one additional LLM call (`rewrite.contextualize_query`), so follow-up questions like "tell me again" or "what about tomatoes?" resolve to something retrieval can actually match. Only runs when the conversation history is non-empty; a failed or empty rewrite falls back to the Current Question unchanged, so retrieval is never blocked by it. Runs once per request in `search_authorized`. **Adds one sequential LLM call per follow-up turn when active** - on a slow local model this is a deliberate latency/quality trade-off, which is why it defaults to off.
- `subqueries_keywords`, `history_subqueries_keywords`, etc.: combine the components above; when `history` is combined with `subqueries`/`keywords`, the latter operate on the already-rewritten (contextualized) query, not the raw Current Question.
- Prompts for subquery/keywords/history-rewrite generation can be modified in `rewrite.py`.

#### 5) `history_length`

- Clients of the OpenAI API (for example Open WebUI) send the whole conversation with every request. `history_length` limits how much of it reaches the model, because how much history a model can handle depends on the model.
- Counted in answers of the pipeline. The question belonging to an answer is always kept as well, so a value of `5` results in up to 10 messages.
- Only the roles `user` and `assistant` (also accepted: `model`) are used. Any other role in the request is dropped, so that a client cannot inject system instructions through the history.
- Retrieval is unaffected by default: the search uses the Current Question as-is, and every request is filtered by the authorization of the user again. This changes only if `rewrite_mode` includes the `history` component (see [rewrite_mode notes](#4-rewrite_mode)), in which case retrieval searches a history-derived standalone query instead of the raw Current Question.

#### 6) Retrieval flow, `top_k`, `fusion_weights`

`search_authorized` builds one candidate pool per request and checks the authorization lazily:

1. Rewrite (history, subqueries, keywords) and the query embeddings run once per request.
2. One Qdrant call returns the pool: the union of the question, all subqueries and all keywords, each limited by `prefetch_limit_*`, `top_k_subqueries` and `top_k_keywords`. Each point is in the pool once.
3. The pool is sorted: by the reranker (`reranking == "True"`, scoring the **whole** pool) or by the fusion tree of Qdrant. The tree fuses the vector types of each query with `fusion_mode`, the subqueries and the keywords with RRF, and finally the question (A) and the rewrites (B) with a weighted RRF (`fusion_weights`, `rrf_k`). Raw scores of different queries are never compared.
4. In rank order, blocks of `top_k` candidates are checked for authorization. Each document (per `loader_id`) is checked at most once per request. The check stops when `top_k` authorized results are found; users with few permissions can get fewer than `top_k` contexts.

The keys `top_k_reranker`, `max_auth_retries`, `auth_oversample_start` and `auth_oversample_step` no longer exist. They are ignored if they are still present in a configuration.

`search()` and `search_multi()`, which are used by scripts, still use `top_k` as the Qdrant limit of a single search.

### Known Implementation Caveats:

- `reranking` and `rewrite` are checked as **strings** (`"True"`), not booleans.
- `top_k`, prefetch limits, and chunk settings are not schema-validated in code (invalid types fail at runtime).
- `search_mode` has dual functionality as it determines which embedding type is used for both document indexing during ingestion and query embedding during retrieval.
