# Pipeline Configuration Parameters

This document describes all keys present in:

- `learn2rag/pipeline/opt_config.json`
- `learn2rag/pipeline/user_config.json`

## `user_config.json`

| Key | Type | Allowed / expected values | Implemented behavior |
|---|---|---|---|
| `collection_name` | `string` | Qdrant collection name | Used to create/select the Qdrant collection in `qdrant.py`, `ingestion.py`, and `search.py`. Required. |
| `imported_documents_file_path` | `string` | Path to JSON file compatible with `JSONLoader` (`jq_schema='.[]'`, `content_key='content'`) | Used by `ingestion.index()` to load input documents via `json_loader.json_loader(...)`. Required for ingestion. |

### Notes

- `user_config.json` is loaded by `config.py` (or overridden via `PIPELINE_USER_CONFIG`).
- Missing required keys (for example `collection_name` during search/index) will raise `KeyError`.

## `opt_config.json`

| Key | Type | Allowed / expected values                                                                                          | Implemented behavior                                                                                                                                                                                                  |
|---|---|--------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `chunk_size` | `int` | Positive integer                                                                                                   | Number of characters per indexed text segment. Used by `RecursiveCharacterTextSplitter` in `ingestion.py`.                                                                                                            |
| `chunk_overlap` | `int` | Integer, typically `0 <= overlap < chunk_size`                                                                     | Number of shared characters between adjacent chunks. Used by `RecursiveCharacterTextSplitter` in `ingestion.py`.                                                                                                      |
| `embedding_model` | `string` | Implemented options: `BAAI/bge-m3`, `sentence-transformers/all-mpnet-base-v2`                                      | Embedding model used for ingestion and retrieval. Handled in `embeddings.py`; unknown value falls back to `BAAI/bge-m3` with warning. **Important:** requires a matching dimension in `vector_size[embedding_model]`. |
| `vector_size` | `object` (`model_name -> int`) | Must contain dimension for active `embedding_model`                                                                | Length of vector representation from embedding model. Missing model key causes failure (`KeyError`).                                                                                                                  |
| `search_mode` | `string` | Implemented options: `dense`, `sparse`, `dense_sparse`, `dense_sparse_colbert`, `multi_search`                     | Controls embedding creation and retrieval flow in `search.py`; also affects ingestion behavior in `ingestion.py`.                                                                                                     |
| `reranking` | `string` | Effective enabled value: `"True"` (string). Any other value disables reranking.                                    | If exactly `"True"`, reranking is executed during retrieval.                                                                                                                                                          |
| `reranking_mode` | `string` | Implemented options: `reranking_with_flagreranker`, `reranking_with_sentence_transformers`, `reranking_with_colbert` | Selects reranking implementation during retrieval when `reranking == "True"`.                                                                                                                                         |
| `top_k_reranker` | `int` | Positive integer                                                                                                   | Number of points (search results) kept after reranking.                                                                                                                                                               |
| `fusion_mode` | `string` | Implemented options: `RRF`, `DBSF`                                                                                 | Used in retrieval with hybrid search (if `search_mode` is `dense_sparse` or `dense_sparse_colbert`) to combine search results.                                                                                        |
| `rewrite` | `string` | Effective enabled value: `"True"` (string)                                                                         | If exactly `"True"`, query rewriting is used for retrieval.                                                                                                                                                           |
| `rewrite_mode` | `string` | Implemented: `subqueries`, `keywords`, `subqueries_keywords`                                                       | Controls whether subqueries and/or keyword expansions are generated if `rewrite == "True"`.                                                                                                                           |
| `n_subqueries` | `int` | Positive integer                                                                                                   | Number of generated subqueries if rewriting with subqueries is performed.                                                                                                                                             |
| `n_keywords` | `int` | Positive integer                                                                                                   | Number of generated keywords if rewriting with keywords is performed.                                                                                                                                                 |
| `top_k_subqueries` | `int` | Positive integer                                                                                                   | Number of retrieval results for each generated subquery in rewritten search flow.                                                                                                                                     |
| `top_k_keywords` | `int` | Positive integer                                                                                                   | Number of retrieval results for each generated keyword query in rewritten search flow.                                                                                                                                |
| `prefetch_limit_dense` | `int` | Positive integer                                                                                                   | Used in hybrid search (if `search_mode` is `dense_sparse` or `dense_sparse_colbert`) to define number of dense prefetch results.                                                                                      |
| `prefetch_limit_sparse` | `int` | Positive integer                                                                                                   | Used in hybrid search (if `search_mode` is `dense_sparse` or `dense_sparse_colbert`) to define number of sparse prefetch results.                                                                                     |
| `prefetch_limit_colbert` | `int` | Positive integer                                                                                                   | Used in hybrid search (if `search_mode` is `dense_sparse_colbert`) to define number of ColBERT prefetch results.                                                                                                      |
| `query_mode` | `string` | Implemented options: `single`, `multi` (special handling). Other values than `multi` follow single-query defaults. | `multi` activates multi-vector ingestion schema                                                                                                                                                                       |
| `multi_search` | `list[string]` | List of metadata fields (for example `title`, `summary`, `source_path`)                                            | In multi-query mode, embeddings are concatenated for `content + each listed field` during ingestion/search. Missing metadata fields are replaced with empty strings.                                                  |
| `prompt` | `string` | System prompt template (expects `{context}` placeholder in current design).                                        | Used to build system message for answer generation.                                                                                                                                                                   |
| `top_k` | `int` | Positive integer                                                                                                   | Number of retrieval results.                                                                                                                                                                       |

### Notes:

#### 1) `search_mode`

- `dense`: implemented for ingestion and retrieval.
- `dense_sparse`: implemented for ingestion and retrieval.
- `dense_sparse_colbert`: implemented for ingestion and retrieval.
- `sparse`: retrieval branch exists in `search.py`, but ingestion does not create sparse-only points as a dedicated mode.
- `multi_search`: retrieval branch expects a dict-like query with keys `content` + all `multi_search` fields.

#### 2) `embedding_model`

- `BAAI/bge-m3`: full support for dense/sparse/colbert outputs in `embeddings.py`.
- `sentence-transformers/all-mpnet-base-v2`: dense embedding only in `embeddings.py`.
- Any other model string: warning fallback to default bge-m3 in `embeddings.py`, but still requires consistent `vector_size` mapping in `opt_config` for Qdrant setup.

#### 3) `reranking_mode`

- `reranking_with_flagreranker`: implemented via `FlagReranker` from `FlagEmbedding`.
- `reranking_with_sentence_transformers`: implemented via `CrossEncoder` from `sentence_transformers`.
- `reranking_with_colbert`: implemented via Qdrant `colbert` query over candidate IDs.

#### 4) `rewrite_mode`

- `subqueries`: generates and searches additional subqueries with selected `search_mode`.
- `keywords`: generates keywords and searches them with forced sparse mode.
- `subqueries_keywords`: combines both rewriting behaviors.

### Known Implementation Caveats:

- `reranking` and `rewrite` are checked as **strings** (`"True"`), not booleans.
- `top_k`, prefetch limits, and chunk settings are not schema-validated in code (invalid types fail at runtime).


