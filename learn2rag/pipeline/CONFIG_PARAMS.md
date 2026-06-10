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
| `top_k` | `int` | Positive integer                                                                                                     | Number of retrieval results. If rewriting active: number of search results for each retrieval. If reranking active: number of search results before reranking.                                                                     |
| `reranking` | `string` | Effective enabled value: `"True"` (string). Any other value disables reranking.                                      | If exactly `"True"`, reranking is executed during retrieval.                                                                                                                                                                       |
| `reranking_mode` | `string` | Implemented options: `reranking_with_flagreranker`, `reranking_with_sentence_transformers`, `reranking_with_colbert` | Selects reranking implementation during retrieval when `reranking == "True"`.<sup>3)</sup>                                                                                                                                         |
| `top_k_reranker` | `int` | Positive integer                                                                                                     | Number of points (search results) kept after reranking.                                                                                                                                                                            |
| `fusion_mode` | `string` | Implemented options: `RRF` (Reciprocal Rank Fusion), `DBSF` (Distribution-Based Score Fusion)                        | Used in retrieval with hybrid search (if `search_mode` is `dense_sparse` or `dense_sparse_colbert`) to combine search results.                                                                                                     |
| `rewrite` | `string` | Effective enabled value: `"True"` (string)                                                                           | If exactly `"True"`, query rewriting is used for retrieval.                                                                                                                                                                        |
| `rewrite_mode` | `string` | Implemented: `subqueries`, `keywords`, `subqueries_keywords`                                                         | Controls whether subqueries and/or keyword expansions are generated if `rewrite == "True"`.<sup>4)</sup>                                                                                                                           |
| `n_subqueries` | `int` | Positive integer                                                                                                     | Number of generated subqueries if rewriting with subqueries is performed.                                                                                                                                                          |
| `n_keywords` | `int` | Positive integer                                                                                                     | Number of generated keywords if rewriting with keywords is performed.                                                                                                                                                              |
| `top_k_subqueries` | `int` | Positive integer                                                                                                     | Number of retrieval results for each generated subquery in rewritten search flow.                                                                                                                                                  |
| `top_k_keywords` | `int` | Positive integer                                                                                                     | Number of retrieval results for each generated keyword query in rewritten search flow.                                                                                                                                             |
| `prefetch_limit_dense` | `int` | Positive integer                                                                                                     | Used in hybrid search (if `search_mode` is `dense_sparse` or `dense_sparse_colbert`) to define number of dense prefetch results.                                                                                                   |
| `prefetch_limit_sparse` | `int` | Positive integer                                                                                                     | Used in hybrid search (if `search_mode` is `dense_sparse` or `dense_sparse_colbert`) to define number of sparse prefetch results.                                                                                                  |
| `prefetch_limit_colbert` | `int` | Positive integer                                                                                                     | Used in hybrid search (if `search_mode` is `dense_sparse_colbert`) to define number of ColBERT prefetch results.                                                                                                                   |
| `query_mode` | `string` | Implemented options: `single`, `multi` (special handling). Other values than `multi` follow single-query defaults.   | `multi` activates multi-vector ingestion schema                                                                                                                                                                                    |
| `multi_search` | `list[string]` | List of metadata fields (for example `title`, `summary`, `source_path`)                                              | In multi-query mode, embeddings are concatenated for `content + each listed field` during ingestion/search. Missing metadata fields are replaced with empty strings.                                                               |
| `prompt` | `string` | System prompt template (expects `{context}` placeholder in current design).                    | Used to build system message for answer generation.                                                                                                                                                                                |

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

- `subqueries`: generates and searches additional subqueries with selected `search_mode`.
- `keywords`: generates keywords and searches them with forced sparse mode. Only possible if sparse vectors exist in used Qdrant collection.
- `subqueries_keywords`: combines both rewriting behaviors.
- Prompts for subquery/keywords generation can be modified in `rewrite.py`.

### Known Implementation Caveats:

- `reranking` and `rewrite` are checked as **strings** (`"True"`), not booleans.
- `top_k`, prefetch limits, and chunk settings are not schema-validated in code (invalid types fail at runtime).
- `search_mode` has dual functionality as it determines which embedding type is used for both document indexing during ingestion and query embedding during retrieval.
