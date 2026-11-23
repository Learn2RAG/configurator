# Changelog
## [Unreleased]
### Added
- `fusion_weights` and optional `rrf_k` in `opt_config.json` to weight the original question against subqueries and keywords; invalid values fall back to their default with a warning
- `max_search_rounds` in `opt_config.json` (default 3): authorized search searches again if fewer than `top_k` results are authorized

### Fixed
- authorized search no longer repeats the rewrite LLM calls and the embeddings when too few results are authorized
- authorized search checks each document at most once per request

### Changed
- **BREAKING:** `top_k` is now only the maximum number of contexts handed to the LLM; it no longer limits the candidates of the search
- **BREAKING:** `prefetch_limit_dense`, `prefetch_limit_sparse` and `prefetch_limit_colbert` also apply to the search modes `dense` and `sparse` and define the size of the candidate pool
- authorized search builds a candidate pool, sorts it (reranker or rank fusion) and checks the authorization lazily in rank order; with too few authorized results it repeats this in up to `max_search_rounds` rounds, excluding the candidates and the rejected documents of the previous rounds
- **BREAKING:** for users with few permissions a request runs up to `max_search_rounds` Qdrant searches, reranker runs and authorization checks
- reranking sorts the whole candidate pool instead of cutting it off
- **BREAKING:** default search mode is `dense_sparse`; collections created with `dense` have no sparse vectors, so either keep `search_mode: dense` or re-import the data into a new collection
- default of `prefetch_limit_dense`, `prefetch_limit_sparse` and `prefetch_limit_colbert` is 15 instead of 25

### Removed
- **BREAKING:** `top_k_reranker`, `max_auth_retries`, `auth_oversample_start` and `auth_oversample_step` from `opt_config.json`; they are ignored if still present

## [1.1.0] - 2026-11-09
### Added
- embedding models are downloaded on the first start
- `remove-learn2rag-software` script to remove the Learn2RAG application runtimes
- `remove-user-data` script to remove user data (asks for confirmation; `-y` skips it)
- `uninstall` now removes the software and asks whether to also remove user data

### Fixed
- Drupal document access
- Drupal example configuration

## [1.0.0] - 2026-09-16
### Added
- MediaWiki data import
- Jira data import
- chat history awareness
- per-document user access control (Drupal)
- pipeline auto-optimization
- retrieval optimization
- displaying status during import

### Fixed
- offline installation
- Windows process management

### Changed
- open webui chat UI replaced with llama.cpp's UI
- interface improvements
- performance improvements

### Removed
- unused API endpoints were removed

## [0.3.0] - 2026-06-02
### Added
- continuous import with updating the indexed data
- Sharepoint import and simple configuration interface
- Drupal import and configuration interface
- ability to clean pipeline's storage without removing the pipeline
- simple status messages for the import process
- OpenAI-compatible API for RAG
### Fixed
- uninstall script
- interface language selection when browser accepts both
- web data source configuration
- missing import logs in the terminal
### Changed
- by default configuration interface and chat listen on localhost only
- in config.yml "host" and "port" options are moved under "UI"; see also "CHAT"
- openwebui upgraded to a newer version
### Removed
- openwebui-pipelines (not needed anymore)
