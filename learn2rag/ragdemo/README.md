# Learn2RAG RAG Demo

## Quick start

Clone Learn2RAG and check out the branch that contains the RAG Demo. From the repository root, install the dependencies and start the Configurator:

```bash
uv sync --no-install-project
./start
```

Open the Configurator at <http://localhost:9000>.

## Configure demo documents

In the Configurator, configure or select a local document source containing the documents you want to explore.

## Create/start the pipeline

Create a pipeline for that source and start it. Wait until the documents have been processed and indexed.

## Open the demo

Open `http://localhost:<pipeline-port>/ragdemo/`, replacing `<pipeline-port>` with the port assigned to the running pipeline. The pipeline port is not necessarily `9003`.

## Example questions

The packaged examples target the test demo PDFs. When using different PDFs or another dataset, update [`demo_questions.json`](demo_questions.json) so the suggestions match the indexed content.

## Compare answers

One question prepares Semantic, Keyword, Hybrid, and Generative AI results. Switching between the loaded modes does not rerun retrieval or generation. Generative AI answers directly from the configured language model and uses no retrieved documents.

Semantic results offer PCA, t-SNE, and UMAP views of the same bounded dense-vector point set. Projection changes are visualization-only: retrieval, ranks, similarity, prompt, and answer stay unchanged.

`dense_sparse` is the intended default for new demo indexes so both dense and sparse vectors are available. Hybrid remains unavailable for existing dense-only collections until their documents are reindexed into a compatible collection or storage.

## Language

`EN / DE` changes static interface text only. Indexed source documents are never modified. Retrieved chunks, example questions, user questions, prompts, and generated answers remain in their original or current language.

## Development notes

Restart the pipeline after Python or backend changes. After JavaScript or CSS changes, hard-refresh the browser, for example with `Ctrl + F5`.
