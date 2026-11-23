import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
from langchain_core.documents import Document

from ..ingestion import ingest_batch

OPT_CONFIG: dict[str, Any] = {
    "chunk_size": 50,
    "chunk_overlap": 0,
    "ingestion_batch_size": 32,
    "embedding_model": "test-model",
    "search_mode": "dense",
    "query_mode": "single",
    "multi_search": [],
}


class IngestBatchPayloadTestCase(unittest.TestCase):
    """All chunks of a document carry the same document identity (spec `ingestion-payload`)."""

    def _payloads(self, document: Document) -> list[dict[str, Any]]:
        qdrant = MagicMock()
        qdrant.get_client.return_value.scroll.return_value = ([], None)
        with patch("learn2rag.pipeline.ingestion.create_embeddings",
                   side_effect=lambda texts, *args: {"dense_vecs": [np.zeros(3) for _ in texts]}):
            ingest_batch([document], qdrant, {"collection_name": "test_collection"}, OPT_CONFIG)
        return [call.kwargs["points"][0].payload for call in qdrant.get_client.return_value.upsert.call_args_list]

    def test_all_chunks_of_a_document_have_the_same_identity(self) -> None:
        metadata = {"source": "docs/guide.txt", "loader_id": "loader", "content_hash": "hash"}
        cases: list[tuple[dict[str, Any], Any]] = [
            ({**metadata, "document_id": "doc-1"}, "doc-1"),
            (metadata, ""),
        ]
        for metadata, expected_document_id in cases:
            with self.subTest(document_id=metadata.get("document_id", "missing")):
                document = Document(page_content=" ".join(f"Sentence number {i} of the guide." for i in range(10)),
                                    metadata=metadata)

                payloads = self._payloads(document)

                self.assertGreater(len(payloads), 1)
                self.assertEqual(
                    {(p["loader_id"], p["source"], p["document_id"]) for p in payloads},
                    {("loader", "docs/guide.txt", expected_document_id)},
                )


if __name__ == "__main__":
    unittest.main()
