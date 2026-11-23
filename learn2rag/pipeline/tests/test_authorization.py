import unittest
from typing import Any

from qdrant_client.http.models import ScoredPoint

from ..authorization import get_doc_id_field


def _point(payload: dict[str, Any] | None, point_id: int = 7) -> ScoredPoint:
    return ScoredPoint(id=point_id, score=1.0, version=1, payload=payload)


class DocIdFieldTestCase(unittest.TestCase):
    """`get_doc_id_field` is the one source of the document ID for the authorization cache and the exclusion filter."""

    def test_first_non_empty_field_in_order_document_id_source_node_id(self) -> None:
        cases: list[tuple[dict[str, Any] | None, tuple[str, Any]]] = [
            ({"document_id": "X", "source": "S", "node_id": "N"}, ("document_id", "X")),
            ({"document_id": "", "source": "S", "node_id": "N"}, ("source", "S")),
            ({"document_id": None, "source": "S"}, ("source", "S")),
            ({"source": "S"}, ("source", "S")),
            ({"document_id": "", "source": "", "node_id": "N"}, ("node_id", "N")),
            # Empty like in Python: 0 is skipped as well
            ({"document_id": 0, "source": "S"}, ("source", "S")),
            ({"document_id": "", "source": None}, ("id", 7)),
            ({}, ("id", 7)),
            (None, ("id", 7)),
        ]
        for payload, expected in cases:
            with self.subTest(payload=payload):
                self.assertEqual(get_doc_id_field(_point(payload)), expected)

    def test_value_keeps_its_stored_type(self) -> None:
        field, value = get_doc_id_field(_point({"document_id": 123, "source": "S"}))

        self.assertEqual(field, "document_id")
        self.assertIsInstance(value, int)


if __name__ == "__main__":
    unittest.main()
