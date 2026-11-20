import copy
import unittest
from typing import Any, Mapping, Sequence, cast
from unittest.mock import AsyncMock, patch

from qdrant_client.http.models import ScoredPoint

from ..search import _Queries, search_authorized

LOADER_ID = "loader"


def _pool(*doc_ids: str | None) -> list[ScoredPoint]:
    """One point per entry in rank order; `None` is a point without payload."""
    return [
        ScoredPoint(
            id=i + 1,
            score=1.0 - i * 0.01,
            version=1,
            payload=None if doc_id is None else {"loader_id": LOADER_ID, "document_id": doc_id, "content": f"chunk {i}"},
        )
        for i, doc_id in enumerate(doc_ids)
    ]


def _doc_ids(points: Sequence[ScoredPoint]) -> list[str]:
    return [cast(Mapping[str, str], point.payload)["document_id"] for point in points]


class _FakeFilter:
    """Authorization filter that records its calls and allows a fixed set of documents."""

    def __init__(self, allowed: set[str], failing_calls: frozenset[int] = frozenset()) -> None:
        self.allowed = allowed
        self.failing_calls = failing_calls
        self.calls: list[list[str]] = []
        self.user_auths: list[Any] = []

    async def filter_documents(self, user_auth: Any, documents: Mapping[str, Any]) -> set[str]:
        self.user_auths.append(user_auth)
        self.calls.append(list(documents))
        if len(self.calls) in self.failing_calls:
            raise RuntimeError("authorization backend unavailable")
        return self.allowed & set(documents)

    @property
    def checked(self) -> list[str]:
        return [doc_id for call in self.calls for doc_id in call]


class SearchAuthorizedTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.user_auths = {LOADER_ID: "user-token"}
        self.user_config = {"collection_name": "test_collection"}
        self.opt_config: dict[str, Any] = {"top_k": 3}

        self.build_queries = self._patch(
            "learn2rag.pipeline.search._build_queries", new=AsyncMock(return_value=_Queries("test query", [], [])))
        self._patch("learn2rag.pipeline.search._embed_queries")
        self._patch("learn2rag.pipeline.search._build_prefetch_tree")
        self.retrieve_pool = self._patch("learn2rag.pipeline.search._retrieve_pool", return_value=[])
        self.rerank_pool = self._patch(
            "learn2rag.pipeline.search._rerank_pool", side_effect=lambda question, pool, *args, **kwargs: pool)

    def _patch(self, target: str, **kwargs: Any) -> Any:
        patcher = patch(target, **kwargs)
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        return mock

    async def _run(self, pool: list[ScoredPoint], allowed: set[str], **filter_kwargs: Any) -> tuple[list[ScoredPoint], _FakeFilter]:
        self.retrieve_pool.return_value = pool
        auth_filter = _FakeFilter(allowed, **filter_kwargs)
        with patch("learn2rag.pipeline.authorization._get_authorization_filter", return_value=auth_filter):
            results = await search_authorized(
                "test query", self.user_auths, user_config=self.user_config, opt_config=self.opt_config)
        return results, auth_filter

    async def test_pool_is_retrieved_once_across_blocks(self) -> None:
        results, auth_filter = await self._run(_pool("a", "b", "c", "d", "e", "f", "g"), allowed={"g"})

        self.assertEqual(_doc_ids(results), ["g"])
        self.assertEqual(len(auth_filter.calls), 3)
        self.retrieve_pool.assert_called_once()
        self.build_queries.assert_awaited_once()

    async def test_stops_after_first_block_when_enough_are_allowed(self) -> None:
        pool = _pool(*"abcdefghij")
        results, auth_filter = await self._run(pool, allowed=set("abcdefghij"))

        self.assertEqual(_doc_ids(results), ["a", "b", "c"])
        self.assertEqual(auth_filter.calls, [["a", "b", "c"]])

    async def test_fetches_next_block_until_top_k_is_reached(self) -> None:
        results, auth_filter = await self._run(_pool(*"abcdefghi"), allowed={"b", "e", "f", "h"})

        self.assertEqual(_doc_ids(results), ["b", "e", "f"])
        self.assertEqual(auth_filter.calls, [["a", "b", "c"], ["d", "e", "f"]])

    async def test_exhausted_pool_returns_all_allowed_in_rank_order(self) -> None:
        results, auth_filter = await self._run(_pool(*"abcdefg"), allowed={"c", "g"})

        self.assertEqual(_doc_ids(results), ["c", "g"])
        self.assertEqual(auth_filter.checked, list("abcdefg"))

    async def test_result_is_cut_to_top_k(self) -> None:
        self.opt_config["top_k"] = 2
        results, _ = await self._run(_pool("a", "b", "c", "d"), allowed={"a", "c", "d"})

        self.assertEqual(_doc_ids(results), ["a", "c"])

    async def test_each_document_is_checked_once_across_chunks_and_blocks(self) -> None:
        self.opt_config["top_k"] = 2
        # "a" has three chunks in two blocks and is denied
        pool = _pool("a", "a", "b", "a", "c", "d")
        results, auth_filter = await self._run(pool, allowed={"b", "c", "d"})

        self.assertEqual(auth_filter.calls, [["a"], ["b"], ["c", "d"]])
        self.assertEqual(_doc_ids(results), ["b", "c"])

    async def test_all_chunks_of_an_allowed_document_are_returned(self) -> None:
        self.opt_config["top_k"] = 4
        results, auth_filter = await self._run(_pool("a", "b", "a", "a"), allowed={"a"})

        self.assertEqual(_doc_ids(results), ["a", "a", "a"])
        self.assertEqual(auth_filter.checked, ["a", "b"])

    async def test_result_follows_reranker_order(self) -> None:
        self.opt_config["top_k"] = 4
        self.rerank_pool.side_effect = lambda question, pool, *args, **kwargs: list(reversed(pool))
        results, _ = await self._run(_pool("z", "m", "a", "q"), allowed={"z", "m", "a", "q"})

        self.assertEqual(_doc_ids(results), ["q", "a", "m", "z"])

    async def test_filter_gets_the_auth_of_the_loader(self) -> None:
        _, auth_filter = await self._run(_pool("a"), allowed={"a"})

        self.assertEqual(auth_filter.user_auths, ["user-token"])

    async def test_filter_without_result_means_not_authorized(self) -> None:
        results, auth_filter = await self._run(_pool("a", "b", "c", "d"), allowed=set())

        self.assertEqual(results, [])
        self.assertEqual(auth_filter.checked, ["a", "b", "c", "d"])

    async def test_failing_filter_denies_the_block_and_the_search_continues(self) -> None:
        results, auth_filter = await self._run(
            _pool("a", "b", "c", "d", "e"), allowed={"a", "b", "c", "d", "e"}, failing_calls=frozenset({1}))

        self.assertEqual(_doc_ids(results), ["d", "e"])
        self.assertEqual(auth_filter.calls, [["a", "b", "c"], ["d", "e"]])

    async def test_point_without_payload_is_not_authorized(self) -> None:
        results, auth_filter = await self._run(_pool(None, "a", "b"), allowed={"a", "b"})

        self.assertEqual(_doc_ids(results), ["a", "b"])
        self.assertEqual(auth_filter.checked, ["a", "b"])

    async def test_legacy_retry_and_reranker_keys_have_no_effect(self) -> None:
        self.opt_config.update(max_auth_retries=5, auth_oversample_start=50, auth_oversample_step=50, top_k_reranker=99)
        pool = _pool(*"abcdefgh")
        results, auth_filter = await self._run(pool, allowed=set("abcdefgh"))

        self.assertEqual(_doc_ids(results), ["a", "b", "c"])
        self.assertEqual(auth_filter.calls, [["a", "b", "c"]])
        self.retrieve_pool.assert_called_once()

    async def test_opt_config_is_not_modified(self) -> None:
        original = copy.deepcopy(self.opt_config)
        await self._run(_pool(*"abcdef"), allowed={"f"})

        self.assertEqual(self.opt_config, original)


if __name__ == "__main__":
    unittest.main()
