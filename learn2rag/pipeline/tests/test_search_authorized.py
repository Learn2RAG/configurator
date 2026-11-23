import copy
import unittest
from typing import Any, Mapping, Sequence, cast
from unittest.mock import AsyncMock, patch

import numpy as np
from qdrant_client import models
from qdrant_client.http.models import ScoredPoint

from ..search import (
    _Embedding,
    _Queries,
    _QueryEmbeddings,
    _document_exclusion,
    _exclusion_filter,
    _rerank_pool,
    search_authorized,
)

LOADER_ID = "loader"
SEARCH_LOGGER = "learn2rag.pipeline.search"


def _pool(*doc_ids: str | None, first_id: int = 1, score: float = 1.0) -> list[ScoredPoint]:
    """One point per entry in rank order with the IDs `first_id`, `first_id + 1`, ...; `None` is a point without payload."""
    return [
        ScoredPoint(
            id=first_id + i,
            score=score - i * 0.01,
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


class _SearchAuthorizedBase(unittest.IsolatedAsyncioTestCase):
    """Qdrant, embeddings and reranker are mocked; the authorization filter is a `_FakeFilter`."""

    def setUp(self) -> None:
        self.user_auths = {LOADER_ID: "user-token"}
        self.user_config = {"collection_name": "test_collection"}
        self.opt_config: dict[str, Any] = {"top_k": 3}

        self.build_queries = self._patch(
            "learn2rag.pipeline.search._build_queries", new=AsyncMock(return_value=_Queries("test query", [], [])))
        self.embed_queries = self._patch("learn2rag.pipeline.search._embed_queries")
        self.build_tree = self._patch("learn2rag.pipeline.search._build_prefetch_tree")
        self.retrieve_pool = self._patch("learn2rag.pipeline.search._retrieve_pool", return_value=[])
        self.rerank_pool = self._patch(
            "learn2rag.pipeline.search._rerank_pool", side_effect=lambda question, pool, *args, **kwargs: pool)

    def _patch(self, target: str, **kwargs: Any) -> Any:
        patcher = patch(target, **kwargs)
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        return mock

    async def _run(
        self, *rounds: list[ScoredPoint], allowed: set[str], **filter_kwargs: Any,
    ) -> tuple[list[ScoredPoint], _FakeFilter]:
        """Search with one pool per round; Qdrant returns no candidates after the given rounds."""
        self.retrieve_pool.reset_mock()
        self.build_tree.reset_mock()
        pools = iter(rounds)
        self.retrieve_pool.side_effect = lambda *args, **kwargs: next(pools, [])
        auth_filter = _FakeFilter(allowed, **filter_kwargs)
        with patch("learn2rag.pipeline.authorization._get_authorization_filter", return_value=auth_filter):
            results = await search_authorized(
                "test query", self.user_auths, user_config=self.user_config, opt_config=self.opt_config)
        return results, auth_filter

    def _exclusions(self) -> list[models.Filter | None]:
        """The exclusion filter of each round."""
        return [call.args[3] for call in self.build_tree.call_args_list]


class SearchAuthorizedTestCase(_SearchAuthorizedBase):
    """Sorting and lazy authorization check within a round."""

    async def test_blocks_of_a_round_use_its_pool(self) -> None:
        results, auth_filter = await self._run(_pool("a", "b", "c", "d", "e", "f", "g"), allowed={"g"})

        self.assertEqual(_doc_ids(results), ["g"])
        self.assertEqual(len(auth_filter.calls), 3)
        # The second round only finds that the collection is exhausted.
        self.assertEqual(self.retrieve_pool.call_count, 2)
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


class SearchRoundsTestCase(_SearchAuthorizedBase):
    """Follow-up rounds when fewer than `top_k` candidates of a round are authorized."""

    async def test_enough_in_round_one_searches_once_without_exclusion(self) -> None:
        with self.assertLogs("profiling", "INFO") as logs:
            results, _ = await self._run(_pool(*"abcdef"), _pool(*"ghi", first_id=7), allowed=set("abcdefghi"))

        self.assertEqual(_doc_ids(results), ["a", "b", "c"])
        self.retrieve_pool.assert_called_once()
        self.assertEqual(self._exclusions(), [None])
        self.assertIn("rounds=1 stop_reason=top_k", logs.output[-1])

    async def test_restrictive_user_gets_round_one_first_then_round_two(self) -> None:
        self.opt_config["top_k"] = 5
        round_1 = _pool("a", "b", "x", "y", "z")
        # Higher scores than round 1: the order still follows the rounds.
        round_2 = _pool("c", "d", "e", "f", first_id=6, score=2.0)
        allowed = {"a", "b", "c", "d", "e", "f"}

        for reranking, expected in ((False, ["a", "b", "c", "d", "e"]), (True, ["b", "a", "f", "e", "d"])):
            with self.subTest(reranking=reranking):
                if reranking:
                    self.rerank_pool.side_effect = lambda question, pool, *args, **kwargs: list(reversed(pool))
                results, _ = await self._run(round_1, round_2, allowed=allowed)

                self.assertEqual(_doc_ids(results), expected)
                self.assertEqual(self.retrieve_pool.call_count, 2)

    async def test_user_without_rights_stops_after_max_search_rounds(self) -> None:
        rounds = [_pool("a", "b", first_id=1), _pool("c", "d", first_id=3), _pool("e", "f", first_id=5),
                  _pool("g", "h", first_id=7)]
        with self.assertLogs("profiling", "INFO") as logs:
            results, _ = await self._run(*rounds, allowed=set())

        self.assertEqual(results, [])
        self.assertEqual(self.retrieve_pool.call_count, 3)
        self.assertEqual(len([line for line in logs.output if "search_round" in line]), 3)
        self.assertIn("rounds=3 stop_reason=max_rounds", logs.output[-1])

    async def test_empty_round_ends_the_search(self) -> None:
        with self.assertLogs("profiling", "INFO") as logs:
            results, _ = await self._run(_pool("a", "b"), allowed={"a"})

        self.assertEqual(_doc_ids(results), ["a"])
        self.assertEqual(self.retrieve_pool.call_count, 2)
        self.rerank_pool.assert_called_once()
        self.assertIn("search_round round=2 pool=0", "\n".join(logs.output))
        self.assertIn("rounds=2 stop_reason=exhausted", logs.output[-1])

    async def test_max_search_rounds(self) -> None:
        rounds = [_pool(doc_id, first_id=i + 1) for i, doc_id in enumerate("abcd")]
        cases: list[tuple[dict[str, Any], int, bool]] = [
            ({}, 3, False),
            ({"max_search_rounds": 1}, 1, False),
            ({"max_search_rounds": 2}, 2, False),
            *(({"max_search_rounds": value}, 3, True) for value in (0, -1, True, 2.5, "3")),
        ]
        for config, expected_rounds, warns in cases:
            with self.subTest(**config):
                self.opt_config = {"top_k": 3, **config}
                if warns:
                    with self.assertLogs(SEARCH_LOGGER, "WARNING") as logs:
                        await self._run(*rounds, allowed=set())
                    self.assertIn("max_search_rounds", logs.output[0])
                    self.assertIn(repr(config["max_search_rounds"]), logs.output[0])
                else:
                    with self.assertNoLogs(SEARCH_LOGGER, "WARNING"):
                        await self._run(*rounds, allowed=set())

                self.assertEqual(self.retrieve_pool.call_count, expected_rounds)

    async def test_invalid_parameters_warn_once_per_request(self) -> None:
        self.opt_config.update(max_search_rounds=0, fusion_weights={"original": 0})
        rounds = [_pool(doc_id, first_id=i + 1) for i, doc_id in enumerate("abc")]
        with self.assertLogs(SEARCH_LOGGER, "WARNING") as logs:
            await self._run(*rounds, allowed=set())

        self.assertEqual(self.retrieve_pool.call_count, 3)
        self.assertEqual(len(logs.output), 2)

    async def test_document_allowed_in_round_one_is_not_checked_again(self) -> None:
        results, auth_filter = await self._run(_pool("d", "x", "y"), _pool("d", "e", first_id=4), allowed={"d", "e"})

        self.assertEqual(_doc_ids(results), ["d", "d", "e"])
        self.assertEqual(auth_filter.calls, [["d", "x", "y"], ["e"]])

    async def test_rejected_document_returned_again_is_neither_authorized_nor_checked_again(self) -> None:
        # The exclusion is only an optimization: if Qdrant returns a rejected document again,
        # the authorization check still rejects it.
        results, auth_filter = await self._run(_pool("x", "a"), _pool("x", "b", first_id=3), allowed={"a", "b"})

        self.assertEqual(_doc_ids(results), ["a", "b"])
        self.assertEqual(auth_filter.calls, [["x", "a"], ["b"]])

    async def test_each_round_excludes_delivered_points_and_rejected_documents(self) -> None:
        round_1 = _pool("a", "x", first_id=1)
        round_2 = _pool("b", "y", first_id=3)
        await self._run(round_1, round_2, _pool("z", first_id=5), allowed={"a", "b"})

        x, y = round_1[1], round_2[1]
        self.assertEqual(self._exclusions(), [
            None,
            _exclusion_filter([1, 2], [cast(models.Filter, _document_exclusion(x))]),
            _exclusion_filter([1, 2, 3, 4], [cast(models.Filter, _document_exclusion(x)),
                                             cast(models.Filter, _document_exclusion(y))]),
        ])


class ColbertRerankingTestCase(_SearchAuthorizedBase):
    """The ColBERT query vector of the reranker is computed once per request."""

    ROUNDS = (_pool("a", first_id=1), _pool("b", first_id=2), _pool("c", first_id=3))

    def setUp(self) -> None:
        super().setUp()
        self.opt_config.update(
            reranking="True", reranking_mode="reranking_with_colbert", embedding_model="BAAI/bge-m3")
        # The real `_rerank_pool`, only the ColBERT model and Qdrant are mocked.
        self.rerank_pool.side_effect = _rerank_pool
        self.create_embeddings = self._patch("learn2rag.pipeline.search.create_embeddings")
        self.rerank_colbert = self._patch(
            "learn2rag.pipeline.search._rerank_points_with_colbert",
            side_effect=lambda query, points, **kwargs: points)

    def _colbert_queries(self) -> list[Any]:
        return [call.kwargs["colbert_query"] for call in self.rerank_colbert.call_args_list]

    async def test_vector_is_computed_once_for_all_rounds(self) -> None:
        self.opt_config["search_mode"] = "dense_sparse"
        self.embed_queries.return_value = _QueryEmbeddings(_Embedding(), [], [])
        colbert_query = [np.array([0.1, 0.2])]
        self.create_embeddings.return_value = {"colbert_vecs": [colbert_query]}

        await self._run(*self.ROUNDS, allowed=set())

        self.create_embeddings.assert_called_once_with(["test query"], "BAAI/bge-m3", embedding_mode="colbert")
        self.assertEqual(self._colbert_queries(), [colbert_query] * 3)

    async def test_vector_of_the_search_is_reused(self) -> None:
        self.opt_config["search_mode"] = "dense_sparse_colbert"
        colbert_query = [np.array([0.3, 0.4])]
        self.embed_queries.return_value = _QueryEmbeddings(_Embedding(colbert=colbert_query), [], [])

        await self._run(*self.ROUNDS, allowed=set())

        self.create_embeddings.assert_not_called()
        self.assertEqual(self._colbert_queries(), [colbert_query] * 3)


def _point(payload: dict[str, Any] | None, point_id: int = 1) -> ScoredPoint:
    return ScoredPoint(id=point_id, score=1.0, version=1, payload=payload)


def _match(key: str, value: Any) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _empty(key: str) -> models.Filter:
    return models.Filter(should=[models.IsEmptyCondition(is_empty=models.PayloadField(key=key)), _match(key, "")])


class ExclusionFilterTestCase(unittest.TestCase):
    """The exclusion of a rejected document matches `loader_id` and the field its document ID comes from."""

    def test_points_are_excluded_by_id(self) -> None:
        self.assertEqual(
            _exclusion_filter([1, "a1b2"], []), models.Filter(must_not=[models.HasIdCondition(has_id=[1, "a1b2"])]))

    def test_document_whose_id_is_the_point_id_needs_no_own_condition(self) -> None:
        for payload in (None, {"loader_id": LOADER_ID, "document_id": "", "source": None}):
            with self.subTest(payload=payload):
                self.assertIsNone(_document_exclusion(_point(payload)))

    def test_id_from_document_id(self) -> None:
        point = _point({"loader_id": LOADER_ID, "document_id": "X", "source": "S"})

        self.assertEqual(
            _document_exclusion(point), models.Filter(must=[_match("loader_id", LOADER_ID), _match("document_id", "X")]))

    def test_id_from_source_requires_an_empty_document_id(self) -> None:
        expected = models.Filter(must=[_match("loader_id", LOADER_ID), _match("source", "S"), _empty("document_id")])
        payloads: list[dict[str, Any]] = [
            {"loader_id": LOADER_ID, "document_id": "", "source": "S"},
            {"loader_id": LOADER_ID, "document_id": None, "source": "S"},
            {"loader_id": LOADER_ID, "source": "S"},
        ]
        for payload in payloads:
            with self.subTest(document_id=payload.get("document_id", "missing")):
                self.assertEqual(_document_exclusion(_point(payload)), expected)

    def test_id_from_node_id_requires_empty_document_id_and_source(self) -> None:
        point = _point({"loader_id": LOADER_ID, "document_id": "", "source": "", "node_id": "N"})

        self.assertEqual(_document_exclusion(point), models.Filter(must=[
            _match("loader_id", LOADER_ID), _match("node_id", "N"), _empty("document_id"), _empty("source"),
        ]))

    def test_numeric_document_id_keeps_its_type(self) -> None:
        point = _point({"loader_id": LOADER_ID, "document_id": 123, "source": "S"})

        self.assertEqual(
            _document_exclusion(point), models.Filter(must=[_match("loader_id", LOADER_ID), _match("document_id", 123)]))

    def test_value_qdrant_cannot_match_is_left_out(self) -> None:
        for document_id in (1.5, ["a", "b"]):
            with self.subTest(document_id=document_id):
                self.assertIsNone(_document_exclusion(_point({"loader_id": LOADER_ID, "document_id": document_id})))


if __name__ == "__main__":
    unittest.main()
