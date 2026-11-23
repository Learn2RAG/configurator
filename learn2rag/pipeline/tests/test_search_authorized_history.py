import unittest
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, patch

from qdrant_client.http.models import ScoredPoint

from .. import rewrite
from ..search import _Queries, _build_queries, search_authorized


def _base_opt_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "rewrite": "True",
        "rewrite_mode": "history",
        "top_k": 3,
        "n_subqueries": 2,
        "n_keywords": 2,
    }
    config.update(overrides)
    return config


class BuildQueriesHistoryTestCase(unittest.IsolatedAsyncioTestCase):
    """`_build_queries`: which question is the effective one, and what is generated from it."""

    async def _build(
        self,
        opt_config: dict[str, Any],
        history: list[Any],
        contextualize_return: str = 'carrot colors',
    ) -> tuple[_Queries, MagicMock, MagicMock, MagicMock]:
        with patch.object(rewrite, 'contextualize_query', return_value=contextualize_return) as fake_contextualize, \
             patch.object(rewrite, 'generate_subqueries', return_value=['carrot color varieties']) as fake_subqueries, \
             patch.object(rewrite, 'generate_keywords', return_value=['carrot', 'color']) as fake_keywords:
            queries = await _build_queries("tell me again", history, opt_config)
        return queries, fake_contextualize, fake_subqueries, fake_keywords

    async def test_uses_contextualized_query_when_enabled_and_history_present(self) -> None:
        history = [MagicMock()]
        queries, fake_contextualize, _, _ = await self._build(_base_opt_config(), history)

        self.assertEqual(queries.question, 'carrot colors')
        fake_contextualize.assert_called_once_with('tell me again', history, ANY)

    async def test_uses_raw_question_when_history_is_empty(self) -> None:
        queries, fake_contextualize, _, _ = await self._build(_base_opt_config(), [])

        self.assertEqual(queries.question, 'tell me again')
        fake_contextualize.assert_not_called()

    async def test_uses_raw_question_when_history_component_not_selected(self) -> None:
        queries, fake_contextualize, _, _ = await self._build(
            _base_opt_config(rewrite_mode='subqueries_keywords'), [MagicMock()]
        )

        self.assertEqual(queries.question, 'tell me again')
        fake_contextualize.assert_not_called()

    async def test_uses_raw_question_when_rewrite_disabled(self) -> None:
        queries, fake_contextualize, fake_subqueries, fake_keywords = await self._build(
            _base_opt_config(rewrite='False', rewrite_mode='history_subqueries_keywords'), [MagicMock()]
        )

        self.assertEqual(queries, _Queries('tell me again', [], []))
        fake_contextualize.assert_not_called()
        fake_subqueries.assert_not_called()
        fake_keywords.assert_not_called()

    async def test_uses_raw_question_when_contextualize_returns_empty(self) -> None:
        queries, _, _, _ = await self._build(_base_opt_config(), [MagicMock()], contextualize_return='')

        self.assertEqual(queries.question, 'tell me again')

    async def test_history_alone_generates_no_subqueries_or_keywords(self) -> None:
        queries, _, fake_subqueries, fake_keywords = await self._build(_base_opt_config(), [MagicMock()])

        self.assertEqual(queries, _Queries('carrot colors', [], []))
        fake_subqueries.assert_not_called()
        fake_keywords.assert_not_called()

    async def test_subquery_and_keyword_generation_use_contextualized_query(self) -> None:
        opt_config = _base_opt_config(rewrite_mode='history_subqueries_keywords')
        queries, _, fake_subqueries, fake_keywords = await self._build(opt_config, [MagicMock()])

        fake_subqueries.assert_called_once_with('carrot colors', n=opt_config['n_subqueries'])
        fake_keywords.assert_called_once_with('carrot colors', n=opt_config['n_keywords'])
        self.assertEqual(queries, _Queries('carrot colors', ['carrot color varieties'], ['carrot', 'color']))

    async def test_subquery_and_keyword_generation_use_raw_question_without_history(self) -> None:
        opt_config = _base_opt_config(rewrite_mode='history_subqueries_keywords')
        _, fake_contextualize, fake_subqueries, fake_keywords = await self._build(opt_config, [])

        fake_contextualize.assert_not_called()
        fake_subqueries.assert_called_once_with('tell me again', n=opt_config['n_subqueries'])
        fake_keywords.assert_called_once_with('tell me again', n=opt_config['n_keywords'])


class SearchAuthorizedHistoryTestCase(unittest.IsolatedAsyncioTestCase):
    """`search_authorized`: the effective question reaches embedding and reranker, and is built once."""

    def setUp(self) -> None:
        self.user_config = {"collection_name": "test_collection"}
        self._patch("learn2rag.pipeline.search._collection_exists", return_value=True)
        self.embed_queries = self._patch("learn2rag.pipeline.search._embed_queries")
        self._patch("learn2rag.pipeline.search._build_prefetch_tree")
        self.rerank_pool = self._patch(
            "learn2rag.pipeline.search._rerank_pool", side_effect=lambda question, pool, *args, **kwargs: pool)

    def _patch(self, target: str, **kwargs: Any) -> Any:
        patcher = patch(target, **kwargs)
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        return mock

    async def _run(self, opt_config: dict[str, Any], history: list[Any], contextualize_return: str = 'carrot colors') -> MagicMock:
        # Six chunks of six documents: with top_k=3 the lazy check needs two blocks
        pool = _points(range(6))
        auth_filter = MagicMock()
        auth_filter.filter_documents = AsyncMock(side_effect=lambda _auth, documents: set(list(documents)[:1]))
        pools = iter([pool])
        with patch("learn2rag.pipeline.search._retrieve_pool", side_effect=lambda *args, **kwargs: next(pools, [])), \
             patch("learn2rag.pipeline.authorization._get_authorization_filter", return_value=auth_filter), \
             patch.object(rewrite, 'contextualize_query', return_value=contextualize_return) as fake_contextualize:
            await search_authorized(
                "tell me again", {}, history=history, user_config=self.user_config, opt_config=opt_config)
        return fake_contextualize

    async def _run_rounds_without_rights(self, opt_config: dict[str, Any], rounds: list[list[ScoredPoint]]) -> MagicMock:
        """Search with one pool per round; no document is authorized, so every round runs."""
        auth_filter = MagicMock()
        auth_filter.filter_documents = AsyncMock(return_value=set())
        with patch("learn2rag.pipeline.search._retrieve_pool", side_effect=rounds) as retrieve_pool, \
             patch("learn2rag.pipeline.authorization._get_authorization_filter", return_value=auth_filter):
            await search_authorized("tell me again", {}, user_config=self.user_config, opt_config=opt_config)
        return retrieve_pool

    async def test_embedding_and_reranker_get_contextualized_question(self) -> None:
        await self._run(_base_opt_config(reranking="True"), [MagicMock()])

        self.assertEqual(self.embed_queries.call_args.args[0].question, 'carrot colors')
        self.assertEqual(self.rerank_pool.call_args.args[0], 'carrot colors')

    async def test_embedding_and_reranker_get_raw_question_without_history(self) -> None:
        fake_contextualize = await self._run(_base_opt_config(reranking="True"), [])

        self.assertEqual(self.embed_queries.call_args.args[0].question, 'tell me again')
        self.assertEqual(self.rerank_pool.call_args.args[0], 'tell me again')
        fake_contextualize.assert_not_called()

    async def test_embedding_and_reranker_get_raw_question_when_contextualize_returns_empty(self) -> None:
        await self._run(_base_opt_config(reranking="True"), [MagicMock()], contextualize_return='')

        self.assertEqual(self.embed_queries.call_args.args[0].question, 'tell me again')
        self.assertEqual(self.rerank_pool.call_args.args[0], 'tell me again')

    async def test_contextualize_called_once_across_authorization_blocks(self) -> None:
        fake_contextualize = await self._run(_base_opt_config(), [MagicMock()])

        fake_contextualize.assert_called_once()

    async def test_rewrite_and_embeddings_run_once_across_rounds(self) -> None:
        opt_config = _base_opt_config(rewrite_mode="subqueries_keywords", max_search_rounds=3)
        rounds = [_points(range(0, 3)), _points(range(3, 6)), _points(range(6, 9))]
        with patch.object(rewrite, 'generate_subqueries', return_value=['carrot color varieties']) as fake_subqueries, \
             patch.object(rewrite, 'generate_keywords', return_value=['carrot']) as fake_keywords:
            retrieve_pool = await self._run_rounds_without_rights(opt_config, rounds)

        fake_subqueries.assert_called_once()
        fake_keywords.assert_called_once()
        self.embed_queries.assert_called_once()
        self.assertEqual(retrieve_pool.call_count, 3)

    async def test_reranker_gets_only_the_pool_of_its_round(self) -> None:
        rounds = [_points(range(0, 3)), _points(range(3, 6)), []]
        await self._run_rounds_without_rights(_base_opt_config(rewrite="False", reranking="True"), rounds)

        self.assertEqual([call.args[1] for call in self.rerank_pool.call_args_list], rounds[:2])


def _points(ids: range) -> list[ScoredPoint]:
    """One chunk per document."""
    return [
        ScoredPoint(id=i, score=1.0 - i * 0.01, version=1,
                    payload={"loader_id": "loader", "document_id": f"doc {i}", "content": f"doc {i}"})
        for i in ids
    ]


if __name__ == "__main__":
    unittest.main()
