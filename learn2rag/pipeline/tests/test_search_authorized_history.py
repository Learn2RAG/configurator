import unittest
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, patch

from qdrant_client.http.models import ScoredPoint

from .. import rewrite
from ..search import search_authorized


def _make_points(n: int) -> list[ScoredPoint]:
    return [
        ScoredPoint(id=i + 1, score=1.0 - i * 0.01, version=1, payload={"content": f"doc {i}"})
        for i in range(n)
    ]


def _base_opt_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "rewrite": "True",
        "rewrite_mode": "history",
        "top_k": 3,
        "top_k_subqueries": 3,
        "top_k_keywords": 3,
        "n_subqueries": 2,
        "n_keywords": 2,
        "reranking": "False",
        "max_auth_retries": 1,
        "auth_oversample_start": 1,
        "auth_oversample_step": 1,
    }
    config.update(overrides)
    return config


class SearchAuthorizedHistoryRewriteTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.user_config = {"collection_name": "test_collection"}

    async def _run(self, opt_config: dict[str, Any], history: list[Any], contextualize_return: str = 'carrot colors') -> tuple[MagicMock, MagicMock]:
        with patch('learn2rag.pipeline.search._collect_query_points') as fake_collect, \
             patch('learn2rag.pipeline.search.filter_authorized', new=AsyncMock(side_effect=lambda _auths, resp: resp.points)), \
             patch.object(rewrite, 'contextualize_query', return_value=contextualize_return) as fake_contextualize:
            fake_collect.return_value = _make_points(3)
            await search_authorized(
                "tell me again",
                {},
                history=history,
                user_config=self.user_config,
                opt_config=opt_config,
            )
            return fake_collect, fake_contextualize

    async def test_uses_contextualized_query_when_enabled_and_history_present(self) -> None:
        history = [MagicMock()]
        fake_collect, fake_contextualize = await self._run(_base_opt_config(), history)

        self.assertEqual(fake_collect.call_args_list[0].args[0], 'carrot colors')
        fake_contextualize.assert_called_once_with('tell me again', history, ANY)

    async def test_uses_raw_question_when_history_is_empty(self) -> None:
        fake_collect, fake_contextualize = await self._run(_base_opt_config(), [])

        self.assertEqual(fake_collect.call_args_list[0].args[0], 'tell me again')
        fake_contextualize.assert_not_called()

    async def test_uses_raw_question_when_history_component_not_selected(self) -> None:
        fake_collect, fake_contextualize = await self._run(
            _base_opt_config(rewrite_mode='subqueries_keywords'), [MagicMock()]
        )

        self.assertEqual(fake_collect.call_args_list[0].args[0], 'tell me again')
        fake_contextualize.assert_not_called()

    async def test_uses_raw_question_when_rewrite_disabled(self) -> None:
        fake_collect, fake_contextualize = await self._run(
            _base_opt_config(rewrite='False'), [MagicMock()]
        )

        self.assertEqual(fake_collect.call_args_list[0].args[0], 'tell me again')
        fake_contextualize.assert_not_called()

    async def test_uses_raw_question_when_contextualize_returns_empty(self) -> None:
        fake_collect, _ = await self._run(_base_opt_config(), [MagicMock()], contextualize_return='')

        self.assertEqual(fake_collect.call_args_list[0].args[0], 'tell me again')

    async def test_contextualize_called_once_across_retries(self) -> None:
        opt_config = _base_opt_config(max_auth_retries=3, top_k=3, auth_oversample_start=1, auth_oversample_step=1)
        history = [MagicMock()]
        call_count = {'n': 0}

        async def filter_side_effect(_auths: Any, resp: Any) -> list[Any]:
            call_count['n'] += 1
            points: list[Any] = resp.points
            return points if call_count['n'] >= 2 else points[:1]

        with patch('learn2rag.pipeline.search._collect_query_points') as fake_collect, \
             patch('learn2rag.pipeline.search.filter_authorized', new=AsyncMock(side_effect=filter_side_effect)), \
             patch.object(rewrite, 'contextualize_query', return_value='carrot colors') as fake_contextualize:
            fake_collect.return_value = _make_points(5)

            await search_authorized(
                "tell me again", {}, history=history, user_config=self.user_config, opt_config=opt_config,
            )

        self.assertGreaterEqual(fake_collect.call_count, 2)
        for call in fake_collect.call_args_list:
            self.assertEqual(call.args[0], 'carrot colors')
        fake_contextualize.assert_called_once()

    async def test_subquery_and_keyword_generation_use_contextualized_query(self) -> None:
        """Task 3.3: when 'history' is combined with 'subqueries'/'keywords', _collect_query_points
        (exercised here without mocking it) must pass the contextualized query on to subquery/keyword
        generation, not the raw follow-up text."""
        opt_config = _base_opt_config(
            rewrite_mode='history_subqueries_keywords',
            reranking='False',
        )
        history = [MagicMock()]

        with patch('learn2rag.pipeline.search.search') as fake_search, \
             patch('learn2rag.pipeline.search.filter_authorized', new=AsyncMock(side_effect=lambda _auths, resp: resp.points)), \
             patch.object(rewrite, 'contextualize_query', return_value='carrot colors'), \
             patch.object(rewrite, 'generate_subqueries', return_value=['carrot color varieties']) as fake_subqueries, \
             patch.object(rewrite, 'generate_keywords', return_value=['carrot', 'color']) as fake_keywords:
            fake_search.return_value = MagicMock(points=_make_points(1))

            await search_authorized(
                "tell me again", {}, history=history, user_config=self.user_config, opt_config=opt_config,
            )

        fake_subqueries.assert_called_once_with('carrot colors', n=opt_config['n_subqueries'])
        fake_keywords.assert_called_once_with('carrot colors', n=opt_config['n_keywords'])