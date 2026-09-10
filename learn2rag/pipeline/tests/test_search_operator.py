import unittest
from unittest.mock import AsyncMock, patch

from ..operators.search import SearchOperator


class SearchOperatorTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_passes_history_to_search_authorized(self) -> None:
        history = [object()]
        with patch(
            'learn2rag.pipeline.operators.search.search_authorized',
            new=AsyncMock(return_value=['doc']),
        ) as fake_search:
            result = await SearchOperator()(inputs={
                'question': 'tell me again',
                'user_auths': {},
                'history': history,
            })

        self.assertEqual(result['documents'], ['doc'])
        fake_search.assert_called_once_with(question='tell me again', user_auths={}, history=history)

    async def test_defaults_history_to_empty_tuple_when_missing(self) -> None:
        with patch(
            'learn2rag.pipeline.operators.search.search_authorized',
            new=AsyncMock(return_value=[]),
        ) as fake_search:
            await SearchOperator()(inputs={'question': 'first question', 'user_auths': {}})

        fake_search.assert_called_once_with(question='first question', user_auths={}, history=())