import unittest
from unittest.mock import MagicMock, patch

from ..chat import Message
from .. import rewrite


def _history() -> list[Message]:
    return [
        Message(role='user', content='what colors do carrots have?'),
        Message(role='assistant', content='Carrots can be orange, purple, red, yellow, or white.'),
    ]


class ContextualizeQueryTestCase(unittest.TestCase):
    @patch.object(rewrite, 'llm')
    def test_returns_rewritten_query_from_llm(self, mock_llm: MagicMock) -> None:
        mock_llm.invoke.return_value = MagicMock(content='carrot colors')

        result = rewrite.contextualize_query('tell me again', _history(), {})

        self.assertEqual(result, 'carrot colors')
        messages = mock_llm.invoke.call_args.args[0]
        self.assertIn('Carrots can be orange', messages[1].content)
        self.assertIn('tell me again', messages[1].content)

    @patch.object(rewrite, 'llm', None)
    def test_returns_empty_string_when_llm_not_configured(self) -> None:
        result = rewrite.contextualize_query('tell me again', _history(), {})

        self.assertEqual(result, '')

    @patch.object(rewrite, 'llm')
    def test_returns_empty_string_when_history_is_empty(self, mock_llm: MagicMock) -> None:
        result = rewrite.contextualize_query('first question', [], {})

        self.assertEqual(result, '')
        mock_llm.invoke.assert_not_called()

    @patch.dict('os.environ', {'L2R_OLLAMA_MAX_RETRIES': '1'})
    @patch.object(rewrite, 'llm')
    def test_returns_empty_string_on_llm_failure(self, mock_llm: MagicMock) -> None:
        mock_llm.invoke.side_effect = RuntimeError('boom')

        result = rewrite.contextualize_query('tell me again', _history(), {})

        self.assertEqual(result, '')