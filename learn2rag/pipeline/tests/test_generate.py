from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from learn2rag.pipeline import generate


class FakePrompt:
    def __init__(self, answer: str, formatted_messages=None):
        self.answer = answer
        self.formatted_messages = formatted_messages or ['message']

    def __or__(self, other):
        return self

    def invoke(self, values):
        return SimpleNamespace(content=self.answer)

    def format_messages(self, context=None, question=None):
        return self.formatted_messages


class FakeChunk:
    def __init__(self, text: str):
        self._text = text

    def text(self) -> str:
        return self._text


def test_generate_returns_llm_content() -> None:
    fake_llm = MagicMock()
    fake_prompt = FakePrompt(answer='generated answer')

    with patch('learn2rag.pipeline.generate.ChatPromptTemplate.from_messages', return_value=fake_prompt), \
         patch.object(generate, 'llm', fake_llm):
        result = generate.generate(
            'hello',
            [SimpleNamespace(payload={'path': 'doc', 'content': 'text'})],
            {'prompt': 'System: {context}\nUser: {question}'},
        )

    assert result == 'generated answer'


def test_generate_stream_yields_text_chunks() -> None:
    fake_llm = MagicMock()
    fake_llm.stream.return_value = [FakeChunk('hello'), FakeChunk(' '), FakeChunk('world')]
    fake_prompt = FakePrompt(answer='ignored')

    with patch('learn2rag.pipeline.generate.ChatPromptTemplate.from_messages', return_value=fake_prompt), \
         patch.object(generate, 'llm', fake_llm):
        output = ''.join(
            generate.generate_stream(
                'hello',
                [SimpleNamespace(payload={'path': 'doc', 'content': 'text'})],
                {'prompt': 'System: {context}\nUser: {question}'},
                request_id='req1',
            )
        )

    assert output == 'hello world'
