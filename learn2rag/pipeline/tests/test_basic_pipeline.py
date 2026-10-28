import unittest
from unittest.mock import AsyncMock, patch

from ..operators.basic_pipeline import BasicPipeline


class BasicPipelineTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_passes_history_to_search_and_generation_operators(self) -> None:
        history = [object()]
        with patch('learn2rag.pipeline.operators.basic_pipeline.SearchOperator') as fake_search_cls, \
             patch('learn2rag.pipeline.operators.basic_pipeline.GenerationOperator') as fake_gen_cls:
            fake_search_cls.return_value = AsyncMock(return_value={'documents': ['doc']})
            fake_gen_cls.return_value = AsyncMock(return_value={'answer': 'the answer'})

            result = await BasicPipeline()(inputs={
                'question': 'tell me again',
                'user_auths': {},
                'history': history,
            })

        search_inputs = fake_search_cls.return_value.call_args.kwargs['inputs']
        gen_inputs = fake_gen_cls.return_value.call_args.kwargs['inputs']
        self.assertEqual(search_inputs['history'], history)
        self.assertEqual(gen_inputs['history'], history)
        self.assertEqual(result, {'answer': 'the answer', 'documents': ['doc']})

    async def test_defaults_history_to_empty_tuple_when_missing(self) -> None:
        with patch('learn2rag.pipeline.operators.basic_pipeline.SearchOperator') as fake_search_cls, \
             patch('learn2rag.pipeline.operators.basic_pipeline.GenerationOperator') as fake_gen_cls:
            fake_search_cls.return_value = AsyncMock(return_value={'documents': []})
            fake_gen_cls.return_value = AsyncMock(return_value={'answer': 'the answer'})

            await BasicPipeline()(inputs={'question': 'first question', 'user_auths': {}})

        search_inputs = fake_search_cls.return_value.call_args.kwargs['inputs']
        gen_inputs = fake_gen_cls.return_value.call_args.kwargs['inputs']
        self.assertEqual(search_inputs['history'], ())
        self.assertEqual(gen_inputs['history'], ())