from operator import itemgetter
from typing import Any, TypedDict

from ..prov import Prov
from .base import BaseOperator
from .search import SearchOperator
from .generation import GenerationOperator

Inputs = TypedDict('Inputs', {
    'question': str,
    'user': str | None,
}, total=True)

Outputs = TypedDict('Outputs', {
    'answer': str,
    'documents': Any,
}, total=True)


class BasicPipeline(BaseOperator):
    async def run(self, inputs: Inputs, prov: Prov) -> Outputs:
        documents = itemgetter('documents')(await SearchOperator()(
            inputs={
                'question': inputs['question'],
                'user': inputs['user'],
            },
            prov=prov,
        ))
        answer = itemgetter('answer')(await GenerationOperator()(
            inputs={
                'question': inputs['question'],
                'documents': documents,
            },
            prov=prov,
        ))
        return {
            'answer': answer,
            'documents': documents,
        }
