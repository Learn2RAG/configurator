from typing import Any, TypedDict

from .base import BaseOperator
from ..prov import Prov
from ..config import opt_config
from ..generate import generate

Inputs = TypedDict('Inputs', {
    'question': str,
    'documents': Any,
}, total=True)

Outputs = TypedDict('Outputs', {
    'answer': str,
}, total=True)


class GenerationOperator(BaseOperator):
    async def run(self, inputs: Inputs, prov: Prov) -> Outputs:
        answer = generate(inputs['question'], inputs['documents'], opt_config)
        return {
            'answer': answer,
        }
