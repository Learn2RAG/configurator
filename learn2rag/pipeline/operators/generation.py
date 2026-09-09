from typing import Any, NotRequired, Sequence, TypedDict

from .base import BaseOperator
from ..chat import Message
from ..prov import Prov
from ..config import opt_config
from ..generate import generate

Inputs = TypedDict('Inputs', {
    'question': str,
    'documents': Any,
    'history': NotRequired[Sequence[Message]],
}, total=True)

Outputs = TypedDict('Outputs', {
    'answer': str,
}, total=True)


class GenerationOperator(BaseOperator):
    async def run(self, inputs: Inputs, prov: Prov) -> Outputs:
        answer = generate(inputs['question'], inputs['documents'], opt_config, inputs.get('history', ()))
        return {
            'answer': answer,
        }
