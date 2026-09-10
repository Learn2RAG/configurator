from operator import itemgetter
from typing import Any, Mapping, NotRequired, Sequence, TypedDict

from ..chat import Message
from ..prov import Prov
from .base import BaseOperator
from .search import SearchOperator
from .generation import GenerationOperator

Inputs = TypedDict('Inputs', {
    'question': str,
    'user_auths': Mapping[str, Any] | None,
    # Callers without a conversation, for example the optimization scripts,
    # may leave the history out.
    'history': NotRequired[Sequence[Message]],
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
                'user_auths': inputs.get('user_auths', {}),
                'history': inputs.get('history', ()),
            },
            prov=prov,
        ))
        answer = itemgetter('answer')(await GenerationOperator()(
            inputs={
                'question': inputs['question'],
                'documents': documents,
                'history': inputs.get('history', ()),
            },
            prov=prov,
        ))
        return {
            'answer': answer,
            'documents': documents,
        }
