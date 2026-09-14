from typing import Any, Mapping, NotRequired, Sequence, TypedDict

from ..chat import Message
from ..prov import Prov
from .base import BaseOperator
from ..search import search_authorized

Inputs = TypedDict('Inputs', {
    'question': str,
    'user_auths': Mapping[str, Any],
    # Callers without a conversation, for example the optimization scripts,
    # may leave the history out.
    'history': NotRequired[Sequence[Message]],
}, total=True)

Outputs = TypedDict('Outputs', {
    'documents': Any,
}, total=True)


class SearchOperator(BaseOperator):
    async def run(self, inputs: Inputs, prov: Prov) -> Outputs:
        documents = await search_authorized(
            question=inputs['question'],
            user_auths=inputs['user_auths'],
            history=inputs.get('history', ()),
        )
        return {
            'documents': documents,
        }
