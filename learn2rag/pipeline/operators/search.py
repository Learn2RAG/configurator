from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, TypedDict
import asyncio

from ..prov import Prov
from .base import BaseOperator
from ..search import search_authorized

Inputs = TypedDict('Inputs', {
    'question': str,
    'user_auths': Mapping[str, Any],
}, total=True)

Outputs = TypedDict('Outputs', {
    'documents': Any,
}, total=True)


class SearchOperator(BaseOperator):
    async def run(self, inputs: Inputs, prov: Prov) -> Outputs:
        documents = await search_authorized(question=inputs['question'], user_auths=inputs['user_auths'])
        return {
            'documents': documents,
        }
