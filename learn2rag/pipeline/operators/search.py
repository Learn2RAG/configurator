from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypedDict
import asyncio

from ..prov import Prov
from .base import BaseOperator
from ..search import search_authorized

Inputs = TypedDict('Inputs', {
    'question': str,
    'user': str,
}, total=True)

Outputs = TypedDict('Outputs', {
    'documents': Any,
}, total=True)


class SearchOperator(BaseOperator):
    async def run(self, inputs: Inputs, prov: Prov) -> Outputs:
        documents = await search_authorized(question=inputs['question'], user=inputs['user'])
        return {
            'documents': documents,
        }
