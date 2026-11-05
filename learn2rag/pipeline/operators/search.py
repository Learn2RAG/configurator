from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypedDict
import asyncio

from ..prov import Prov
from .base import BaseOperator
from ..search import search_authorized, search_authorized_with_config

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

    async def run_with_config(
        self,
        inputs: Inputs,
        *,
        user_config: dict[str, Any],
        opt_config: dict[str, Any],
    ) -> Outputs:
        """Use the standard authorized path with request-local configuration."""
        documents = await search_authorized_with_config(
            question=inputs['question'],
            user=inputs['user'],
            request_user_config=user_config,
            request_opt_config=opt_config,
        )
        return {
            'documents': documents,
        }
