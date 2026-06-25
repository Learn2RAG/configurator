from typing import TypedDict
import logging

from ..prov import Prov
from .base import BaseOperator

logger = logging.getLogger(__name__)

Inputs = TypedDict('Inputs', {
    'question': str,
}, total=True)

Outputs = TypedDict('Outputs', {
    'value': int,
}, total=True)


class ExampleOperator(BaseOperator):
    async def run(self, inputs: Inputs, prov: Prov) -> Outputs:
        logger.info('Amount of steps: %i', len(prov.items))
        logger.info('Last step: %s', prov.items[-1].label)
        logger.info('Amount of search steps: %i', sum(1 for item in prov.items if item.label == 'SearchOperator'))
        return {
            'value': 42,
        }
