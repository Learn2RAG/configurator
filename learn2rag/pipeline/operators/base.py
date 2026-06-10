import abc
import logging
import time
from typing import Any

from ..prov import Activity, Prov

profilingLogger = logging.getLogger('profiling')


class BaseOperator(abc.ABC):
    async def __call__(self, inputs: Any, prov: Prov | None = None) -> Any:
        if prov is None:
            prov = Prov()
        label = self.__class__.__name__
        startedAtTime = time.time()
        profilingLogger.info('start', extra={'activity': label, 'request_id': prov.id})
        outputs = await self.run(inputs=inputs, prov=prov)
        endedAtTime = time.time()
        profilingLogger.info('end', extra={'activity': label, 'request_id': prov.id})
        prov.append(Activity(
            label=label,
            startedAtTime=startedAtTime,
            endedAtTime=endedAtTime,
        ))
        return outputs

    @abc.abstractmethod
    async def run(self, inputs: Any, prov: Prov) -> Any:
        pass
