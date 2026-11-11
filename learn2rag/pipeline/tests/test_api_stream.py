import asyncio
import json
import unittest
from typing import Any, List
from unittest.mock import MagicMock, patch

from .. import api


class EventStreamTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_keep_alive_while_the_pipeline_works(self) -> None:
        async def slow_pipeline(*args: Any) -> dict[str, str]:
            await asyncio.sleep(0.25)
            return {'answer': 'Preheat the oven.'}

        with patch.object(api, 'KEEP_ALIVE_INTERVAL_SECONDS', 0.05), patch.object(api, 'run_pipeline', slow_pipeline):
            events: List[str] = [event async for event in api.event_stream(MagicMock(), MagicMock(), MagicMock())]

        # Comments first (ignored by clients), then the answer and the end marker
        self.assertGreaterEqual(events.count(': keep-alive\n\n'), 2)
        self.assertEqual(set(events[:-2]), {': keep-alive\n\n'})
        self.assertEqual(json.loads(events[-2].removeprefix('data: '))['choices'][0]['delta']['content'], 'Preheat the oven.')
        self.assertEqual(events[-1], 'data: [DONE]\n\n')

    async def test_fast_answer_has_no_keep_alive(self) -> None:
        async def fast_pipeline(*args: Any) -> dict[str, str]:
            return {'answer': 'Done.'}

        with patch.object(api, 'run_pipeline', fast_pipeline):
            events = [event async for event in api.event_stream(MagicMock(), MagicMock(), MagicMock())]

        self.assertEqual(len(events), 2)

    async def test_client_disconnect_stops_the_pipeline(self) -> None:
        cancelled = asyncio.Event()

        async def endless_pipeline(*args: Any) -> dict[str, str]:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return {'answer': ''}

        with patch.object(api, 'KEEP_ALIVE_INTERVAL_SECONDS', 0.01), patch.object(api, 'run_pipeline', endless_pipeline):
            stream = api.event_stream(MagicMock(), MagicMock(), MagicMock())
            self.assertEqual(await anext(stream), ': keep-alive\n\n')
            # Starlette closes the generator when the client disconnects
            await stream.aclose()

        await asyncio.wait_for(cancelled.wait(), timeout=1)
