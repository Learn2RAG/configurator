import logging
from collections.abc import Iterator

import pytest
from _pytest.logging import LogCaptureFixture

from .integration import RagEnvironment


@pytest.fixture
def rag_env(caplog: LogCaptureFixture) -> Iterator[RagEnvironment]:
    """A clean storage path and project for running the import / pipeline templates."""
    caplog.set_level(logging.WARNING, logger='httpcore')
    caplog.set_level(logging.WARNING, logger='httpx')
    caplog.set_level(logging.WARNING, logger='openai')
    env = RagEnvironment()
    env.setup()
    yield env
    env.teardown()
