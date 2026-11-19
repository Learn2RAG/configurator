from collections.abc import Iterator
from pathlib import Path

import pytest

import learn2rag.compose as compose


@pytest.fixture(autouse=True)
def compose_db(tmp_path: Path) -> Iterator[Path]:
    """Give every test its own compose database instead of ./compose.db."""
    previous = compose._db_path
    path = tmp_path / 'compose.db'
    compose.use_database(path)
    yield path
    compose.use_database(previous)
