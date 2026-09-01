from pathlib import Path
from typing import Generator

from flask import Response
from pygtail import Pygtail  # type: ignore[import-untyped]


def wrap_as_sse(data: str) -> str:
    return '\n'.join(map(lambda line: 'data: ' + line, data.split('\n'))) + '\n\n'


def tail_generator(file: Path) -> Generator[str, None, None]:
    yield from Pygtail(
        str(file),
        full_lines=True,
        read_from_end=True,
        save_on_end=False,
    )


def make_sse_response(generator: Generator[str, None, None]) -> Response:
    return Response(
        map(wrap_as_sse, generator),
        mimetype='text/event-stream',
    )


def tail_sse_response(file: Path) -> Response:
    return make_sse_response(tail_generator(file))
