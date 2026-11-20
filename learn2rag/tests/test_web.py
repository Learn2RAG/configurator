"""Import a small static site from web/docker-compose.yml (nginx) with the HTMLLoader."""
from collections.abc import Iterator
from typing import Any

import pytest

from .integration import RagEnvironment, docker_service, repo_dir, requires_docker_compose

BASE_URL = 'http://localhost:7339'

# One sentence from each page in web/html/
INDEX_TEXT = 'The ferry to Fernholm leaves the harbour of Kesslund every Tuesday at dawn.'
PAGE2_TEXT = 'Its lamp is tended by the keeper Marta Lindqvist'
PAGE3_TEXT = 'the keeper grows blue potatoes and sea kale'
OUTSIDE_URL = 'http://localhost:9/outside.html'


@pytest.fixture(scope='module')
def web_instance() -> Iterator[str]:
    with docker_service(repo_dir / 'web' / 'docker-compose.yml', f'{BASE_URL}/index.html') as url:
        yield url


def html_loader(depth: int) -> dict[str, Any]:
    return {
        'loader_id': 'web_test',
        'loader_type': 'HTMLLoader',
        'url': f'{BASE_URL}/index.html',
        'depth': depth,
    }


@requires_docker_compose
@pytest.mark.parametrize(('depth', 'expected', 'not_expected'), [
    (0, [INDEX_TEXT], [PAGE2_TEXT, PAGE3_TEXT]),
    (1, [INDEX_TEXT, PAGE2_TEXT], [PAGE3_TEXT]),
    (-1, [INDEX_TEXT, PAGE2_TEXT, PAGE3_TEXT], []),
])
def test_web_import_depth(web_instance: str, rag_env: RagEnvironment, depth: int, expected: list[str], not_expected: list[str]) -> None:
    project = rag_env.run_import([html_loader(depth)])
    assert not project.failed, 'import succeeded'

    text = rag_env.imported_text()
    for sentence in expected:
        assert sentence in text, f'depth {depth} imports {sentence!r}'
    for sentence in not_expected:
        assert sentence not in text, f'depth {depth} does not import {sentence!r}'

    if depth == -1:
        assert OUTSIDE_URL not in rag_env.imported_sources(), 'a whole-site crawl stays on the site'


@requires_docker_compose
def test_web_pipeline(web_instance: str, rag_env: RagEnvironment) -> None:
    loaders = [html_loader(-1)]
    assert not rag_env.run_import(loaders).failed, 'import succeeded'

    rag_env.start('pipeline.yml', loaders)
    content = rag_env.ask('Who tends the lamp of the Fernholm lighthouse?')
    assert not content.endswith('Information:\n'), 'contains any document chunks in the prompt'
    assert PAGE2_TEXT in content, 'specific text from a crawled page'
