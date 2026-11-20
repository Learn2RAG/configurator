"""Import the default Main Page from mediawiki/docker-compose.yml with the MediaWikiLoader."""
from collections.abc import Iterator
from typing import Any

import pytest

from .integration import RagEnvironment, docker_service, repo_dir, requires_docker_compose

BASE_URL = 'http://localhost:7338'

# Credentials from mediawiki/opt/entrypoint
USERNAME = 'admin'
PASSWORD = 'uniquetest'

# Text of the Main Page that MediaWiki creates on installation
MAIN_PAGE_TEXT = 'MediaWiki has been installed.'


@pytest.fixture(scope='module')
def mediawiki_instance() -> Iterator[str]:
    # The entrypoint installs the wiki before starting Apache, so any answer means it is ready
    with docker_service(repo_dir / 'mediawiki' / 'docker-compose.yml', BASE_URL) as url:
        yield url


def mediawiki_loader(**auth: str) -> dict[str, Any]:
    return {
        'loader_id': 'mediawiki_test',
        'loader_type': 'MediaWikiLoader',
        'base_url': BASE_URL,
        'namespaces': [0],
        **auth,
    }


@requires_docker_compose
def test_mediawiki_import(mediawiki_instance: str, rag_env: RagEnvironment) -> None:
    loaders = [mediawiki_loader(auth_type='basic', username=USERNAME, password=PASSWORD)]
    project = rag_env.run_import(loaders)
    assert not project.failed, 'import succeeded'

    assert MAIN_PAGE_TEXT in rag_env.imported_text(), 'Main Page content was imported, not just its title'
    assert f'{BASE_URL}/?curid=1' in rag_env.imported_sources(), 'Main Page source URL'

    rag_env.start('pipeline.yml', loaders)
    content = rag_env.ask('What does the main page of the wiki say about the installation of MediaWiki?')
    assert not content.endswith('Information:\n'), 'contains any document chunks in the prompt'
    assert MAIN_PAGE_TEXT in content, 'specific text from the Main Page'


@requires_docker_compose
def test_mediawiki_import_requires_login(mediawiki_instance: str, rag_env: RagEnvironment) -> None:
    # mediawiki/opt/LocalSettings.php denies reading to anonymous users
    project = rag_env.run_import([mediawiki_loader(auth_type='none')])

    assert project.failed, 'import reports the access error'
    assert MAIN_PAGE_TEXT not in rag_env.imported_text(), 'nothing is imported without login'
