"""Helpers for integration tests that import from real sources running in `docker compose`."""
import contextlib
import logging
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from openai import APIConnectionError, OpenAI
from qdrant_client import QdrantClient

from ..compose import Project
from ..utils import is_windows, save_data_path, waitUntil

logger = logging.getLogger(__name__)

repo_dir = Path(__file__).resolve().parent.parent.parent
template_dir = repo_dir / 'learn2rag' / 'ui' / 'templates' / 'compose' / 'pipelines'


def docker_compose_available() -> bool:
    try:
        subprocess.run(['docker', 'compose', '--version'], check=True, capture_output=True)
        subprocess.run(['docker', 'ps'], check=True, capture_output=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


requires_docker_compose = pytest.mark.skipif(not docker_compose_available(), reason='no usable `docker compose`')


def docker_compose(compose_file: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run([
        'docker',
        'compose',
        '--file', str(compose_file),
        '--progress', 'quiet',
        *args
    ], check=True)


def url_ready(url: str) -> bool:
    """True once the server answers at all; errors below 500 (e.g. login required) count as up."""
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return bool(response.status < 500)
    except urllib.error.HTTPError as e:
        return e.code < 500
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        # Not listening yet, or reset while the container is still starting
        return False


@contextlib.contextmanager
def docker_service(compose_file: Path, url: str, *, timeout: int = 180) -> Iterator[str]:
    """Start a fresh instance from `compose_file`, wait until `url` answers, and remove it afterwards."""
    logger.debug('Starting %s', compose_file)
    docker_compose(compose_file, 'up', '-d', '--force-recreate')
    try:
        waitUntil(lambda: url_ready(url), timeout=timeout * 1000)
        logger.debug('Started %s', compose_file)
        yield url
    finally:
        # `down` also removes the network, so the next run never reuses a stale one
        docker_compose(compose_file, 'down', '--remove-orphans')


class RagEnvironment:
    """Runs the import and pipeline compose templates against a scratch storage path."""

    def __init__(self, project_name: str = 'test', rag_port: int = 5002) -> None:
        self.project_name = project_name
        self.rag_port = rag_port
        self.storage_path = Path(save_data_path('Learn2RAG', 'tests'))
        self.openai_client = OpenAI(
            api_key='',
            base_url=f'http://localhost:{self.rag_port}/api/v1',
            max_retries=0,
        )

    def setup(self) -> None:
        self.remove_project()
        shutil.rmtree(self.storage_path, ignore_errors=True)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def teardown(self) -> None:
        self.remove_project()
        shutil.rmtree(self.storage_path, ignore_errors=True)

    def remove_project(self) -> None:
        if project := Project.get(self.project_name):
            if project.running:
                project.stop()
            project.remove()

    def template_context(self, loaders: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            'is_windows': is_windows(),
            'learn2rag_path': repo_dir,
            'config': {},
            'storage_path': self.storage_path,
            'ports': {
                'pipeline': self.rag_port,
            },
            'qdrant_api_key': '',
            'language_model': {'api': 'ChatFake'},
            'pipeline': {
                'qdrant_path': self.storage_path / 'qdrant_persistence',
            },
            'import_config': {
                'loaders': loaders,
            },
        }

    def start(self, template: str, loaders: list[dict[str, Any]]) -> Project:
        project = Project.create(template_dir / template, self.project_name, template=True, template_context=self.template_context(loaders))
        assert project is not None, 'project should not be None'
        project.start()
        assert project.running
        return project

    def run_import(self, loaders: list[dict[str, Any]], *, timeout: int = 600) -> Project:
        """Run the import to completion and return the finished project (already removed)."""
        logger.info('Starting import')
        self.start('import.yml', loaders)

        finished: list[Project] = []

        def check_import() -> None:
            project = Project.get(self.project_name)
            assert project is not None, 'project exists'
            assert not project.running, 'import completed'
            finished.append(project)
        waitUntil(check_import, timeout=timeout * 1000)

        project = finished[-1]
        project.remove()
        logger.info('Finished import')
        return project

    def imported_points(self) -> list[dict[str, Any]]:
        """Payloads of everything stored in Qdrant; empty if nothing was imported."""
        path = self.storage_path / 'qdrant_persistence'
        if not path.exists():
            return []
        qdrant = QdrantClient(path=str(path))
        try:
            if not qdrant.collection_exists('learn2rag'):
                return []
            points, _ = qdrant.scroll('learn2rag', limit=10000, with_payload=True)
        finally:
            qdrant.close()
        return [point.payload or {} for point in points]

    def imported_text(self) -> str:
        return '\n'.join(str(point.get('content', '')) for point in self.imported_points())

    def imported_sources(self) -> set[str]:
        return {str(point.get('source', '')) for point in self.imported_points()}

    def ask(self, question: str, *, check: Callable[[str], None] | None = None, timeout: int = 120) -> str:
        """Ask the running pipeline; with ChatFake the answer echoes the retrieved chunks.

        Retries until the pipeline answers and `check(answer)` (if given) stops raising AssertionError.
        """
        answers: list[str] = []

        def answered() -> None:
            try:
                completion = self.openai_client.chat.completions.create(
                    model='learn2rag',
                    messages=[{'role': 'user', 'content': question}],
                )
            except APIConnectionError:
                assert False, 'pipeline API is available'
            content = completion.choices[-1].message.content or ''
            logger.debug('Response content: %s', content)
            assert 'for testing only' in content, 'contains test marker'
            assert 'Information:\n' in content, 'contains the prompt'
            if check is not None:
                check(content)
            answers.append(content)
        waitUntil(answered, timeout=timeout * 1000)
        return answers[-1]
