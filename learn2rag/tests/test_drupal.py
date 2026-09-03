import logging
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Generator

from ..compose import Project
from ..utils import is_windows, save_data_path, waitUntil

import pytest
from openai import APIConnectionError, OpenAI
from _pytest.logging import LogCaptureFixture

logger = logging.getLogger(__name__)

template_dir = Path(__file__).resolve().parent.parent / 'ui' / 'templates' / 'compose' / 'pipelines'
data_dir = Path(__file__).resolve().parent / 'data'


def docker_compose_available() -> bool:
    try:
        subprocess.run(['docker', 'compose', '--version'], check=True)
        subprocess.run(['docker', 'ps'], check=True)
        return True
    except:
        return False


def docker_compose(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(['docker', 'compose', '--file', 'drupal/docker-compose.yml', *args], check=True)


@pytest.fixture(scope='module')
def drupal_instance() -> Generator[str, None, None]:
    logging.info('Starting Drupal')
    docker_compose('up', '-d', '--force-recreate')
    base_url = 'http://localhost:3470'
    def drupal_ready() -> bool:
        try:
            return bool(urllib.request.urlopen(base_url).getcode() == 200)
        except ConnectionResetError:
            return False
    waitUntil(drupal_ready, timeout=1 * 180 * 1000)
    logging.info('Started Drupal')
    yield base_url
    docker_compose('rm',  '--stop', '--force')


@pytest.mark.skipif(not docker_compose_available(), reason='no usable `docker compose`')
class TestDrupal():
    openai_client: Any
    project_name: str
    rag_port: int
    storage_path: Path

    @pytest.fixture(autouse=True)
    def use_caplog(self, caplog: LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING, logger='httpcore')
        caplog.set_level(logging.WARNING, logger='httpx')
        caplog.set_level(logging.WARNING, logger='openai')

    @pytest.fixture(autouse=True)
    def setup(self) -> Generator[None, None, None]:
        self.project_name = 'test'
        self.rag_port = 5002
        self.storage_path = Path(save_data_path('Learn2RAG', 'tests'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.openai_client = OpenAI(
            api_key='',
            base_url=f'http://localhost:{self.rag_port}/api/v1',
            max_retries=0,
        )
        if project := Project.get(self.project_name):
            if project.running:
                project.stop()
            project.remove()
        yield
        if self.storage_path is not None:
            shutil.rmtree(self.storage_path, ignore_errors=True)
        if project := Project.get(self.project_name):
            if project.running:
                project.stop()
            project.remove()

    def test_drupal_import(self, drupal_instance: str) -> None:
        template_context = {
            'is_windows': is_windows(),
            'learn2rag_path': Path('.').absolute(),
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
                'loaders': [
                    {
                        'loader_id': 'drupal_test',
                        'loader_type': 'DrupalLoader',
                        'auth_type': 'none',
                        'base_url': drupal_instance,
                        'content_types': ['article'],  # articles are the least amount in demo_umami of all types
                    },
                ],
            },
        }

        logging.info('Starting import')
        project = Project.create(template_dir / 'import.yml', self.project_name, template=True, template_context=template_context)
        assert project is not None, 'project should not be None'
        project.start()
        assert project.running

        def check_import() -> None:
            project = Project.get(self.project_name)
            assert project is not None
            assert not project.running
        waitUntil(check_import, timeout=1 * 180 * 1000)

        project.remove()
        logging.info('Finished import')

        logging.info('Starting pipeline')
        project = Project.create(template_dir / 'pipeline.yml', self.project_name, template=True, template_context=template_context)
        assert project is not None, 'project should not be None'
        project.start()
        assert project.running

        def check_rag() -> None:
            try:
                completion = self.openai_client.chat.completions.create(
                    model='learn2rag',
                    messages=[
                        {'role': 'user', 'content': f'Which colors can be carrots?'},
                    ],
                )
                content = completion.choices[-1].message.content
                logger.debug('Response content: %s', content)
                assert 'for testing only' in content, 'contains test marker'
                assert "Information:\\n" in content, 'contains the prompt'
                assert not content.endswith("Information:\\n"), 'contains any document chunks in the prompt'
                assert 'purple, red, yellow or white' in content, 'specific text from a test file'
            except APIConnectionError:
                assert False
        waitUntil(check_rag, timeout=1 * 60 * 1000)
        logging.info("Finished pipeline")
