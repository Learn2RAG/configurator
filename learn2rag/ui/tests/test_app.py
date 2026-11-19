import logging
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Generator
from unittest import TestCase

import pytest
import yaml
from flask import Flask
from flask.testing import FlaskClient
from pydantic import ValidationError

from .. import create_app
from ...pipeline import llm

logger = logging.getLogger(__name__)

@pytest.fixture
def app() -> Generator[Flask, None, None]:
    logger.debug('Creating app')
    instance_path = tempfile.mkdtemp()
    app = create_app({
        'flask': {
            'instance_path': instance_path,
        },
    })
    app.testing = True
    with app.app_context():
        pass
    yield app
    shutil.rmtree(instance_path)


@pytest.fixture
def client(app: Flask) -> Generator[FlaskClient, None, None]:
    logger.debug('Creating test client')
    yield app.test_client()


class TestData():
    def test_models(self, app: Flask, client: FlaskClient) -> None:
        res = client.get('/models')
        assert res.status_code == 200

    def test_create_remote(self, app: Flask, client: FlaskClient) -> None:
        client.post('/models', data={
            'api': llm.OpenAIClient.ID,
            'label': 'Test label',
            'model': 'test_model',
            'token': 'test_token',
            'url': 'http://пример.испытание/',
        })
        files = list((Path(app.instance_path) / 'data' / 'models').glob('*.yml'))
        assert len(files) == 1
        file = files[0]
        TestCase().assertDictEqual(yaml.safe_load(file.open()), {
            'api': llm.OpenAIClient.ID,
            'label': 'Test label',
            'model': 'test_model',
            'token': 'test_token',
            'url': 'http://пример.испытание/',
        })

    def test_create_local(self, app: Flask, client: FlaskClient) -> None:
        client.post('/models', data={
            'api': llm.OllamaClient.ID,
            'label': 'Test label',
            'model': 'test_model',
        })
        files = list((Path(app.instance_path) / 'data' / 'models').glob('*.yml'))
        assert len(files) == 1
        file = files[0]
        TestCase().assertDictEqual(yaml.safe_load(file.open()), {
            'api': llm.OllamaClient.ID,
            'label': 'Test label',
            'model': 'test_model',
            'token': '',
            'url': 'http://127.0.0.1:11434/',
        })

    def test_unsupported_api(self, app: Flask, client: FlaskClient) -> None:
        with suppress(ValidationError):
            client.post('/models', data={
                'api': 'unsupported',
                'label': 'Test label',
                'model': 'test_model',
                'token': 'test_token',
                'url': 'http://пример.испытание/',
            })
        files = list((Path(app.instance_path) / 'data' / 'models').glob('*.yml'))
        assert len(files) == 0
