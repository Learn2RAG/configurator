import logging
import shutil
from pathlib import Path
from unittest import TestCase
from typing import Any

from ..compose import Project
from ..utils import is_windows, save_data_path, waitUntil

import pytest
from openai import APIConnectionError, OpenAI
from _pytest.logging import LogCaptureFixture

# for test the resource delete

from unittest.mock import patch
from learn2rag.ui import create_app
import learn2rag.data

logger = logging.getLogger(__name__)

template_dir = Path(__file__).resolve().parent.parent / 'ui' / 'templates' / 'compose' / 'pipelines'
# optimization_dir = Path(__file__).resolve().parent.parent / 'optimization'
data_dir = Path(__file__).resolve().parent / 'data'


class Learn2RAGTestCase(TestCase):
    openai_client: Any
    project_name: str
    rag_port: int
    storage_path: Path

    @pytest.fixture(autouse=True)
    def use_caplog(self, caplog: LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING, logger='httpcore')
        caplog.set_level(logging.WARNING, logger='httpx')
        caplog.set_level(logging.WARNING, logger='openai')

    def setUp(self) -> None:
        self.project_name = 'test'
        self.rag_port = 5002
        self.storage_path = Path(save_data_path('Learn2RAG', 'tests'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.openai_client = OpenAI(
            api_key='',
            base_url=f'http://localhost:{self.rag_port}',
            max_retries=0,
        )
        if project := Project.get(self.project_name):
            if project.running:
                project.stop()
            project.remove()

    def tearDown(self) -> None:
        if self.storage_path is not None:
            shutil.rmtree(self.storage_path, ignore_errors=True)
        if project := Project.get(self.project_name):
            if project.running:
                project.stop()
            project.remove()

    def test_learn2rag(self) -> None:
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
                        'loader_id': 'local_test',
                        'loader_type': 'DirectoryLoader',
                        'recursive': 'True',
                        'path': str(data_dir),
                    },
                ],
            },
        }

        project = Project.create(template_dir / 'import.yml', self.project_name, template=True, template_context=template_context)
        assert project is not None, 'project should not be None'
        project.start()
        assert project.running

        def check_import() -> None:
            project = Project.get(self.project_name)
            assert project is not None
            assert not project.running
        waitUntil(check_import, timeout=1 * 60 * 1000)

        project.remove()

        project = Project.create(template_dir / 'pipeline.yml', self.project_name, template=True, template_context=template_context)
        assert project is not None, 'project should not be None'
        project.start()
        assert project.running

        def check_rag() -> None:
            try:
                completion = self.openai_client.chat.completions.create(
                    model='learn2rag',
                    messages=[
                        {'role': 'user', 'content': f'What are rabbits?'},
                    ],
                )
                content = completion.choices[-1].message.content
                logger.debug('Response content: %s', content)
                assert 'for testing only' in content, 'contains test marker'
                assert "Information:\\n" in content, 'contains the prompt'
                assert not content.endswith("Information:\\n"), 'contains any document chunks in the prompt'
                assert 'Lagomorpha' in content, 'specific text from a test file'
            except APIConnectionError:
                assert False
        waitUntil(check_rag, timeout=1 * 60 * 1000)

    def test_optimization(self) -> None:
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
                        'loader_id': 'local_test',
                        'loader_type': 'DirectoryLoader',
                        'recursive': 'True',
                        'path': str(data_dir),
                    },
                ],
            },
        }

        project = Project.create(template_dir / 'import.yml', self.project_name, template=True,
                                 template_context=template_context)
        assert project is not None, 'project should not be None'
        project.start()
        assert project.running

        def check_import() -> None:
            project = Project.get(self.project_name)
            assert project is not None
            assert not project.running

        waitUntil(check_import, timeout=1 * 60 * 1000)

        project.remove()

        project = Project.create(template_dir / 'pipeline.yml', self.project_name, template=True,
                                 template_context=template_context)

        assert project is not None
        project.start()

        def check_pipeline() -> None:
            try:
                self.openai_client.models.list()
            except APIConnectionError:
                assert False

        waitUntil(check_pipeline, timeout=1 * 60 * 1000)

        # Optimization
        from learn2rag.optimization.baseline_optimization import run
        import os
        import json
        dataset_name = "test_rabbit"
        opt_out_dir = self.storage_path / "opt_output"
        results_file = opt_out_dir / dataset_name / "optimization_results.json"

        mock_user_config = {
            "collection_name": dataset_name,
            "qdrant_path": str(self.storage_path / 'qdrant_persistence')
        }
        os.environ["PIPELINE_USER_CONFIG"] = json.dumps(mock_user_config)

        initial_mtime = results_file.stat().st_mtime if results_file.exists() else 0.0


        mock_registry = {
            "datasets": {
                dataset_name: {
                    "subdirectory": "", "split": "train",
                    "fields": {"q": "question", "a": "answer", "id": "id"},
                    "path": str(data_dir / "rabbit_eval.csv")
                }
            },
            "prompts": {
                "default": "Answer using ONLY the provided information: {context}",
                "concise": "Be concise. Information: {context}"
            }
        }

        best_cfg, history, importance = run(
            dataset_name=dataset_name ,
            max_questions=2,
            n_trials=2,
            output_dir=opt_out_dir,
            registry_path=mock_registry
        )


        assert best_cfg is not None, "Optimization should return a valid configuration"
        assert len(history) == 2, "History length should match n_trials"


        assert results_file.exists(), "Optimization output JSON was not created"
        current_mtime = results_file.stat().st_mtime
        assert current_mtime > initial_mtime, "The optimization results file was not updated during the run!"

        with open(results_file, 'r') as f:
            results_data = json.load(f)
            assert "best_config" in results_data
            assert "top_k" in results_data["best_config"], "Optimization failed to output expected parameters"

    def test_pipeline_source_delete(self) -> None:
        # Setup Flask App for testing
        app_config = {'flask': {'instance_path': str(self.storage_path)}}
        app = create_app(config=app_config)
        client = app.test_client()

        # Seed required database entries
        model_id = learn2rag.data.create_entry(
            str(self.storage_path), 'models',
            {'label': 'dummy', 'url': 'http://localhost', 'api': 'ChatFake'}
        )
        source_id = learn2rag.data.create_entry(
            str(self.storage_path), 'sources',
            {'label': 'Test Source', 'type': 'local', 'path': '/tmp'}
        )
        pipeline_id = learn2rag.data.create_entry(
            str(self.storage_path), 'pipelines',
            {
                'label': 'Test Pipeline',
                'storage_path': str(self.storage_path / 'test_pipeline'),
                'sources': [source_id],
                'language_model': model_id,
                'ports': []
            }
        )

        #  Patch start_project
        with patch('learn2rag.ui.start_project') as mock_start_project:
            #  Execute the exact DELETE route specified in the ticket
            response = client.delete(f'/pipelines/{pipeline_id}/sources/{source_id}')

            assert response.status_code == 302

            # Was the source removed from the pipeline config in the DB?
            updated_pipeline = learn2rag.data.get_entry(str(self.storage_path), 'pipelines', pipeline_id)
            assert source_id not in updated_pipeline['sources']

            # Was the cleanup job triggered?
            mock_start_project.assert_called_once()
            args, kwargs = mock_start_project.call_args

            # Extract the render_context passed to the template
            render_context = kwargs.get('render_context', args[3] if len(args) > 3 else {})
            import_config = render_context.get('import_config', {})

            # Was the IMPORTER_DELETE_LOADER_ID environment variable injected?
            assert 'environment' in import_config, "Environment variable block missing from import config"
            assert import_config['environment'].get(
                'IMPORTER_DELETE_LOADER_ID') == source_id, "Deletion ID not injected"