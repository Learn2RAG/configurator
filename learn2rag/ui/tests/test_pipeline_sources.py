import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import learn2rag.data
from .. import create_app


class PipelineSourceDeleteTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.client = create_app({'flask': {'instance_path': self.root}}).test_client()
        self.source_id = learn2rag.data.create_entry(self.root, 'sources', {'label': 'Source', 'type': 'local', 'path': '/tmp'})
        self.other_id = learn2rag.data.create_entry(self.root, 'sources', {'label': 'Other', 'type': 'local', 'path': '/tmp'})
        self.pipeline_id = learn2rag.data.create_entry(self.root, 'pipelines', {
            'label': 'Pipeline',
            'storage_path': str(Path(self.root) / 'pipeline'),
            'sources': [self.source_id, self.other_id],
            'ports': [],
        })

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def sources(self) -> list[str]:
        sources: list[str] = learn2rag.data.get_entry(self.root, 'pipelines', self.pipeline_id)['sources']
        return sources

    @patch('learn2rag.ui.Project.get', return_value=None)
    def test_removes_source(self, _: MagicMock) -> None:
        response = self.client.delete(f'/pipelines/{self.pipeline_id}/sources/{self.source_id}')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.sources(), [self.other_id])

    @patch('learn2rag.ui.Project.get', return_value=MagicMock(running=True))
    def test_running_pipeline_is_unchanged(self, _: MagicMock) -> None:
        response = self.client.delete(f'/pipelines/{self.pipeline_id}/sources/{self.source_id}')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.sources(), [self.source_id, self.other_id])

    def test_unknown_pipeline(self) -> None:
        response = self.client.delete(f'/pipelines/missing/sources/{self.source_id}')
        self.assertEqual(response.status_code, 302)

    @patch('learn2rag.ui.Project.get', return_value=None)
    def test_last_source_is_kept(self, _: MagicMock) -> None:
        self.client.delete(f'/pipelines/{self.pipeline_id}/sources/{self.source_id}')
        response = self.client.delete(f'/pipelines/{self.pipeline_id}/sources/{self.other_id}')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.sources(), [self.other_id])
