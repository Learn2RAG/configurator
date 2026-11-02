import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from .. import download_models


class DownloadModelsTestCase(unittest.TestCase):
    @patch.object(download_models, 'snapshot_download')
    def test_downloads_the_default_embedding_model(self, snapshot_download: MagicMock) -> None:
        with patch.dict(os.environ, {}, clear=True):
            download_models.main()

        snapshot_download.assert_called_once_with(repo_id='BAAI/bge-m3', ignore_patterns=download_models.IGNORE_PATTERNS)

    @patch.object(download_models, 'snapshot_download')
    def test_downloads_the_configured_embedding_model(self, snapshot_download: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'opt_config.json')
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({'embedding_model': 'sentence-transformers/all-mpnet-base-v2'}, f)
            with patch.dict(os.environ, {'PIPELINE_OPT_CONFIG': path}):
                download_models.main()

        snapshot_download.assert_called_once_with(
            repo_id='sentence-transformers/all-mpnet-base-v2',
            ignore_patterns=download_models.IGNORE_PATTERNS,
        )
