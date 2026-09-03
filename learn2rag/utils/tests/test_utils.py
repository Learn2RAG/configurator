import os
import unittest
from getpass import getuser
from pathlib import Path
from unittest.mock import patch
from typing import Any

from .. import (
    is_windows,
    normalize_path,
    save_data_path,
    get_default_rag_dir
)


class UtilsTestCase(unittest.TestCase):
    @unittest.skipIf(is_windows(), 'This test is not adapted for Windows')
    def test_normalize_path(self) -> None:
        username = getuser()
        assert str(normalize_path(Path('~' + username))).startswith('/')
        assert str(normalize_path(Path('.'))).startswith('/')
        with self.assertRaises(ValueError):
            str(normalize_path(Path('..'))).index('..')

    def test_save_data_path(self) -> None:
        path = Path(save_data_path('Learn2RAG', 'tests'))
        assert path.exists()
        assert path.is_dir()
        (path / 'writeable').touch()

    @patch('learn2rag.utils.is_windows', return_value=True)
    @patch('os.getenv', return_value=r'C:\Users\TestUser')
    def test_get_default_rag_dir_windows(self, mock_getenv: Any, mock_is_windows: Any) -> None:
        result = get_default_rag_dir()
        expected = os.path.join(r'C:\Users\TestUser', 'Documents', 'RAG')
        self.assertEqual(result, expected)

    @patch('learn2rag.utils.is_windows', return_value=False)
    @patch('pathlib.Path.home', return_value=Path('/home/testuser'))
    def test_get_default_rag_dir_linux_fallback(self, mock_home: Any, mock_is_windows: Any) -> None:
        # Hide the xdg module to simulate an environment where it's missing, forcing the fallback path
        with patch.dict('sys.modules', {'xdg.userdirs': None}):
            result = get_default_rag_dir()

        expected = os.path.join('/home/testuser', 'Documents', 'RAG')
        # Use replace to handle potential slash differences if tests are executed on Windows
        self.assertEqual(result.replace('\\', '/'), expected.replace('\\', '/'))