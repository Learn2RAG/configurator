import unittest
from getpass import getuser
from pathlib import Path

from .. import (
    is_windows,
    normalize_path,
    save_data_path,
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
