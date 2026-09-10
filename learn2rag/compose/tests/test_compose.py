import pathlib
import time
import unittest
import sys

from .. import Project


data_dir = pathlib.Path(__file__).parent.resolve() / 'data'
is_windows = sys.platform == 'win32'

def get_yaml_path(base_name: str) -> pathlib.Path:
    filename = f"{base_name}_win.yml" if is_windows else f"{base_name}.yml"
    return data_dir / filename

class ComposeTestCase(unittest.TestCase):
    project: Project | None

    def setUp(self) -> None:
        self.project = None

    def tearDown(self) -> None:
        if self.project is not None:
            self.project.remove()

    def test_zero_exitcode(self) -> None:
        name = 'test_zero_exitcode'
        self.project = Project.create(get_yaml_path('zero_exitcode'), name)
        assert self.project is not None
        self.project.start()
        assert self.project.running
        for _ in range(20):
            self.project = Project.get(name)
            assert self.project is not None, "Project disappeared unexpectedly"
            if not self.project.running:
                break
            time.sleep(0.1)
        self.project = Project.get(name)
        assert self.project is not None
        assert not self.project.running
        assert self.project.succeeded
        assert not self.project.failed

    def test_nonzero_exitcode(self) -> None:
        name = 'test_nonzero_exitcode'
        self.project = Project.create(get_yaml_path('nonzero_exitcode'), name)
        assert self.project is not None
        self.project.start()
        assert self.project.running
        for _ in range(20):
            self.project = Project.get(name)
            assert self.project is not None, "Project disappeared unexpectedly"
            if not self.project.running:
                break
            time.sleep(0.1)
        self.project = Project.get(name)
        assert self.project is not None
        assert not self.project.running
        assert not self.project.succeeded
        assert self.project.failed

    def test_stop(self) -> None:
        name = 'stop'
        self.project = Project.create(get_yaml_path('sleep'), name)
        assert self.project is not None
        self.project.start()
        assert self.project.running
        for _ in range(20):
            self.project = Project.get(name)
            assert self.project is not None, "Project disappeared unexpectedly"
            if not self.project.running:
                break
            time.sleep(0.1)
        self.project = Project.get(name)
        assert self.project is not None
        assert self.project.running
        self.project.stop()
        assert not self.project.running
        assert not self.project.succeeded
        assert not self.project.failed
