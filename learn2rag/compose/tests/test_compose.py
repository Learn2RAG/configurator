import pathlib
import time
import unittest

from .. import Project


data_dir = pathlib.Path(__file__).parent.resolve() / 'data'


class ComposeTestCase(unittest.TestCase):
    def setUp(self):
        self.project = None

    def tearDown(self):
        if self.project is not None:
            self.project.remove()

    def test_zero_exitcode(self):
        name = 'test_zero_exitcode'
        self.project = Project.create(data_dir / 'zero_exitcode.yml', name)
        assert self.project is not None
        self.project.start()
        assert self.project.running
        time.sleep(0.1)
        self.project = Project.get(name)
        assert self.project is not None
        assert not self.project.running
        assert self.project.succeeded
        assert not self.project.failed

    def test_nonzero_exitcode(self):
        name = 'test_nonzero_exitcode'
        self.project = Project.create(data_dir / 'nonzero_exitcode.yml', name)
        assert self.project is not None
        self.project.start()
        assert self.project.running
        time.sleep(0.1)
        self.project = Project.get(name)
        assert self.project is not None
        assert not self.project.running
        assert not self.project.succeeded
        assert self.project.failed
