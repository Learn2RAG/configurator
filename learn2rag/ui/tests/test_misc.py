import unittest

from .. import find_free_ports, merge

class UIMiscTestCase(unittest.TestCase):
    def test_merge(self):
        assert merge({'s1': 11, 'c': {'s2': 22, 'd3': 33}}, {'d1': 1, 'c': {'d2': 2, 'd3': 3}}) == {'d1': 1, 's1': 11, 'c': {'d2': 2, 'd3': 33, 's2': 22}}

    def test_find_free_ports(self):
        assert len(find_free_ports(1)) == 1
        assert len(find_free_ports(4)) == 4

    # TODO: actual tests
