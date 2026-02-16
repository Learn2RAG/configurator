import unittest
import socket

from .. import find_free_ports, merge

class UIMiscTestCase(unittest.TestCase):
    def test_merge(self):
        assert merge({'s1': 11, 'c': {'s2': 22, 'd3': 33}}, {'d1': 1, 'c': {'d2': 2, 'd3': 3}}) == {'d1': 1, 's1': 11, 'c': {'d2': 2, 'd3': 33, 's2': 22}}

    def test_find_free_ports_count(self):
        self.assertEqual(len(find_free_ports(1)), 1)
        self.assertEqual(len(find_free_ports(4)), 4)

    def test_find_free_ports_preferred(self):
        # Using specific ports that are likely free
        # if this failed check if ports are free
        preferred = [9990, 9991]
        ports = find_free_ports(2, preferred_ports=preferred)
        self.assertEqual(ports, preferred)

    def test_find_free_ports_collision(self):
        # Create a collision to test fallback
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.bind(('', 0))
        taken_port = blocker.getsockname()[1]

        try:
            # Ask for 1 port, but the preferred one is already blocked by 'blocker'
            ports = find_free_ports(1, preferred_ports=[taken_port])
            self.assertEqual(len(ports), 1)
            self.assertNotEqual(ports[0], taken_port, "Should have avoided the busy port")
        finally:
            blocker.close()
