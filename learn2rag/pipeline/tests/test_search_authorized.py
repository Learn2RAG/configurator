import sys
import inspect
import unittest
from typing import Any

from qdrant_client.http.models import ScoredPoint

# Import the target function
from ..search import search_authorized


class SearchAuthorizedTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # 1. Verify the search.py file was actually updated!
        source = inspect.getsource(search_authorized)
        if "max_auth_retries" not in source:
            self.fail("CRITICAL ERROR: pipeline/search.py does not contain the updated retry loop.")

        self.user_auths = {"roles": ["user", "admin"]}
        self.user_config = {"collection_name": "test_collection"}

        self.opt_config: dict[str, Any] = {
            "top_k": 3,
            "max_auth_retries": 3,
            "auth_oversample_start": 2,
            "auth_oversample_step": 2,
            "top_k_reranker": 3,
            "prefetch_limit_dense": 10,
        }

        # 2. BRUTE FORCE MOCKING: Find ALL instances of the search module in memory
        self.search_modules = [
            mod for name, mod in sys.modules.items()
            if name.endswith('search') and hasattr(mod, '_collect_query_points')
        ]

        # Store original functions to restore them cleanly after tests
        self.originals = {
            mod: (mod._collect_query_points, getattr(mod, 'filter_authorized', None))
            for mod in self.search_modules
        }

        self.collect_calls = []
        self.filter_call_count = 0

    def tearDown(self) -> None:
        # Restore all original functions to memory
        for mod, (orig_collect, orig_filter) in self.originals.items():
            mod._collect_query_points = orig_collect
            if orig_filter:
                mod.filter_authorized = orig_filter

        search_authorized.__globals__['_collect_query_points'] = self.originals[self.search_modules[0]][0]
        search_authorized.__globals__['filter_authorized'] = self.originals[self.search_modules[0]][1]

    def _apply_patches(self, fake_collect, fake_filter):
        """Inject our fakes into every possible memory space where the code might execute"""
        for mod in self.search_modules:
            mod._collect_query_points = fake_collect
            mod.filter_authorized = fake_filter

        # Also patch the direct function globals as a fallback
        search_authorized.__globals__['_collect_query_points'] = fake_collect
        search_authorized.__globals__['filter_authorized'] = fake_filter

    def _make_mock_points(self, n: int) -> list[ScoredPoint]:
        return [
            ScoredPoint(
                id=i + 1,
                score=1.0 - (i * 0.01),
                version=1,
                payload={"content": f"mock doc {i}"}
            ) for i in range(n)
        ]

    def _extract_opt(self, args, kwargs) -> dict:
        if len(args) >= 3:
            return args[2]
        return kwargs.get('opt_config', kwargs.get('local_opt', {}))

    async def test_success_on_first_attempt(self) -> None:
        def fake_collect(*args, **kwargs):
            self.collect_calls.append(self._extract_opt(args, kwargs))
            return self._make_mock_points(6)

        async def fake_filter(*args, **kwargs):
            self.filter_call_count += 1
            return self._make_mock_points(4)

        self._apply_patches(fake_collect, fake_filter)

        results = await search_authorized(
            "test query", self.user_auths, user_config=self.user_config, opt_config=self.opt_config
        )

        self.assertEqual(len(self.collect_calls), 1, "Collect was not called exactly once.")
        self.assertEqual(len(results), 3)
        self.assertEqual(self.collect_calls[0]["top_k"], 6)
        self.assertEqual(self.collect_calls[0]["top_k_reranker"], 6)

    async def test_success_on_second_attempt_after_scaling(self) -> None:
        def fake_collect(*args, **kwargs):
            self.collect_calls.append(self._extract_opt(args, kwargs))
            return self._make_mock_points(12)

        async def fake_filter(*args, **kwargs):
            self.filter_call_count += 1
            if self.filter_call_count == 1:
                return self._make_mock_points(1)  # Fail first try
            return self._make_mock_points(5)  # Succeed second try

        self._apply_patches(fake_collect, fake_filter)

        results = await search_authorized(
            "test query", self.user_auths, user_config=self.user_config, opt_config=self.opt_config
        )

        self.assertEqual(len(self.collect_calls), 2, "It should have retried twice")
        self.assertEqual(len(results), 3)
        self.assertEqual(self.collect_calls[1]["top_k"], 12)

    async def test_max_retries_exhausted(self) -> None:
        def fake_collect(*args, **kwargs):
            self.collect_calls.append(self._extract_opt(args, kwargs))
            return self._make_mock_points(12)

        async def fake_filter(*args, **kwargs):
            return self._make_mock_points(1)  # Always drop all but 1

        self._apply_patches(fake_collect, fake_filter)
        self.opt_config["max_auth_retries"] = 2

        results = await search_authorized(
            "test query", self.user_auths, user_config=self.user_config, opt_config=self.opt_config
        )

        self.assertEqual(len(self.collect_calls), 2, "Should exhaust exactly 2 max retries")
        self.assertEqual(len(results), 1, "Should return what it managed to find")

    async def test_maintains_original_opt_config_immutability(self) -> None:
        def fake_collect(*args, **kwargs):
            self.collect_calls.append(self._extract_opt(args, kwargs))
            return self._make_mock_points(12)

        async def fake_filter(*args, **kwargs):
            self.filter_call_count += 1
            if self.filter_call_count == 1:
                return self._make_mock_points(1)
            return self._make_mock_points(4)

        self._apply_patches(fake_collect, fake_filter)

        await search_authorized(
            "test query", self.user_auths, user_config=self.user_config, opt_config=self.opt_config
        )

        self.assertEqual(self.opt_config["top_k"], 3, "Original dictionary should not have been mutated")