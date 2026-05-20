import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from collections import defaultdict
from typing import Any, ClassVar, DefaultDict, Dict, List, Optional, Set
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document
from ..loaders.directory_loader import load_from_directory
from ..loaders.html_loader import load_html_content, _is_same_site
from ..loaders.jira_loader import load_from_jira, get_all_jira_document_ids

# Set RUN_INTEGRATION_TESTS=1 to run tests that require network access.
_RUN_INTEGRATION: bool = os.environ.get("RUN_INTEGRATION_TESTS", "0") == "1"

class ImporterLoadersTestCase(unittest.TestCase):
    """Tests for directory and HTML loaders. Runs fully offline without Qdrant."""

    test_path: ClassVar[str]
    _temp_dir: ClassVar[Optional[str]]

    @classmethod
    def setUpClass(cls) -> None:
        env_path = os.environ.get("TEST_IMPORT_PATH", "")
        if env_path and pathlib.Path(env_path).is_dir():
            cls.test_path = env_path
            cls._temp_dir = None
        else:
            cls._temp_dir = tempfile.mkdtemp(prefix="learn2rag_test_")
            cls.test_path = cls._temp_dir
            (pathlib.Path(cls._temp_dir) / "sample.txt").write_text(
                "This is a test document.\nIt has multiple lines of content.\nLine three.",
                encoding="utf-8",
            )

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._temp_dir:
            shutil.rmtree(cls._temp_dir, ignore_errors=True)

    @unittest.skipUnless(_RUN_INTEGRATION, "Set RUN_INTEGRATION_TESTS=1 to run")
    def test_remote_url(self) -> None:
        docs = load_html_content('https://learn2rag.de')
        assert len(docs) == 1
        doc, = docs
        assert 'source' in doc.metadata
        assert 'wird das Projekt von einem Konsortium führender wissenschaftlicher Institutionen und Softwareentwickler durchgeführt und durch weitreichende Unternehmensnetzwerke unterstützt' in doc.page_content

    def test_import_directory(self) -> None:
        """Loads files from test_path, prints metadata incl. content_hash, and
        verifies that each source yields exactly one Document with a stable hash.

        Intentionally Qdrant-free: only the loader and hash consistency are tested.
        Set env var SKIP_HASH_ASSERT=1 to print-only without assertions (debugging).
        """
        skip_assert = os.environ.get("SKIP_HASH_ASSERT", "0") == "1"

        docs: List[Document] = load_from_directory(
            self.test_path, recursive=False, loader_id="test_import"
        )

        def _safe(text: str, limit: int = 500) -> str:
            """Truncate and replace unencodable characters for safe terminal output."""
            encoding = sys.stdout.encoding or "utf-8"
            return text[:limit].encode(encoding, errors="replace").decode(encoding)

        print(f"\n=== {len(docs)} document(s) loaded from '{self.test_path}' ===")

        by_source: DefaultDict[str, List[Document]] = defaultdict(list)
        for doc in docs:
            by_source[doc.metadata.get("source", "?")].append(doc)

        for i, doc in enumerate(docs, start=1):
            print(f"\n--- Document {i} ---")
            print(f"Metadata: {json.dumps(doc.metadata, indent=2, default=str)}")
            print(f"page_content (first 500 chars):\n{_safe(doc.page_content)}")

        print("\n=== Hash consistency per source ===")
        for source, source_docs in sorted(by_source.items()):
            hashes: Set[Optional[str]] = {
                d.metadata.get("content_hash", "MISSING") for d in source_docs
            }
            status = "OK" if len(hashes) == 1 else "MISMATCH"
            print(f"[{status}] {source}  ({len(source_docs)} doc(s))  hashes={hashes}")

        self.assertGreater(len(docs), 0, f"No documents found in: {self.test_path}")

        if not skip_assert:
            for source, source_docs in by_source.items():
                hashes2: Set[Optional[str]] = {d.metadata.get("content_hash") for d in source_docs}
                self.assertEqual(
                    len(hashes2), 1, f"Hash mismatch for '{source}': {hashes2}"
                )
                self.assertEqual(
                    len(source_docs),
                    1,
                    f"Expected 1 Document per source, got {len(source_docs)} for '{source}'",
                )
        else:
            print("\n[SKIP_HASH_ASSERT=1] Hash assertion skipped.")


@unittest.skipUnless(_RUN_INTEGRATION, "Set RUN_INTEGRATION_TESTS=1 to run")
class HtmlLoaderLearn2RagFullCrawlTestCase(unittest.TestCase):
    """Integration test: full site crawl of https://learn2rag.de with depth=-1."""

    def test_full_site_crawl(self) -> None:
        """Crawls the entire learn2rag.de domain and prints all discovered pages."""
        root_url = "https://learn2rag.de"
        skipped: Set[str] = set()
        docs = load_html_content(root_url, depth=-1, loader_id="learn2rag_full", skipped=skipped)

        by_url: DefaultDict[str, List[Document]] = defaultdict(list)
        for doc in docs:
            by_url[doc.metadata.get("source", "?")].append(doc)

        visited_urls = set(by_url.keys())
        for i, doc in enumerate(docs, start=1):
            print(f"\n--- Document {i}: {doc.metadata.get('source')} ---")
            print(f"page_content (first 300 characters):\n{doc.page_content[:300]}")

        print(f"\n{'=' * 60}")
        print(f"SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Integrated (unique pages loaded): {len(visited_urls)}")
        print(f"  Skipped (off-site links):         {len(skipped)}")
        print(f"  Total documents (incl. duplicates): {len(docs)}")
        print(f"\n  Documents per URL:")
        for url in sorted(visited_urls):
            count = len(by_url[url])
            print(f"    [{count} doc(s)] {url}")
        print(f"\n  Skipped URLs (sample, max 20):")
        for url in sorted(skipped)[:20]:
            print(f"    [--] {url}")
        if len(skipped) > 20:
            print(f"         ... and {len(skipped) - 20} more")
        print(f"{'=' * 60}")

        # At least the root page must have been loaded
        self.assertTrue(len(docs) >= 1, "No documents found")
        # All documents must stay on the learn2rag.de domain
        for url in visited_urls:
            self.assertIn("learn2rag.de", url, f"External URL found: {url}")
        # Metadata must be complete
        for doc in docs:
            self.assertIn("loader_id", doc.metadata)
            self.assertIn("content_hash", doc.metadata)
            self.assertEqual(doc.metadata["loader_id"], "learn2rag_full")


class IsSameSiteTestCase(unittest.TestCase):
    """Unit tests for _is_same_site — no network access."""

    def test_same_domain_no_path(self) -> None:
        self.assertTrue(_is_same_site("https://example.com/page", "https://example.com"))

    def test_same_domain_with_path_prefix(self) -> None:
        self.assertTrue(_is_same_site("https://example.com/docs/guide", "https://example.com/docs/"))

    def test_different_path_prefix(self) -> None:
        self.assertFalse(_is_same_site("https://example.com/blog/post", "https://example.com/docs/"))

    def test_different_domain(self) -> None:
        self.assertFalse(_is_same_site("https://other.com/docs/page", "https://example.com/docs/"))

    def test_non_http_scheme(self) -> None:
        self.assertFalse(_is_same_site("mailto:info@example.com", "https://example.com"))

    def test_anchor_link(self) -> None:
        # Fragment-only links resolve to the same page and remain on-site
        self.assertTrue(_is_same_site("https://example.com/docs/page#section", "https://example.com/docs/"))


def _make_response(text: str, status_code: int = 200) -> MagicMock:
    """Helper: creates a fake requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.raise_for_status = MagicMock()
    return resp


def _make_json_response(payload: Dict[str, Any], status_code: int = 200) -> MagicMock:
    """Helper: creates a fake JSON response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload)
    return resp


ROOT_HTML = """
<html><head><title>Root</title></head><body>
  <p>Root page</p>
  <a href="/docs/page1">Page 1</a>
  <a href="/docs/page2">Page 2</a>
  <a href="/blog/post">Blog (other subtree)</a>
  <a href="https://external.com/">External</a>
</body></html>
"""

PAGE1_HTML = """
<html><head><title>Page 1</title></head><body>
  <p>Page 1 content</p>
  <a href="/docs/page2">Page 2 (already visited)</a>
</body></html>
"""

PAGE2_HTML = """
<html><head><title>Page 2</title></head><body>
  <p>Page 2 content</p>
</body></html>
"""


class HtmlLoaderDepthMinusOneTestCase(unittest.TestCase):
    """Tests for depth=-1 (full site crawl) using mocked HTTP requests."""

    def _fake_get(self, url: str, **kwargs: Any) -> MagicMock:
        pages = {
            "https://example.com/docs/": ROOT_HTML,
            "https://example.com/docs/page1": PAGE1_HTML,
            "https://example.com/docs/page2": PAGE2_HTML,
        }
        return _make_response(pages.get(url, "<html><body>Not found</body></html>"))

    @patch("learn2rag.importer.loaders.html_loader.requests.get")
    def test_crawls_same_subtree(self, mock_get: MagicMock) -> None:
        """depth=-1 should only visit URLs under /docs/, not /blog/ or external.com."""
        mock_get.side_effect = self._fake_get
        docs = load_html_content("https://example.com/docs/", depth=-1, loader_id="test")

        visited_urls = {doc.metadata["source"] for doc in docs}
        print(f"\nVisited URLs: {visited_urls}")

        # Expected: root + page1 + page2
        self.assertIn("https://example.com/docs/", visited_urls)
        self.assertIn("https://example.com/docs/page1", visited_urls)
        self.assertIn("https://example.com/docs/page2", visited_urls)

        # Not expected: different subtree and external domain
        self.assertNotIn("https://example.com/blog/post", visited_urls)
        self.assertNotIn("https://external.com/", visited_urls)

    @patch("learn2rag.importer.loaders.html_loader.requests.get")
    def test_no_duplicate_visits(self, mock_get: MagicMock) -> None:
        """Each URL is loaded only once (page2 is linked from both root and page1)."""
        mock_get.side_effect = self._fake_get
        docs = load_html_content("https://example.com/docs/", depth=-1, loader_id="test")

        sources = [doc.metadata["source"] for doc in docs]
        self.assertEqual(len(sources), len(set(sources)), "Duplicate URLs found")

    @patch("learn2rag.importer.loaders.html_loader.requests.get")
    def test_metadata_set_correctly(self, mock_get: MagicMock) -> None:
        """All documents have loader_id, content_hash and loader_type set."""
        mock_get.side_effect = self._fake_get
        docs = load_html_content("https://example.com/docs/", depth=-1, loader_id="test_meta")

        for doc in docs:
            self.assertEqual(doc.metadata.get("loader_id"), "test_meta")
            self.assertIn("content_hash", doc.metadata)
            self.assertEqual(doc.metadata.get("loader_type"), "HTMLLoader")


class JiraLoaderUnitTestCase(unittest.TestCase):
    """Unit tests for Jira loader with mocked HTTP session."""

    @patch("learn2rag.importer.loaders.jira_loader.requests.Session")
    def test_load_from_jira_maps_issue_to_document(self, mock_session_cls: MagicMock) -> None:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        issue_payload = {
            "issues": [
                {
                    "id": "10001",
                    "key": "DEMO-1",
                    "fields": {
                        "summary": "Demo issue",
                        "description": {
                            "type": "doc",
                            "content": [
                                {
                                    "type": "paragraph",
                                    "content": [{"type": "text", "text": "Description text"}],
                                }
                            ],
                        },
                        "status": {"name": "In Progress"},
                        "assignee": {"displayName": "Alice"},
                        "labels": ["backend", "priority-high"],
                        "updated": "2026-05-18T10:00:00.000+0000",
                        "created": "2026-05-17T09:00:00.000+0000",
                        "project": {"key": "DEMO", "name": "Demo Project"},
                        "comment": {
                            "comments": [
                                {
                                    "author": {"displayName": "Bob"},
                                    "body": {
                                        "type": "doc",
                                        "content": [
                                            {
                                                "type": "paragraph",
                                                "content": [{"type": "text", "text": "Looks good"}],
                                            }
                                        ],
                                    },
                                }
                            ]
                        },
                    },
                }
            ],
            "total": 1,
            "startAt": 0,
            "maxResults": 50,
        }

        mock_session.get.return_value = _make_json_response(issue_payload)

        docs = load_from_jira(
            base_url="https://jira.example.com",
            loader_id="jira_test",
            auth_type="basic",
            username="user@example.com",
            password="token123",
            jql="project = DEMO ORDER BY updated DESC",
            include_comments=True,
        )

        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertEqual(doc.metadata.get("loader_id"), "jira_test")
        self.assertEqual(doc.metadata.get("loader"), "JiraLoader")
        self.assertEqual(doc.metadata.get("issue_key"), "DEMO-1")
        self.assertEqual(doc.metadata.get("source"), "https://jira.example.com/browse/DEMO-1")
        self.assertIn("content_hash", doc.metadata)
        self.assertIn("Description text", doc.page_content)
        self.assertIn("Bob: Looks good", doc.page_content)

    @patch("learn2rag.importer.loaders.jira_loader.requests.Session")
    def test_get_all_jira_document_ids_pages_results(self, mock_session_cls: MagicMock) -> None:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        first_page = _make_json_response(
            {
                "issues": [{"id": "10001", "key": "DEMO-1"}],
                "total": 2,
                "startAt": 0,
                "maxResults": 1,
            }
        )
        second_page = _make_json_response(
            {
                "issues": [{"id": "10002", "key": "DEMO-2"}],
                "total": 2,
                "startAt": 1,
                "maxResults": 1,
            }
        )
        mock_session.get.side_effect = [first_page, second_page]

        ids = get_all_jira_document_ids(
            base_url="https://jira.example.com",
            auth_type="token",
            token="abc",
            projects=["DEMO"],
            page_size=1,
        )

        self.assertEqual(
            ids,
            [
                "https://jira.example.com/browse/DEMO-1",
                "https://jira.example.com/browse/DEMO-2",
            ],
        )

    @patch("learn2rag.importer.loaders.jira_loader.requests.Session")
    def test_load_from_jira_applies_since_filter_to_jql(self, mock_session_cls: MagicMock) -> None:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.get.return_value = _make_json_response(
            {
                "issues": [],
                "total": 0,
                "startAt": 0,
                "maxResults": 50,
            }
        )

        since = datetime(2026, 5, 18, 8, 30, tzinfo=timezone.utc)
        load_from_jira(
            base_url="https://jira.example.com",
            auth_type="none",
            jql="project = DEMO",
            since=since,
        )

        self.assertTrue(mock_session.get.called)
        _, kwargs = mock_session.get.call_args
        params = kwargs.get("params", {})
        jql_query = params.get("jql", "")
        self.assertIn("updated >= \"2026-05-18 08:30\"", jql_query)
