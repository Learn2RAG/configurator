import json
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

from ..loaders.directory_loader import load_from_directory
from ..loaders.html_loader import load_html_content, _is_same_site

class ImporterLoadersTestCase(unittest.TestCase):
    def test_remote_url(self) -> None:
        docs = load_html_content('https://learn2rag.de')
        assert len(docs) == 1
        doc, = docs
        assert 'source' in doc.metadata
        assert 'The DICE group at Paderborn University' in doc.page_content

    # def test_data_uri(self):
    #     docs = load_html_content('data:text/html;charset=utf-8,%3Cbody%3E%3Cp%3EData%20URI%20content%3C%2Fp%3E%3C%2Fbody%3E')
    #     assert len(docs) == 1
    #     doc, = docs
    #     assert doc.page_content == 'Data URI content'

    # TODO: actual tests

    def test_import_directory(self) -> None:
        """Loads files from ./data and prints what would be passed to Qdrant."""
        path = Path(__file__).parent.resolve() / 'data'
        docs = load_from_directory(str(path), recursive=True, loader_id="test_import")
        print(f"\n=== {len(docs)} document(s) loaded ===")
        for i, doc in enumerate(docs, start=1):
            print(f"\n--- Document {i} ---")
            print(f"Metadata: {json.dumps(doc.metadata, indent=2, default=str)}")
            print(f"page_content (first 500 characters):\n{doc.page_content[:500]}")
        self.assertTrue(len(docs) > 0, f"No documents found in: {path}")


class HtmlLoaderLearn2RagFullCrawlTestCase(unittest.TestCase):
    """Integration test: full site crawl of https://learn2rag.de with depth=-1."""

    def test_full_site_crawl(self) -> None:
        """Crawls the entire learn2rag.de domain and prints all discovered pages."""
        root_url = "https://learn2rag.de"
        skipped: set[str] = set()
        docs = load_html_content(root_url, depth=-1, loader_id="learn2rag_full", skipped=skipped)

        visited_urls = {doc.metadata["source"] for doc in docs}
        for i, doc in enumerate(docs, start=1):
            print(f"\n--- Document {i}: {doc.metadata.get('source')} ---")
            print(f"page_content (first 300 characters):\n{doc.page_content[:300]}")

        print(f"\n{'=' * 60}")
        print(f"SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Integrated (unique pages loaded): {len(visited_urls)}")
        print(f"  Skipped (off-site links):         {len(skipped)}")
        print(f"  Total documents (incl. duplicates): {len(docs)}")
        print(f"\n  Integrated URLs:")
        for url in sorted(visited_urls):
            print(f"    [OK] {url}")
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
