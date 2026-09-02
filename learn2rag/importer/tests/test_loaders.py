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
from .generate_test_documents import create_test_documents
from ..loaders.directory_loader import load_from_directory
from ..loaders.html_loader import load_html_content, _is_same_site
from ..loaders.jira_loader import load_from_jira, get_all_jira_document_ids
from ..loaders.mediawiki_loader import (
    MediaWikiSourceUnavailable,
    get_all_mediawiki_document_ids,
    load_from_mediawiki,
)
from ..loaders.errors import LoaderAccessError
from ..utils.import_state import ImportState

# Set RUN_INTEGRATION_TESTS=1 to run tests that require network access.
_RUN_INTEGRATION: bool = os.environ.get("RUN_INTEGRATION_TESTS", "0") == "1"

# Ensure test documents are generated at module load time
_TEST_DATA_DIR = pathlib.Path(__file__).parent / "data"


def setUpModule() -> None:
    """
    Module-level setup: Generate test documents once before all tests.
    This ensures all test data is available before any test runs.
    """
    try:
        create_test_documents(_TEST_DATA_DIR)
        print(f"\n✓ Test documents generated/verified in {_TEST_DATA_DIR}")
    except Exception as e:
        print(f"Warning: Could not generate test documents: {e}")
        raise RuntimeError(f"Failed to set up test environment: {e}")


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


class LoaderFailureHandlingTestCase(unittest.TestCase):
    """Regression tests for access errors and import-state thresholds."""

    def test_directory_loader_raises_loader_access_error_for_missing_path(self) -> None:
        with self.assertRaises(LoaderAccessError):
            load_from_directory("/definitely/not/here", recursive=False, loader_id="missing_dir")

    def test_import_state_tracks_consecutive_failures(self) -> None:
        temp_state = pathlib.Path(tempfile.mkdtemp(prefix="learn2rag_import_state_")) / "import_state.json"
        state = ImportState(str(temp_state))
        state.record_import_start("loader_1", datetime(2026, 8, 1, tzinfo=timezone.utc))
        state.save_success("loader_1")

        self.assertEqual(state.get_consecutive_failures("loader_1"), 0)
        self.assertEqual(state.record_failure("loader_1"), 1)
        self.assertEqual(state.get_consecutive_failures("loader_1"), 1)
        state.reset_failures("loader_1")
        self.assertEqual(state.get_consecutive_failures("loader_1"), 0)


class AllFileTypesTestCase(unittest.TestCase):
    """
    Comprehensive test case for all supported file types across loaders.
    Tests DirectoryLoader with various file formats and verifies proper parsing.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Ensure test documents exist in the data directory."""
        # Generate test documents (should be redundant due to setUpModule, but ensures safety)
        create_test_documents(_TEST_DATA_DIR)
        
        # Verify all required files are present
        required_files = {
            "sample.txt", "sample.csv", "product_inventory.csv",
            "event_logs.csv", "sample.md", "advanced.md", "simple.md",
            "sample.html", "sample.pdf", "sample.docx", "sample.xlsx", "sample.pptx"
        }
        existing_files = {f.name for f in _TEST_DATA_DIR.glob("*") if f.is_file()}
        missing = required_files - existing_files

        if missing:
            raise RuntimeError(
                f"❌ Missing test files: {missing}\n"
                f"   Expected files in: {_TEST_DATA_DIR}\n"
                f"   Found: {existing_files}\n"
                f"   Call create_test_documents() or run setUpModule()"
            )

    def test_all_supported_file_types_present(self) -> None:
        """Verify that all expected test documents exist."""
        expected_files = {
            "sample.txt": "Text document",
            "sample.csv": "CSV spreadsheet",
            "product_inventory.csv": "CSV with product data",
            "event_logs.csv": "CSV with event logs",
            "sample.md": "Markdown document",
            "advanced.md": "Advanced markdown with tables",
            "simple.md": "Simple markdown",
            "sample.html": "HTML document",
            "sample.pdf": "PDF document",
            "sample.docx": "Word document",
            "sample.xlsx": "Excel spreadsheet",
            "sample.pptx": "PowerPoint presentation",
        }

        for filename, description in expected_files.items():
            file_path = _TEST_DATA_DIR / filename
            self.assertTrue(
                file_path.exists(),
                f"Missing test file: {filename} ({description})"
            )
            self.assertGreater(
                file_path.stat().st_size, 0,
                f"Test file is empty: {filename}"
            )

    def test_load_all_file_types(self) -> None:
        """Load all test documents using DirectoryLoader and verify extraction."""
        docs = load_from_directory(str(_TEST_DATA_DIR), recursive=False, loader_id="all_types_test")

        # Organize by file type
        by_extension: DefaultDict[str, List[Document]] = defaultdict(list)
        for doc in docs:
            source = doc.metadata.get("source", "")
            ext = source.split(".")[-1].lower() if "." in source else "unknown"
            by_extension[ext].append(doc)

        print(f"\n=== Loaded {len(docs)} documents ===")
        for ext in sorted(by_extension.keys()):
            docs_of_type = by_extension[ext]
            print(f"\n.{ext}: {len(docs_of_type)} document(s)")
            for i, doc in enumerate(docs_of_type, 1):
                source = doc.metadata.get("source", "unknown")
                content_length = len(doc.page_content)
                print(f"  [{i}] {source}")
                print(f"      Content length: {content_length} chars")
                print(f"      Hash: {doc.metadata.get('content_hash', 'N/A')[:16]}...")
                if doc.page_content:
                    first_line = doc.page_content.split('\n')[0][:60]
                    print(f"      First line: {first_line}...")

        # Verify document counts
        self.assertGreater(len(docs), 0, "No documents loaded")

        # Each supported file type should have been loaded
        supported_extensions = {"txt", "csv", "md", "html", "pdf", "docx", "xlsx", "pptx"}
        loaded_extensions = set(by_extension.keys())
        self.assertTrue(
            supported_extensions.issubset(loaded_extensions),
            f"Not all file types loaded. Missing: {supported_extensions - loaded_extensions}"
        )

    def test_text_file_parsing(self) -> None:
        """Verify text files are parsed correctly."""
        docs = load_from_directory(str(_TEST_DATA_DIR), recursive=False, loader_id="txt_test")
        txt_docs = [d for d in docs if d.metadata.get("source", "").endswith(".txt")]

        self.assertGreater(len(txt_docs), 0, "No TXT files loaded")
        for doc in txt_docs:
            self.assertGreater(len(doc.page_content), 0, "TXT document has no content")
            self.assertIn("plain text", doc.page_content.lower(), "Expected content not found in TXT")

    def test_csv_file_parsing(self) -> None:
        """Verify CSV files are parsed correctly."""
        docs = load_from_directory(str(_TEST_DATA_DIR), recursive=False, loader_id="csv_test")
        csv_docs = [d for d in docs if d.metadata.get("source", "").endswith(".csv")]

        self.assertGreater(len(csv_docs), 0, "No CSV files loaded")
        
        # Should have at least 2 CSV files (basic + variants)
        self.assertGreaterEqual(len(csv_docs), 2, "Expected multiple CSV files")
        
        for doc in csv_docs:
            self.assertGreater(len(doc.page_content), 0, "CSV document has no content")

    def test_csv_product_inventory(self) -> None:
        """Verify product inventory CSV parsing."""
        docs = load_from_directory(str(_TEST_DATA_DIR), recursive=False, loader_id="csv_product_test")
        product_docs = [d for d in docs if "product_inventory" in d.metadata.get("source", "")]

        self.assertEqual(len(product_docs), 1, "Product inventory CSV not found")
        doc = product_docs[0]
        self.assertIn("product", doc.page_content.lower(), "Expected product data not found")
        self.assertIn("electronics", doc.page_content.lower(), "Expected category not found")

    def test_csv_event_logs(self) -> None:
        """Verify event logs CSV parsing."""
        docs = load_from_directory(str(_TEST_DATA_DIR), recursive=False, loader_id="csv_events_test")
        event_docs = [d for d in docs if "event_logs" in d.metadata.get("source", "")]

        self.assertEqual(len(event_docs), 1, "Event logs CSV not found")
        doc = event_docs[0]
        self.assertIn("login", doc.page_content.lower(), "Expected event type not found")
        self.assertIn("2026", doc.page_content, "Expected timestamp not found")

    def test_markdown_file_parsing(self) -> None:
        """Verify Markdown files are parsed correctly."""
        docs = load_from_directory(str(_TEST_DATA_DIR), recursive=False, loader_id="md_test")
        md_docs = [d for d in docs if d.metadata.get("source", "").endswith(".md")]

        self.assertGreater(len(md_docs), 0, "No MD files loaded")
        
        # Should have multiple Markdown files (basic + variants)
        self.assertGreaterEqual(len(md_docs), 2, "Expected multiple Markdown files")
        
        for doc in md_docs:
            self.assertGreater(len(doc.page_content), 0, "Markdown document has no content")

    def test_markdown_advanced_features(self) -> None:
        """Verify advanced markdown features are parsed."""
        docs = load_from_directory(str(_TEST_DATA_DIR), recursive=False, loader_id="md_advanced_test")
        advanced_docs = [d for d in docs if "advanced" in d.metadata.get("source", "")]

        self.assertEqual(len(advanced_docs), 1, "Advanced markdown file not found")
        doc = advanced_docs[0]
        # Verify content - HTML tags should be stripped but content preserved
        self.assertIn("markdown", doc.page_content.lower(), "Expected markdown content not found")

    def test_markdown_simple_variant(self) -> None:
        """Verify simple markdown variant parsing."""
        docs = load_from_directory(str(_TEST_DATA_DIR), recursive=False, loader_id="md_simple_test")
        simple_docs = [d for d in docs if "simple" in d.metadata.get("source", "") and d.metadata.get("source", "").endswith(".md")]

        self.assertEqual(len(simple_docs), 1, "Simple markdown file not found")
        doc = simple_docs[0]
        self.assertIn("simple", doc.page_content.lower(), "Expected 'simple' keyword in content")

    def test_html_file_parsing(self) -> None:
        """Verify HTML files are parsed correctly (HTML tags removed)."""
        docs = load_from_directory(str(_TEST_DATA_DIR), recursive=False, loader_id="html_test")
        html_docs = [d for d in docs if d.metadata.get("source", "").endswith(".html")]

        self.assertGreater(len(html_docs), 0, "No HTML files loaded")
        for doc in html_docs:
            self.assertGreater(len(doc.page_content), 0, "HTML document has no content")
            # Verify that HTML tags have been stripped
            self.assertNotIn("<html", doc.page_content.lower(), "HTML tags not stripped")
            self.assertIn("document", doc.page_content.lower(), "Expected content not found in HTML")

    def test_pdf_file_parsing(self) -> None:
        """Verify PDF files are parsed correctly."""
        docs = load_from_directory(str(_TEST_DATA_DIR), recursive=False, loader_id="pdf_test")
        pdf_docs = [d for d in docs if d.metadata.get("source", "").endswith(".pdf")]

        self.assertGreater(len(pdf_docs), 0, "No PDF files loaded")
        for doc in pdf_docs:
            self.assertGreater(len(doc.page_content), 0, "PDF document has no content")

    def test_docx_file_parsing(self) -> None:
        """Verify DOCX (Word) files are parsed correctly."""
        docs = load_from_directory(str(_TEST_DATA_DIR), recursive=False, loader_id="docx_test")
        docx_docs = [d for d in docs if d.metadata.get("source", "").endswith(".docx")]

        self.assertGreater(len(docx_docs), 0, "No DOCX files loaded")
        for doc in docx_docs:
            self.assertGreater(len(doc.page_content), 0, "DOCX document has no content")

    def test_xlsx_file_parsing(self) -> None:
        """Verify XLSX (Excel) files are parsed correctly."""
        docs = load_from_directory(str(_TEST_DATA_DIR), recursive=False, loader_id="xlsx_test")
        xlsx_docs = [d for d in docs if d.metadata.get("source", "").endswith(".xlsx")]

        self.assertGreater(len(xlsx_docs), 0, "No XLSX files loaded")
        for doc in xlsx_docs:
            self.assertGreater(len(doc.page_content), 0, "XLSX document has no content")

    def test_pptx_file_parsing(self) -> None:
        """Verify PPTX (PowerPoint) files are parsed correctly."""
        docs = load_from_directory(str(_TEST_DATA_DIR), recursive=False, loader_id="pptx_test")
        pptx_docs = [d for d in docs if d.metadata.get("source", "").endswith(".pptx")]

        self.assertGreater(len(pptx_docs), 0, "No PPTX files loaded")
        for doc in pptx_docs:
            self.assertGreater(len(doc.page_content), 0, "PPTX document has no content")

    def test_metadata_consistency(self) -> None:
        """Verify that all loaded documents have consistent metadata."""
        docs = load_from_directory(str(_TEST_DATA_DIR), recursive=False, loader_id="metadata_test")

        required_metadata_fields = {
            "source",
            "content_hash",
            "file_extension",
            "loader_type",
            "loader_id",
        }

        for doc in docs:
            doc_source = doc.metadata.get("source", "unknown")
            for field in required_metadata_fields:
                self.assertIn(
                    field, doc.metadata,
                    f"Missing metadata field '{field}' in {doc_source}"
                )
            self.assertEqual(
                doc.metadata.get("loader_type"), "DirectoryLoader",
                f"Unexpected loader_type for {doc_source}"
            )
            self.assertEqual(
                doc.metadata.get("loader_id"), "metadata_test",
                f"Unexpected loader_id for {doc_source}"
            )


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


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _doc(source: str, content_hash: str) -> Document:
    """Create a minimal Document with source and content_hash metadata."""
    return Document(
        page_content="content",
        metadata={"source": source, "content_hash": content_hash},
    )


_USER_CONFIG: dict[str, str] = {"collection_name": "test_collection"}
_OPT_CONFIG: dict[str, Any] = {}

# ---------------------------------------------------------------------------
# _delta_by_source unit tests (no Qdrant required)
# ---------------------------------------------------------------------------


class DeltaBySourceTestCase(unittest.TestCase):
    """Unit tests for _delta_by_source.

    All Qdrant calls (delete_documents, update_documents) are mocked so no
    running Qdrant instance is required.
    """

    def setUp(self) -> None:
        from ..loaders.process_loaders import _delta_by_source  # local import to avoid top-level side effects
        self._delta_by_source = _delta_by_source
        self._patch_delete = patch("learn2rag.importer.loaders.process_loaders.delete_documents")
        self._patch_update = patch("learn2rag.importer.loaders.process_loaders.update_documents")
        self.mock_delete = self._patch_delete.start()
        self.mock_update = self._patch_update.start()

    def tearDown(self) -> None:
        self._patch_delete.stop()
        self._patch_update.stop()

    # -- no changes --------------------------------------------------------

    def test_no_changes_no_qdrant_calls(self) -> None:
        """When nothing changed, neither delete nor update should be called."""
        existing_map = {"a.txt": "hash_a", "b.txt": "hash_b"}
        all_docs = [_doc("a.txt", "hash_a"), _doc("b.txt", "hash_b")]

        self._delta_by_source(all_docs, existing_map, "loader1", _USER_CONFIG, _OPT_CONFIG)

        self.mock_delete.assert_not_called()
        self.mock_update.assert_not_called()

    # -- new document ------------------------------------------------------

    def test_new_document_is_updated(self) -> None:
        """A source that does not exist in Qdrant yet must be passed to update_documents."""
        existing_map = {"a.txt": "hash_a"}
        all_docs = [_doc("a.txt", "hash_a"), _doc("new.txt", "hash_new")]

        self._delta_by_source(all_docs, existing_map, "loader1", _USER_CONFIG, _OPT_CONFIG)

        self.mock_delete.assert_not_called()
        updated_sources = {d.metadata["source"] for d in self.mock_update.call_args[0][1]}
        self.assertIn("new.txt", updated_sources)
        self.assertNotIn("a.txt", updated_sources)

    # -- changed document --------------------------------------------------

    def test_changed_document_is_updated(self) -> None:
        """A source whose hash differs from the stored one must be re-indexed."""
        existing_map = {"a.txt": "hash_a_old"}
        all_docs = [_doc("a.txt", "hash_a_new")]

        self._delta_by_source(all_docs, existing_map, "loader1", _USER_CONFIG, _OPT_CONFIG)

        self.mock_delete.assert_not_called()
        updated_sources = {d.metadata["source"] for d in self.mock_update.call_args[0][1]}
        self.assertIn("a.txt", updated_sources)

    # -- deleted document --------------------------------------------------

    def test_deleted_document_is_removed(self) -> None:
        """A source present in Qdrant but absent from the fresh load must be deleted."""
        existing_map = {"a.txt": "hash_a", "gone.txt": "hash_gone"}
        all_docs = [_doc("a.txt", "hash_a")]

        self._delta_by_source(all_docs, existing_map, "loader1", _USER_CONFIG, _OPT_CONFIG)

        deleted = self.mock_delete.call_args[0][1]
        self.assertIn("gone.txt", deleted)
        self.mock_update.assert_not_called()

    # -- mixed scenario ----------------------------------------------------

    def test_mixed_new_changed_deleted_unchanged(self) -> None:
        """Combined scenario: new, changed, deleted and unchanged in one call."""
        existing_map = {
            "unchanged.txt": "hash_u",
            "changed.txt": "hash_c_old",
            "deleted.txt": "hash_d",
        }
        all_docs = [
            _doc("unchanged.txt", "hash_u"),     # unchanged — must not be touched
            _doc("changed.txt", "hash_c_new"),   # changed   — must be re-indexed
            _doc("new.txt", "hash_n"),            # new       — must be indexed
            # deleted.txt is absent              # deleted   — must be removed
        ]

        self._delta_by_source(all_docs, existing_map, "loader1", _USER_CONFIG, _OPT_CONFIG)

        deleted = self.mock_delete.call_args[0][1]
        self.assertIn("deleted.txt", deleted)
        self.assertNotIn("unchanged.txt", deleted)

        updated_sources = {d.metadata["source"] for d in self.mock_update.call_args[0][1]}
        self.assertIn("changed.txt", updated_sources)
        self.assertIn("new.txt", updated_sources)
        self.assertNotIn("unchanged.txt", updated_sources)

    # -- empty existing map (initial run) ----------------------------------

    def test_initial_run_all_documents_indexed(self) -> None:
        """On the first run (empty Qdrant) every document must be passed to update_documents."""
        existing_map: dict[str, str] = {}
        all_docs = [_doc("a.txt", "hash_a"), _doc("b.txt", "hash_b")]

        self._delta_by_source(all_docs, existing_map, "loader1", _USER_CONFIG, _OPT_CONFIG)

        self.mock_delete.assert_not_called()
        updated_sources = {d.metadata["source"] for d in self.mock_update.call_args[0][1]}
        self.assertEqual(updated_sources, {"a.txt", "b.txt"})

    # -- empty fresh load (all documents removed) --------------------------

    def test_all_documents_removed(self) -> None:
        """If the loader returns nothing, all existing sources must be deleted."""
        existing_map = {"a.txt": "hash_a", "b.txt": "hash_b"}
        all_docs: list[Document] = []

        self._delta_by_source(all_docs, existing_map, "loader1", _USER_CONFIG, _OPT_CONFIG)

        deleted = self.mock_delete.call_args[0][1]
        self.assertCountEqual(deleted, ["a.txt", "b.txt"])
        self.mock_update.assert_not_called()


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

        # Simulate a Jira Server without the new POST /search/jql endpoint so the loader falls back to GET.
        mock_session.post.return_value = _make_response("", status_code=404)
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
        mock_session.post.return_value = _make_response("", status_code=404)
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
        mock_session.post.return_value = _make_response("", status_code=404)
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


class MediaWikiLoaderUnitTestCase(unittest.TestCase):
    """Unit tests for MediaWiki loader with mocked HTTP session."""

    @patch("learn2rag.importer.loaders.mediawiki_loader.requests.Session")
    @patch("learn2rag.importer.loaders.mediawiki_loader._resolve_api_url", return_value="https://wiki.example.com/api.php")
    def test_load_from_mediawiki_maps_pages_to_documents(
        self,
        mock_resolve_api: MagicMock,  # 1. Argument (unten)
        mock_session_cls: MagicMock,  # 2. Argument (oben)
    ) -> None:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        allpages_page_1 = _make_json_response(
            {
                "query": {"allpages": [{"pageid": 1, "title": "Page One"}]},
                "continue": {"apcontinue": "Page_Two"},
            }
        )
        allpages_page_2 = _make_json_response(
            {
                "query": {"allpages": [{"pageid": 2, "title": "Page Two"}]},
            }
        )
        pages_payload = _make_json_response(
            {
                "query": {
                    "pages": [
                        {
                            "pageid": 1,
                            "ns": 0,
                            "title": "Page One",
                            "extract": "First page content",
                            "revisions": [{"timestamp": "2026-08-10T11:00:00Z"}],
                        },
                        {
                            "pageid": 2,
                            "ns": 0,
                            "title": "Page Two",
                            "extract": "Second page content",
                            "revisions": [{"timestamp": "2026-08-10T12:00:00Z"}],
                        },
                    ]
                }
            }
        )
        mock_session.get.side_effect = [allpages_page_1, allpages_page_2, pages_payload]

        docs = load_from_mediawiki(
            base_url="https://wiki.example.com",
            loader_id="wiki_test",
            auth_type="none",
            namespaces=[0],
            page_size=50,
        )

        self.assertEqual(len(docs), 2)
        sources = {doc.metadata.get("source") for doc in docs}
        self.assertIn("https://wiki.example.com/?curid=1", sources)
        self.assertIn("https://wiki.example.com/?curid=2", sources)
        self.assertTrue(all(doc.metadata.get("loader") == "MediaWikiLoader" for doc in docs))
        self.assertTrue(all("content_hash" in doc.metadata for doc in docs))

    @patch("learn2rag.importer.loaders.mediawiki_loader.requests.Session")
    @patch("learn2rag.importer.loaders.mediawiki_loader._resolve_api_url", return_value="https://wiki.example.com/api.php")
    def test_get_all_mediawiki_document_ids_with_pagination(
        self,
        mock_resolve_api: MagicMock,  # 1. Argument entspricht dem unteren Patch
        mock_session_cls: MagicMock,  # 2. Argument entspricht dem oberen Patch
    ) -> None:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        mock_session.get.side_effect = [
            _make_json_response(
                {
                    "query": {"allpages": [{"pageid": 10, "title": "Alpha"}]},
                    "continue": {"apcontinue": "Beta"},
                }
            ),
            _make_json_response(
                {
                    "query": {"allpages": [{"pageid": 11, "title": "Beta"}]},
                }
            ),
        ]

        ids = get_all_mediawiki_document_ids(
            base_url="https://wiki.example.com",
            auth_type="none",
            namespaces=[0],
            page_size=1,
        )

        self.assertEqual(
            ids,
            [
                "https://wiki.example.com/?curid=10",
                "https://wiki.example.com/?curid=11",
            ],
        )

    @patch("learn2rag.importer.loaders.mediawiki_loader._resolve_api_url", return_value="https://wiki.example.com/api.php")
    @patch("learn2rag.importer.loaders.mediawiki_loader.requests.Session")
    def test_load_from_mediawiki_applies_since_filter(self, mock_session_cls: MagicMock, mock_resolve_api: MagicMock) -> None:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.get.return_value = _make_json_response({"query": {"recentchanges": []}})

        since = datetime(2026, 8, 10, 9, 30, tzinfo=timezone.utc)
        docs = load_from_mediawiki(
            base_url="https://wiki.example.com",
            auth_type="none",
            namespaces=[0],
            since=since,
        )

        self.assertEqual(docs, [])
        self.assertTrue(mock_session.get.called)
        _, kwargs = mock_session.get.call_args
        params = kwargs.get("params", {})
        self.assertEqual(params.get("list"), "recentchanges")
        self.assertEqual(params.get("rcstart"), "2026-08-10T09:30:00Z")


class MediaWikiDeltaFailSafeTestCase(unittest.TestCase):
    """Ensure MediaWiki delta mode does not alter index on source outages."""

    @patch("learn2rag.importer.loaders.process_loaders.index")
    @patch("learn2rag.importer.loaders.process_loaders.delete_documents")
    @patch("learn2rag.importer.loaders.process_loaders.load_from_mediawiki")
    @patch("learn2rag.importer.loaders.process_loaders.get_all_mediawiki_document_ids")
    @patch("learn2rag.importer.loaders.process_loaders.get_documents")
    def test_empty_snapshot_skips_delta_without_index_changes(
        self,
        mock_get_documents: MagicMock,
        mock_get_all_ids: MagicMock,
        mock_load_from_mediawiki: MagicMock,
        mock_delete_documents: MagicMock,
        mock_index: MagicMock,
    ) -> None:
        from ..loaders.process_loaders import process_delta_imports

        mock_get_documents.return_value = {"https://wiki.example.com/?curid=1": "hash1"}
        mock_get_all_ids.return_value = []

        import_state = MagicMock()
        import_state.get_last_import_time.return_value = datetime(2026, 8, 1, tzinfo=timezone.utc)

        process_delta_imports(
            config_entries=[
                {
                    "loader_type": "MediaWikiLoader",
                    "loader_id": "wiki_loader",
                    "base_url": "https://wiki.example.com",
                    "auth_type": "none",
                    "namespaces": [0],
                }
            ],
            user_config=_USER_CONFIG,
            opt_config=_OPT_CONFIG,
            import_state=import_state,
            progress=None,
        )

        mock_delete_documents.assert_not_called()
        mock_load_from_mediawiki.assert_not_called()
        mock_index.assert_not_called()
        import_state.save_success.assert_not_called()

    @patch("learn2rag.importer.loaders.process_loaders.index")
    @patch("learn2rag.importer.loaders.process_loaders.delete_documents")
    @patch("learn2rag.importer.loaders.process_loaders.get_all_mediawiki_document_ids")
    @patch("learn2rag.importer.loaders.process_loaders.get_documents")
    def test_source_unavailable_skips_delta_without_index_changes(
        self,
        mock_get_documents: MagicMock,
        mock_get_all_ids: MagicMock,
        mock_delete_documents: MagicMock,
        mock_index: MagicMock,
    ) -> None:
        from ..loaders.process_loaders import process_delta_imports

        mock_get_documents.return_value = {"https://wiki.example.com/?curid=1": "hash1"}
        mock_get_all_ids.side_effect = MediaWikiSourceUnavailable("unreachable")

        import_state = MagicMock()
        import_state.get_last_import_time.return_value = datetime(2026, 8, 1, tzinfo=timezone.utc)

        process_delta_imports(
            config_entries=[
                {
                    "loader_type": "MediaWikiLoader",
                    "loader_id": "wiki_loader",
                    "base_url": "https://wiki.example.com",
                    "auth_type": "none",
                    "namespaces": [0],
                }
            ],
            user_config=_USER_CONFIG,
            opt_config=_OPT_CONFIG,
            import_state=import_state,
            progress=None,
        )

        mock_delete_documents.assert_not_called()
        mock_index.assert_not_called()
        import_state.save_success.assert_not_called()
