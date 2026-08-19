import os
import unittest
from pathlib import Path
from typing import Any, ClassVar

from dotenv import load_dotenv
from langchain_core.documents import Document
from qdrant_client.models import Distance

from ..qdrant import Qdrant
from ..ingestion import index
from ..search import search, search_multi, _collect_query_points
from ..store import get_documents, delete_documents, update_documents

load_dotenv(Path(__file__).resolve().parents[3] / '.env')

QDRANT_TEST_LOCATION = os.environ.get('QDRANT_TEST_LOCATION') or os.environ.get('QDRANT_LOCATION', 'http://localhost:6336')
QDRANT_TEST_API_KEY = os.environ.get('QDRANT_TEST_API_KEY') or os.environ.get('QDRANT__SERVICE__API_KEY', '')

EMBEDDING_MODEL = 'BAAI/bge-m3'
VECTOR_SIZE = 1024


def make_opt_config(
    search_mode: str = 'dense',
    query_mode: str = 'single',
    multi_search: list[str] | None = None,
) -> dict[str, Any]:
    return {
        'chunk_size': 500,
        'chunk_overlap': 50,
        'ingestion_batch_size': 32,
        'embedding_model': EMBEDDING_MODEL,
        'vector_size': {
            'sentence-transformers/all-mpnet-base-v2': 768,
            'BAAI/bge-m3': VECTOR_SIZE,
        },
        'search_mode': search_mode,
        'top_k': 3,
        'reranking': 'False',
        'reranking_mode': 'reranking_with_sentence_transformers',
        'top_k_reranker': 3,
        'fusion_mode': 'DBSF',
        'rewrite': 'False',
        'rewrite_mode': 'subqueries_keywords',
        'n_subqueries': 3,
        'n_keywords': 3,
        'top_k_subqueries': 3,
        'top_k_keywords': 3,
        'prefetch_limit_dense': 25,
        'prefetch_limit_sparse': 25,
        'prefetch_limit_colbert': 25,
        'query_mode': query_mode,
        'multi_search': multi_search or [],
    }


SAMPLE_DOCUMENTS = [
    Document(
        page_content='Rabbits are small mammals in the family Leporidae, in the order Lagomorpha. '
                     'They are familiar throughout the world as a small herbivore and a domesticated pet.',
        metadata={
            'source': 'test/rabbits.txt', 'content_hash': 'abc123',
            'loader_id': 'test_loader', 'title': 'Rabbits', 'summary': 'About rabbits',
        },
    ),
    Document(
        page_content='Python is a high-level programming language known for its readability and versatility. '
                     'It supports multiple programming paradigms including procedural, object-oriented, and functional.',
        metadata={
            'source': 'test/python.txt', 'content_hash': 'def456',
            'loader_id': 'test_loader', 'title': 'Python', 'summary': 'About Python programming',
        },
    ),
    Document(
        page_content='Qdrant is a vector similarity search engine that provides a production-ready service '
                     'with a convenient API to store, search, and manage vectors with additional payload.',
        metadata={
            'source': 'test/qdrant.txt', 'content_hash': 'ghi789',
            'loader_id': 'test_loader', 'title': 'Qdrant', 'summary': 'About vector search',
        },
    ),
]


def _setup_qdrant_env() -> None:
    os.environ['QDRANT_LOCATION'] = QDRANT_TEST_LOCATION
    os.environ['QDRANT__SERVICE__API_KEY'] = QDRANT_TEST_API_KEY
    os.environ.pop('QDRANT_PATH', None)

    from qdrant_client import QdrantClient
    Qdrant.client = QdrantClient(
        location=QDRANT_TEST_LOCATION,
        api_key=QDRANT_TEST_API_KEY,
    )


def _cleanup_collection(name: str) -> None:
    if Qdrant.client.collection_exists(name):
        Qdrant.client.delete_collection(name)


@unittest.skipUnless(
    os.environ.get('RUN_INTEGRATION_TESTS'),
    'Requires RUN_INTEGRATION_TESTS=1 and a running Qdrant server',
)
class QdrantConnectionTestCase(unittest.TestCase):
    """Basic connection and collection management tests."""

    collection_name = 'test_integration_connection'

    @classmethod
    def setUpClass(cls) -> None:
        _setup_qdrant_env()

    def setUp(self) -> None:
        _cleanup_collection(self.collection_name)

    def tearDown(self) -> None:
        _cleanup_collection(self.collection_name)

    def test_connection(self) -> None:
        collections = Qdrant.client.get_collections()
        self.assertIsNotNone(collections)

    def test_ensure_collection_dense(self) -> None:
        opt = make_opt_config(search_mode='dense')
        Qdrant.ensure_collection(self.collection_name, opt)

        info = Qdrant.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        assert isinstance(vectors, dict)
        self.assertIn('dense', vectors)
        self.assertEqual(vectors['dense'].size, VECTOR_SIZE)
        self.assertEqual(vectors['dense'].distance, Distance.COSINE)

    def test_ensure_collection_dense_sparse(self) -> None:
        opt = make_opt_config(search_mode='dense_sparse')
        Qdrant.ensure_collection(self.collection_name, opt)

        info = Qdrant.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        sparse_vectors = info.config.params.sparse_vectors
        assert isinstance(vectors, dict)
        assert isinstance(sparse_vectors, dict)
        self.assertIn('dense', vectors)
        self.assertIn('sparse', sparse_vectors)

    def test_ensure_collection_dense_sparse_colbert(self) -> None:
        opt = make_opt_config(search_mode='dense_sparse_colbert')
        Qdrant.ensure_collection(self.collection_name, opt)

        info = Qdrant.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        sparse_vectors = info.config.params.sparse_vectors
        assert isinstance(vectors, dict)
        assert isinstance(sparse_vectors, dict)
        self.assertIn('dense', vectors)
        self.assertIn('colbert', vectors)
        self.assertIn('sparse', sparse_vectors)
        self.assertEqual(vectors['colbert'].size, VECTOR_SIZE)
        self.assertIsNotNone(vectors['colbert'].multivector_config)

    def test_ensure_collection_multi_vector(self) -> None:
        multi_search = ['title', 'summary']
        opt = make_opt_config(query_mode='multi', multi_search=multi_search)
        Qdrant.ensure_collection(self.collection_name, opt)

        expected_size = (len(multi_search) + 1) * VECTOR_SIZE
        info = Qdrant.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        assert isinstance(vectors, dict)
        self.assertIn('multi', vectors)
        self.assertEqual(vectors['multi'].size, expected_size)

    def test_ensure_collection_idempotent(self) -> None:
        opt = make_opt_config()
        Qdrant.ensure_collection(self.collection_name, opt)
        Qdrant.ensure_collection(self.collection_name, opt)
        self.assertTrue(Qdrant.client.collection_exists(self.collection_name))

    def test_delete_collection(self) -> None:
        Qdrant.ensure_collection(self.collection_name, make_opt_config())
        self.assertTrue(Qdrant.client.collection_exists(self.collection_name))

        Qdrant.client.delete_collection(self.collection_name)
        self.assertFalse(Qdrant.client.collection_exists(self.collection_name))


@unittest.skipUnless(
    os.environ.get('RUN_INTEGRATION_TESTS'),
    'Requires RUN_INTEGRATION_TESTS=1 and a running Qdrant server',
)
class QdrantDenseSearchTestCase(unittest.TestCase):
    """Ingest and search with dense-only mode (bge-m3)."""

    collection_name = 'test_integration_dense'
    user_config: ClassVar[dict[str, Any]]
    opt_config: ClassVar[dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        _setup_qdrant_env()
        cls.user_config = {'collection_name': cls.collection_name}
        cls.opt_config = make_opt_config(search_mode='dense')
        _cleanup_collection(cls.collection_name)
        index(SAMPLE_DOCUMENTS, cls.user_config, cls.opt_config)

    @classmethod
    def tearDownClass(cls) -> None:
        _cleanup_collection(cls.collection_name)

    def test_documents_ingested(self) -> None:
        info = Qdrant.client.get_collection(self.collection_name)
        self.assertEqual(info.points_count, len(SAMPLE_DOCUMENTS))

    def test_search_finds_relevant_result(self) -> None:
        results = search('What are rabbits?', self.user_config, self.opt_config)
        self.assertGreater(len(results.points), 0)
        payload = results.points[0].payload
        assert payload is not None
        self.assertIn('Lagomorpha', payload['content'])

    def test_search_relevance_ranking(self) -> None:
        results = search('vector similarity search engine', self.user_config, self.opt_config)
        payload = results.points[0].payload
        assert payload is not None
        self.assertIn('Qdrant', payload['content'])

    def test_search_returns_payload_fields(self) -> None:
        results = search('programming language', self.user_config, self.opt_config)
        self.assertGreater(len(results.points), 0)
        payload = results.points[0].payload
        assert payload is not None
        for field in ('content', 'source', 'content_hash', 'loader_id', 'chunk_hash'):
            self.assertIn(field, payload, f'Missing payload field: {field}')

    def test_deduplication(self) -> None:
        index(SAMPLE_DOCUMENTS, self.user_config, self.opt_config)
        info = Qdrant.client.get_collection(self.collection_name)
        self.assertEqual(info.points_count, len(SAMPLE_DOCUMENTS))


@unittest.skipUnless(
    os.environ.get('RUN_INTEGRATION_TESTS'),
    'Requires RUN_INTEGRATION_TESTS=1 and a running Qdrant server',
)
class QdrantSparseSearchTestCase(unittest.TestCase):
    """Ingest with dense_sparse mode but search using only sparse (BM25)."""

    collection_name = 'test_integration_sparse'
    user_config: ClassVar[dict[str, Any]]
    opt_config_ingest: ClassVar[dict[str, Any]]
    opt_config_search: ClassVar[dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        _setup_qdrant_env()
        cls.user_config = {'collection_name': cls.collection_name}
        cls.opt_config_ingest = make_opt_config(search_mode='dense_sparse')
        cls.opt_config_search = make_opt_config(search_mode='sparse')
        _cleanup_collection(cls.collection_name)
        index(SAMPLE_DOCUMENTS, cls.user_config, cls.opt_config_ingest)

    @classmethod
    def tearDownClass(cls) -> None:
        _cleanup_collection(cls.collection_name)

    def test_documents_ingested(self) -> None:
        info = Qdrant.client.get_collection(self.collection_name)
        self.assertEqual(info.points_count, len(SAMPLE_DOCUMENTS))

    def test_sparse_search_finds_results(self) -> None:
        results = search('rabbits mammals Leporidae', self.user_config, self.opt_config_search)
        self.assertGreater(len(results.points), 0)

    def test_sparse_search_relevance(self) -> None:
        results = search('vector similarity Qdrant', self.user_config, self.opt_config_search)
        self.assertGreater(len(results.points), 0)
        payload = results.points[0].payload
        assert payload is not None
        self.assertIn('Qdrant', payload['content'])


@unittest.skipUnless(
    os.environ.get('RUN_INTEGRATION_TESTS'),
    'Requires RUN_INTEGRATION_TESTS=1 and a running Qdrant server',
)
class QdrantDenseSparseSearchTestCase(unittest.TestCase):
    """Ingest and search with hybrid dense+sparse mode (fusion)."""

    collection_name = 'test_integration_dense_sparse'
    user_config: ClassVar[dict[str, Any]]
    opt_config: ClassVar[dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        _setup_qdrant_env()
        cls.user_config = {'collection_name': cls.collection_name}
        cls.opt_config = make_opt_config(search_mode='dense_sparse')
        _cleanup_collection(cls.collection_name)
        index(SAMPLE_DOCUMENTS, cls.user_config, cls.opt_config)

    @classmethod
    def tearDownClass(cls) -> None:
        _cleanup_collection(cls.collection_name)

    def test_documents_ingested(self) -> None:
        info = Qdrant.client.get_collection(self.collection_name)
        self.assertEqual(info.points_count, len(SAMPLE_DOCUMENTS))

    def test_hybrid_search_finds_results(self) -> None:
        results = search('What are rabbits?', self.user_config, self.opt_config)
        self.assertGreater(len(results.points), 0)
        payload = results.points[0].payload
        assert payload is not None
        self.assertIn('Lagomorpha', payload['content'])

    def test_hybrid_search_relevance(self) -> None:
        results = search('programming language paradigms', self.user_config, self.opt_config)
        self.assertGreater(len(results.points), 0)
        payload = results.points[0].payload
        assert payload is not None
        self.assertIn('Python', payload['content'])


@unittest.skipUnless(
    os.environ.get('RUN_INTEGRATION_TESTS'),
    'Requires RUN_INTEGRATION_TESTS=1 and a running Qdrant server',
)
class QdrantDenseSparseColbertSearchTestCase(unittest.TestCase):
    """Ingest and search with dense+sparse+colbert mode."""

    collection_name = 'test_integration_dense_sparse_colbert'
    user_config: ClassVar[dict[str, Any]]
    opt_config: ClassVar[dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        _setup_qdrant_env()
        cls.user_config = {'collection_name': cls.collection_name}
        cls.opt_config = make_opt_config(search_mode='dense_sparse_colbert')
        _cleanup_collection(cls.collection_name)
        index(SAMPLE_DOCUMENTS, cls.user_config, cls.opt_config)

    @classmethod
    def tearDownClass(cls) -> None:
        _cleanup_collection(cls.collection_name)

    def test_documents_ingested(self) -> None:
        info = Qdrant.client.get_collection(self.collection_name)
        self.assertEqual(info.points_count, len(SAMPLE_DOCUMENTS))

    def test_collection_has_all_vector_types(self) -> None:
        info = Qdrant.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        sparse_vectors = info.config.params.sparse_vectors
        assert isinstance(vectors, dict)
        assert isinstance(sparse_vectors, dict)
        self.assertIn('dense', vectors)
        self.assertIn('colbert', vectors)
        self.assertIn('sparse', sparse_vectors)

    def test_search_finds_results(self) -> None:
        results = search('What are rabbits?', self.user_config, self.opt_config)
        self.assertGreater(len(results.points), 0)
        payload = results.points[0].payload
        assert payload is not None
        self.assertIn('Lagomorpha', payload['content'])

    def test_search_relevance(self) -> None:
        results = search('vector database search engine API', self.user_config, self.opt_config)
        self.assertGreater(len(results.points), 0)
        payload = results.points[0].payload
        assert payload is not None
        self.assertIn('Qdrant', payload['content'])


@unittest.skipUnless(
    os.environ.get('RUN_INTEGRATION_TESTS'),
    'Requires RUN_INTEGRATION_TESTS=1 and a running Qdrant server',
)
class QdrantMultiVectorSearchTestCase(unittest.TestCase):
    """Ingest and search with multi-vector mode (content + metadata embeddings concatenated)."""

    collection_name = 'test_integration_multi_vector'
    multi_search_fields: ClassVar[list[str]] = ['title', 'summary']
    user_config: ClassVar[dict[str, Any]]
    opt_config: ClassVar[dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        _setup_qdrant_env()
        cls.user_config = {'collection_name': cls.collection_name}
        cls.opt_config = make_opt_config(
            search_mode='dense',
            query_mode='multi',
            multi_search=cls.multi_search_fields,
        )
        _cleanup_collection(cls.collection_name)
        index(SAMPLE_DOCUMENTS, cls.user_config, cls.opt_config)

    @classmethod
    def tearDownClass(cls) -> None:
        _cleanup_collection(cls.collection_name)

    def test_documents_ingested(self) -> None:
        info = Qdrant.client.get_collection(self.collection_name)
        self.assertEqual(info.points_count, len(SAMPLE_DOCUMENTS))

    def test_collection_has_correct_vector_size(self) -> None:
        expected_size = (len(self.multi_search_fields) + 1) * VECTOR_SIZE
        info = Qdrant.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        assert isinstance(vectors, dict)
        self.assertIn('multi', vectors)
        self.assertEqual(vectors['multi'].size, expected_size)

    def test_multi_search_finds_results(self) -> None:
        multi_query = {
            'content': 'What are rabbits?',
            'title': 'Rabbits',
            'summary': 'About rabbits',
        }
        results = search_multi(multi_query, self.user_config, self.opt_config)
        self.assertGreater(len(results.points), 0)

    def test_multi_search_relevance(self) -> None:
        multi_query = {
            'content': 'vector search engine',
            'title': 'Qdrant',
            'summary': 'vector search',
        }
        results = search_multi(multi_query, self.user_config, self.opt_config)
        self.assertGreater(len(results.points), 0)
        payload = results.points[0].payload
        assert payload is not None
        self.assertIn('Qdrant', payload['content'])


@unittest.skipUnless(
    os.environ.get('RUN_INTEGRATION_TESTS'),
    'Requires RUN_INTEGRATION_TESTS=1 and a running Qdrant server',
)
class QdrantFusionRRFTestCase(unittest.TestCase):
    """Test hybrid search with RRF fusion mode instead of DBSF."""

    collection_name = 'test_integration_rrf'
    user_config: ClassVar[dict[str, Any]]
    opt_config: ClassVar[dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        _setup_qdrant_env()
        cls.user_config = {'collection_name': cls.collection_name}
        cls.opt_config = make_opt_config(search_mode='dense_sparse')
        cls.opt_config['fusion_mode'] = 'RRF'
        _cleanup_collection(cls.collection_name)
        index(SAMPLE_DOCUMENTS, cls.user_config, cls.opt_config)

    @classmethod
    def tearDownClass(cls) -> None:
        _cleanup_collection(cls.collection_name)

    def test_rrf_search_finds_results(self) -> None:
        results = search('What are rabbits?', self.user_config, self.opt_config)
        self.assertGreater(len(results.points), 0)

    def test_rrf_search_relevance(self) -> None:
        results = search('vector similarity search engine', self.user_config, self.opt_config)
        self.assertGreater(len(results.points), 0)
        payload = results.points[0].payload
        assert payload is not None
        self.assertIn('Qdrant', payload['content'])

    def test_rrf_returns_scores(self) -> None:
        results = search('programming language', self.user_config, self.opt_config)
        for point in results.points:
            self.assertIsNotNone(point.score)


@unittest.skipUnless(
    os.environ.get('RUN_INTEGRATION_TESTS'),
    'Requires RUN_INTEGRATION_TESTS=1 and a running Qdrant server',
)
class QdrantRerankingFlagRerankerTestCase(unittest.TestCase):
    """Test search with FlagReranker reranking."""

    collection_name = 'test_integration_rerank_flag'
    user_config: ClassVar[dict[str, Any]]
    opt_config: ClassVar[dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        _setup_qdrant_env()
        cls.user_config = {'collection_name': cls.collection_name}
        cls.opt_config = make_opt_config(search_mode='dense')
        cls.opt_config['reranking'] = 'True'
        cls.opt_config['reranking_mode'] = 'reranking_with_flagreranker'
        cls.opt_config['top_k_reranker'] = 3
        _cleanup_collection(cls.collection_name)
        index(SAMPLE_DOCUMENTS, cls.user_config, cls.opt_config)

    @classmethod
    def tearDownClass(cls) -> None:
        _cleanup_collection(cls.collection_name)

    def test_reranking_returns_results(self) -> None:
        points = _collect_query_points('What are rabbits?', self.user_config, self.opt_config)
        self.assertGreater(len(points), 0)

    def test_reranking_adds_score_to_payload(self) -> None:
        points = _collect_query_points('small mammals herbivore', self.user_config, self.opt_config)
        self.assertGreater(len(points), 0)
        for point in points:
            assert point.payload is not None
            self.assertIn('reranking_score', point.payload)

    def test_reranking_relevance(self) -> None:
        points = _collect_query_points('vector similarity search engine', self.user_config, self.opt_config)
        self.assertGreater(len(points), 0)
        payload = points[0].payload
        assert payload is not None
        self.assertIn('Qdrant', payload['content'])

    def test_reranking_respects_top_k(self) -> None:
        points = _collect_query_points('programming', self.user_config, self.opt_config)
        self.assertLessEqual(len(points), self.opt_config['top_k_reranker'])


@unittest.skipUnless(
    os.environ.get('RUN_INTEGRATION_TESTS'),
    'Requires RUN_INTEGRATION_TESTS=1 and a running Qdrant server',
)
class QdrantRerankingCrossEncoderTestCase(unittest.TestCase):
    """Test search with sentence-transformers CrossEncoder reranking."""

    collection_name = 'test_integration_rerank_crossencoder'
    user_config: ClassVar[dict[str, Any]]
    opt_config: ClassVar[dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        _setup_qdrant_env()
        cls.user_config = {'collection_name': cls.collection_name}
        cls.opt_config = make_opt_config(search_mode='dense')
        cls.opt_config['reranking'] = 'True'
        cls.opt_config['reranking_mode'] = 'reranking_with_sentence_transformers'
        cls.opt_config['top_k_reranker'] = 3
        _cleanup_collection(cls.collection_name)
        index(SAMPLE_DOCUMENTS, cls.user_config, cls.opt_config)

    @classmethod
    def tearDownClass(cls) -> None:
        _cleanup_collection(cls.collection_name)

    def test_reranking_returns_results(self) -> None:
        points = _collect_query_points('What are rabbits?', self.user_config, self.opt_config)
        self.assertGreater(len(points), 0)

    def test_reranking_adds_score_to_payload(self) -> None:
        points = _collect_query_points('small mammals herbivore', self.user_config, self.opt_config)
        self.assertGreater(len(points), 0)
        for point in points:
            assert point.payload is not None
            self.assertIn('reranking_score', point.payload)

    def test_reranking_relevance(self) -> None:
        points = _collect_query_points('vector similarity search engine', self.user_config, self.opt_config)
        self.assertGreater(len(points), 0)
        payload = points[0].payload
        assert payload is not None
        self.assertIn('Qdrant', payload['content'])

    def test_reranking_respects_top_k(self) -> None:
        points = _collect_query_points('programming', self.user_config, self.opt_config)
        self.assertLessEqual(len(points), self.opt_config['top_k_reranker'])


@unittest.skipUnless(
    os.environ.get('RUN_INTEGRATION_TESTS'),
    'Requires RUN_INTEGRATION_TESTS=1 and a running Qdrant server',
)
class QdrantStoreOperationsTestCase(unittest.TestCase):
    """Test get_documents, delete_documents, update_documents against a real Qdrant server."""

    collection_name = 'test_integration_store'
    user_config: ClassVar[dict[str, Any]]
    opt_config: ClassVar[dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        _setup_qdrant_env()
        cls.user_config = {'collection_name': cls.collection_name}
        cls.opt_config = make_opt_config(search_mode='dense')

    def setUp(self) -> None:
        _cleanup_collection(self.collection_name)
        index(SAMPLE_DOCUMENTS, self.user_config, self.opt_config)

    def tearDown(self) -> None:
        _cleanup_collection(self.collection_name)

    def test_get_documents_returns_all_sources(self) -> None:
        docs = get_documents('test_loader', self.user_config, self.opt_config)
        expected_sources = {d.metadata['source'] for d in SAMPLE_DOCUMENTS}
        self.assertEqual(set(docs.keys()), expected_sources)

    def test_get_documents_returns_correct_hashes(self) -> None:
        docs = get_documents('test_loader', self.user_config, self.opt_config)
        for sample_doc in SAMPLE_DOCUMENTS:
            source = sample_doc.metadata['source']
            self.assertEqual(docs[source], sample_doc.metadata['content_hash'])

    def test_get_documents_unknown_loader_returns_empty(self) -> None:
        docs = get_documents('nonexistent_loader', self.user_config, self.opt_config)
        self.assertEqual(docs, {})

    def test_delete_documents_removes_specific_source(self) -> None:
        delete_documents('test_loader', ['test/rabbits.txt'], self.user_config, self.opt_config)
        docs = get_documents('test_loader', self.user_config, self.opt_config)
        self.assertNotIn('test/rabbits.txt', docs)
        self.assertIn('test/python.txt', docs)
        self.assertIn('test/qdrant.txt', docs)

    def test_delete_documents_multiple_sources(self) -> None:
        delete_documents('test_loader', ['test/rabbits.txt', 'test/python.txt'], self.user_config, self.opt_config)
        docs = get_documents('test_loader', self.user_config, self.opt_config)
        self.assertEqual(set(docs.keys()), {'test/qdrant.txt'})

    def test_update_documents_changes_content(self) -> None:
        updated_doc = Document(
            page_content='Rabbits are now known to be secretly running the internet infrastructure worldwide.',
            metadata={
                'source': 'test/rabbits.txt', 'content_hash': 'updated_hash_999',
                'loader_id': 'test_loader', 'title': 'Rabbits Updated', 'summary': 'Rabbits run the internet',
            },
        )
        update_documents('test_loader', [updated_doc], self.user_config, self.opt_config)

        docs = get_documents('test_loader', self.user_config, self.opt_config)
        self.assertEqual(docs['test/rabbits.txt'], 'updated_hash_999')

        results = search('rabbits internet infrastructure', self.user_config, self.opt_config)
        self.assertGreater(len(results.points), 0)
        payload = results.points[0].payload
        assert payload is not None
        self.assertIn('internet infrastructure', payload['content'])

    def test_update_documents_does_not_affect_other_sources(self) -> None:
        updated_doc = Document(
            page_content='Python has been completely rewritten in Rust.',
            metadata={
                'source': 'test/python.txt', 'content_hash': 'new_python_hash',
                'loader_id': 'test_loader', 'title': 'Python', 'summary': 'Python in Rust',
            },
        )
        update_documents('test_loader', [updated_doc], self.user_config, self.opt_config)

        docs = get_documents('test_loader', self.user_config, self.opt_config)
        self.assertEqual(docs['test/rabbits.txt'], 'abc123')
        self.assertEqual(docs['test/qdrant.txt'], 'ghi789')
        self.assertEqual(docs['test/python.txt'], 'new_python_hash')


@unittest.skipUnless(
    os.environ.get('RUN_INTEGRATION_TESTS'),
    'Requires RUN_INTEGRATION_TESTS=1 and a running Qdrant server',
)
class QdrantDeduplicationTestCase(unittest.TestCase):
    """Verify that ingesting the same document multiple times does not create duplicates."""

    collection_name = 'test_integration_dedup'
    user_config: ClassVar[dict[str, Any]]
    opt_config: ClassVar[dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        _setup_qdrant_env()
        cls.user_config = {'collection_name': cls.collection_name}
        cls.opt_config = make_opt_config(search_mode='dense')

    def setUp(self) -> None:
        _cleanup_collection(self.collection_name)

    def tearDown(self) -> None:
        _cleanup_collection(self.collection_name)

    def test_double_ingest_no_duplicates(self) -> None:
        index(SAMPLE_DOCUMENTS, self.user_config, self.opt_config)
        index(SAMPLE_DOCUMENTS, self.user_config, self.opt_config)

        info = Qdrant.client.get_collection(self.collection_name)
        self.assertEqual(info.points_count, len(SAMPLE_DOCUMENTS))

    def test_triple_ingest_no_duplicates(self) -> None:
        for _ in range(3):
            index(SAMPLE_DOCUMENTS, self.user_config, self.opt_config)

        info = Qdrant.client.get_collection(self.collection_name)
        self.assertEqual(info.points_count, len(SAMPLE_DOCUMENTS))

    def test_partial_reingest_no_duplicates(self) -> None:
        index(SAMPLE_DOCUMENTS, self.user_config, self.opt_config)
        index([SAMPLE_DOCUMENTS[0]], self.user_config, self.opt_config)

        info = Qdrant.client.get_collection(self.collection_name)
        self.assertEqual(info.points_count, len(SAMPLE_DOCUMENTS))
