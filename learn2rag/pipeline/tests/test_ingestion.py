import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

from learn2rag.pipeline import ingestion
from learn2rag.pipeline.embeddings import create_embeddings
from learn2rag.pipeline.qdrant import Qdrant


def test_payload_builds_expected_metadata() -> None:
    sample = {
        'page_content': 'hello',
        'chunk_hash': 'hash123',
        'metadata': {
            'source': '/tmp/doc',
            'content_hash': 'contenthash',
            'loader_id': 'loader1',
            'title': 'Test Title',
            'uri': 'https://example.com',
            'document_id': 'doc123',
        },
    }

    payload = ingestion.payload(sample)

    assert payload['content'] == 'hello'
    assert payload['path'] == '/tmp/doc'
    assert payload['content_hash'] == 'contenthash'
    assert payload['chunk_hash'] == 'hash123'
    assert payload['title'] == 'Test Title'
    assert payload['uri'] == 'https://example.com'
    assert payload['loader_id'] == 'loader1'
    assert payload['document_id'] == 'doc123'


def test_point_exists_returns_true_when_match_found() -> None:
    fake_qdrant = MagicMock()
    fake_qdrant.client.scroll.return_value = ([{'id': 'x'}], None)

    exists = ingestion.point_exists(
        fake_qdrant,
        collection_name='test-collection',
        loader_id='loader1',
        path='/tmp/doc',
        content_hash='contenthash',
        chunk_hash='chunkhash',
    )

    assert exists is True
    fake_qdrant.client.scroll.assert_called_once()


def test_point_exists_returns_false_when_no_match() -> None:
    fake_qdrant = MagicMock()
    fake_qdrant.client.scroll.return_value = ([], None)

    exists = ingestion.point_exists(
        fake_qdrant,
        collection_name='test-collection',
        loader_id='loader1',
        path='/tmp/doc',
        content_hash='contenthash',
        chunk_hash='chunkhash',
    )

    assert exists is False
    fake_qdrant.client.scroll.assert_called_once()


def test_get_chunks_metadata_with_missing_item() -> None:
    chunks = [
        Document(page_content='chunk1', metadata={'title': 'Title1'}),
        Document(page_content='chunk2', metadata={}),  # missing title
        Document(page_content='chunk3', metadata={'title': 'Title3'}),
    ]

    metadata_list = list(ingestion.get_chunks_metadata(chunks, 'title'))

    assert metadata_list == ['Title1', '', 'Title3']


def test_index_inserts_chunks_into_qdrant() -> None:
    documents = [
        Document(page_content='Hello world', metadata={'source': '/tmp/doc1', 'content_hash': 'hash1', 'loader_id': 'loader1'}),
        Document(page_content='Second chunk', metadata={'source': '/tmp/doc2', 'content_hash': 'hash2', 'loader_id': 'loader1'}),
    ]

    fake_qdrant = MagicMock()
    fake_qdrant.client.scroll.return_value = ([], None)
    fake_qdrant.client.upsert = MagicMock()

    fake_splitter = MagicMock()
    fake_splitter.return_value.split_documents.return_value = documents

    fake_embeddings = {
        'dense_vecs': np.array([[0.1, 0.2], [0.3, 0.4]]),
    }

    with patch('learn2rag.pipeline.ingestion.json_loader.json_loader', return_value=documents), \
         patch('learn2rag.pipeline.ingestion.RecursiveCharacterTextSplitter', fake_splitter), \
         patch('learn2rag.pipeline.ingestion.create_embeddings', return_value=fake_embeddings), \
         patch('learn2rag.pipeline.ingestion.Qdrant', return_value=fake_qdrant):
        user_config = {
            'collection_name': 'test-collection',
            'imported_documents_file_path': '/tmp/doesnotmatter.json',
        }
        opt_config = {
            'chunk_size': 1000,
            'chunk_overlap': 0,
            'search_mode': 'dense',
            'query_mode': 'default',
            'multi_search': [],
            'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
            'vector_size': {'sentence-transformers/all-mpnet-base-v2': 2},
        }

        ingestion.index(user_config, opt_config)

    assert fake_qdrant.client.upsert.call_count == 2
    
    # Verify first upsert call
    all_upsert_calls = fake_qdrant.client.upsert.call_args_list
    first_call_kwargs = all_upsert_calls[0].kwargs
    assert first_call_kwargs['collection_name'] == 'test-collection'
    assert len(first_call_kwargs['points']) == 1
    assert first_call_kwargs['points'][0].payload['content'] == 'Hello world'
    assert first_call_kwargs['points'][0].payload['path'] == '/tmp/doc1'
    
    # Verify second upsert call
    second_call_kwargs = all_upsert_calls[1].kwargs
    assert second_call_kwargs['collection_name'] == 'test-collection'
    assert len(second_call_kwargs['points']) == 1
    assert second_call_kwargs['points'][0].payload['content'] == 'Second chunk'
    assert second_call_kwargs['points'][0].payload['path'] == '/tmp/doc2'


def test_index_with_dense_sparse_mode() -> None:
    documents = [
        Document(page_content='Hello world', metadata={'source': '/tmp/doc1', 'content_hash': 'hash1', 'loader_id': 'loader1'}),
    ]

    fake_qdrant = MagicMock()
    fake_qdrant.client.scroll.return_value = ([], None)
    fake_qdrant.client.upsert = MagicMock()

    fake_splitter = MagicMock()
    fake_splitter.return_value.split_documents.return_value = documents

    fake_embeddings = {
        'dense_vecs': [np.array([0.1, 0.2])],
        'lexical_weights': [{1: 0.5, 2: 0.3}],
    }

    with patch('learn2rag.pipeline.ingestion.json_loader.json_loader', return_value=documents), \
         patch('learn2rag.pipeline.ingestion.RecursiveCharacterTextSplitter', fake_splitter), \
         patch('learn2rag.pipeline.ingestion.create_embeddings', return_value=fake_embeddings), \
         patch('learn2rag.pipeline.ingestion.Qdrant', return_value=fake_qdrant):
        user_config = {
            'collection_name': 'test-collection',
            'imported_documents_file_path': '/tmp/doesnotmatter.json',
        }
        opt_config = {
            'chunk_size': 1000,
            'chunk_overlap': 0,
            'search_mode': 'dense_sparse',
            'query_mode': 'default',
            'multi_search': [],
            'embedding_model': 'BAAI/bge-m3',
            'vector_size': {'BAAI/bge-m3': 2},
        }

        ingestion.index(user_config, opt_config)

    assert fake_qdrant.client.upsert.call_count == 1
    upsert_call = fake_qdrant.client.upsert.call_args
    point = upsert_call.kwargs['points'][0]
    assert 'dense' in point.vector
    assert 'sparse' in point.vector
    assert point.vector['sparse'].indices == [1, 2]
    assert point.vector['sparse'].values == [0.5, 0.3]


def test_index_with_multi_mode() -> None:
    documents = [
        Document(page_content='Hello world', metadata={'source': '/tmp/doc1', 'content_hash': 'hash1', 'loader_id': 'loader1', 'title': 'Title1'}),
    ]

    fake_qdrant = MagicMock()
    fake_qdrant.client.scroll.return_value = ([], None)
    fake_qdrant.client.upsert = MagicMock()

    fake_splitter = MagicMock()
    fake_splitter.return_value.split_documents.return_value = documents

    fake_embeddings = {
        'dense_vecs': [np.array([0.1, 0.2])],
    }

    fake_metadata_embeddings = {
        'dense_vecs': np.array([[0.3, 0.4]]),
    }

    def fake_create_embeddings(content: list[str], model: str, mode: str | None = None) -> Any:
        if content == ['Title1']:
            return fake_metadata_embeddings
        return fake_embeddings

    with patch('learn2rag.pipeline.ingestion.json_loader.json_loader', return_value=documents), \
         patch('learn2rag.pipeline.ingestion.RecursiveCharacterTextSplitter', fake_splitter), \
         patch('learn2rag.pipeline.ingestion.create_embeddings', side_effect=fake_create_embeddings), \
         patch('learn2rag.pipeline.ingestion.Qdrant', return_value=fake_qdrant):
        user_config = {
            'collection_name': 'test-collection',
            'imported_documents_file_path': '/tmp/doesnotmatter.json',
        }
        opt_config = {
            'chunk_size': 1000,
            'chunk_overlap': 0,
            'search_mode': 'dense',
            'query_mode': 'multi',
            'multi_search': ['title'],
            'embedding_model': 'BAAI/bge-m3',
            'vector_size': {'BAAI/bge-m3': 2},
        }

        ingestion.index(user_config, opt_config)

    assert fake_qdrant.client.upsert.call_count == 1
    upsert_call = fake_qdrant.client.upsert.call_args
    point = upsert_call.kwargs['points'][0]
    assert 'multi' in point.vector
    # Multi vector should be concatenated: content + title
    expected_multi = np.concatenate([np.array([0.1, 0.2]), np.array([0.3, 0.4])])
    assert np.allclose(point.vector['multi'], expected_multi)


def test_text_splitter_splits_long_random_content() -> None:
    import random
    import string

    # Generate random text longer than chunk_size
    random_text = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=500))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    chunks = text_splitter.split_text(random_text)

    # The text (500 characters) should be split into multiple chunks
    assert len(chunks) > 1

    # All chunks should not exceed chunk_size
    for chunk in chunks:
        assert len(chunk) <= 100


def test_index_ingests_multiple_chunks_with_same_content_hash() -> None:
    # Regression test: Two chunks with same content_hash but different chunk_hash
    # should both be ingested (bug in the past: only one chunk_hash per content_hash possible)
    documents = [
        Document(page_content='Part 1 of content', metadata={'source': '/tmp/doc', 'content_hash': 'hash_same', 'loader_id': 'loader1'}),
        Document(page_content='Part 2 of content', metadata={'source': '/tmp/doc', 'content_hash': 'hash_same', 'loader_id': 'loader1'}),
    ]

    fake_qdrant = MagicMock()
    fake_qdrant.client.upsert = MagicMock()

    # Mock scroll: Both chunks are new (not existing)
    fake_qdrant.client.scroll.return_value = ([], None)

    fake_splitter = MagicMock()
    fake_splitter.return_value.split_documents.return_value = documents

    fake_embeddings = {
        'dense_vecs': np.array([[0.1, 0.2], [0.3, 0.4]]),
    }

    with patch('learn2rag.pipeline.ingestion.json_loader.json_loader', return_value=documents), \
         patch('learn2rag.pipeline.ingestion.RecursiveCharacterTextSplitter', fake_splitter), \
         patch('learn2rag.pipeline.ingestion.create_embeddings', return_value=fake_embeddings), \
         patch('learn2rag.pipeline.ingestion.Qdrant', return_value=fake_qdrant):
        user_config = {
            'collection_name': 'test-collection',
            'imported_documents_file_path': '/tmp/doesnotmatter.json',
        }
        opt_config = {
            'chunk_size': 1000,
            'chunk_overlap': 0,
            'search_mode': 'dense',
            'query_mode': 'default',
            'multi_search': [],
            'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
            'vector_size': {'sentence-transformers/all-mpnet-base-v2': 2},
        }

        ingestion.index(user_config, opt_config)

    # Both chunks should be ingested
    assert fake_qdrant.client.upsert.call_count == 2
    
    # Verify that both chunks with different content were ingested
    all_upsert_calls = fake_qdrant.client.upsert.call_args_list
    first_payload = all_upsert_calls[0].kwargs['points'][0].payload
    second_payload = all_upsert_calls[1].kwargs['points'][0].payload
    
    assert first_payload['content'] == 'Part 1 of content'
    assert second_payload['content'] == 'Part 2 of content'
    
    # Both should have the same content_hash
    assert first_payload['content_hash'] == 'hash_same'
    assert second_payload['content_hash'] == 'hash_same'
    
    # But different chunk_hashes
    assert first_payload['chunk_hash'] != second_payload['chunk_hash']


def test_index_stores_vectors_with_correct_dimensions() -> None:
    # Test: Vectors should have the correct dimension (here 2)
    documents = [
        Document(page_content='Content 1', metadata={'source': '/tmp/doc1', 'content_hash': 'hash1', 'loader_id': 'loader1'}),
        Document(page_content='Content 2', metadata={'source': '/tmp/doc2', 'content_hash': 'hash2', 'loader_id': 'loader1'}),
    ]

    fake_qdrant = MagicMock()
    fake_qdrant.client.scroll.return_value = ([], None)
    fake_qdrant.client.upsert = MagicMock()

    fake_splitter = MagicMock()
    fake_splitter.return_value.split_documents.return_value = documents

    # Embeddings with dimension 2 (matching 'vector_size': 2 in opt_config)
    fake_embeddings = {
        'dense_vecs': np.array([[0.1, 0.2], [0.3, 0.4]]),
    }

    with patch('learn2rag.pipeline.ingestion.json_loader.json_loader', return_value=documents), \
         patch('learn2rag.pipeline.ingestion.RecursiveCharacterTextSplitter', fake_splitter), \
         patch('learn2rag.pipeline.ingestion.create_embeddings', return_value=fake_embeddings), \
         patch('learn2rag.pipeline.ingestion.Qdrant', return_value=fake_qdrant):
        user_config = {
            'collection_name': 'test-collection',
            'imported_documents_file_path': '/tmp/doesnotmatter.json',
        }
        opt_config = {
            'chunk_size': 1000,
            'chunk_overlap': 0,
            'search_mode': 'dense',
            'query_mode': 'default',
            'multi_search': [],
            'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
            'vector_size': {'sentence-transformers/all-mpnet-base-v2': 2},
        }

        ingestion.index(user_config, opt_config)

    # Verify that the vectors have the correct dimension
    all_upsert_calls = fake_qdrant.client.upsert.call_args_list
    for call in all_upsert_calls:
        points = call.kwargs['points']
        for point in points:
            # The vector should be a numpy array with length 2
            dense_vector = point.vector['dense']
            assert len(dense_vector) == 2, f"Expected vector dimension 2, got {len(dense_vector)}"
            # Verify that the values match the mock values
            assert np.allclose(dense_vector, [0.1, 0.2]) or np.allclose(dense_vector, [0.3, 0.4])


TESTS_DIR = Path(__file__).resolve().parent
TEST_COMPOSE_FILE = TESTS_DIR / 'docker-compose.test.yml'
TEST_DOCUMENTS_FILE = TESTS_DIR / 'loaded_documents.json'
TEST_QDRANT_PORT = 6337


def _wait_for_qdrant(client: QdrantClient, timeout_seconds: int = 30) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            client.get_collections()
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError('Timed out waiting for test Qdrant instance to become ready')


@pytest.fixture(scope='module')
def qdrant_test_service() -> Iterator[QdrantClient]:
    original_client = Qdrant.client
    original_http_port = os.environ.get('QDRANT__SERVICE__HTTP_PORT')
    original_api_key = os.environ.get('QDRANT__SERVICE__API_KEY')

    os.environ['QDRANT__SERVICE__HTTP_PORT'] = str(TEST_QDRANT_PORT)
    os.environ['QDRANT__SERVICE__API_KEY'] = ''

    subprocess.run(
        ['docker', 'compose', '-f', str(TEST_COMPOSE_FILE), 'up', '-d', 'qdrant-test'],
        check=True,
        capture_output=True,
    )

    client = QdrantClient(host='localhost', port=TEST_QDRANT_PORT, api_key='', https=False)
    _wait_for_qdrant(client)
    Qdrant.client = client

    try:
        yield client
    finally:
        Qdrant.client = original_client
        if original_http_port is None:
            os.environ.pop('QDRANT__SERVICE__HTTP_PORT', None)
        else:
            os.environ['QDRANT__SERVICE__HTTP_PORT'] = original_http_port
        if original_api_key is None:
            os.environ.pop('QDRANT__SERVICE__API_KEY', None)
        else:
            os.environ['QDRANT__SERVICE__API_KEY'] = original_api_key
        subprocess.run(
            ['docker', 'compose', '-f', str(TEST_COMPOSE_FILE), 'down'],
            capture_output=True,
            check=False,
        )


@pytest.fixture(scope='module')
def seeded_dense_collection(qdrant_test_service: QdrantClient) -> Iterator[dict[str, Any]]:
    collection_name = f'test-integration-dense-{uuid4().hex[:8]}'
    user_config = {
        'collection_name': collection_name,
        'imported_documents_file_path': str(TEST_DOCUMENTS_FILE),
    }
    opt_config = {
        'chunk_size': 2000,
        'chunk_overlap': 200,
        'search_mode': 'dense',
        'query_mode': 'default',
        'multi_search': [],
        'embedding_model': 'BAAI/bge-m3',
        'vector_size': {'BAAI/bge-m3': 1024},
    }

    ingestion.index(user_config, opt_config)
    initial_count = qdrant_test_service.count(collection_name).count
    assert initial_count > 2, f'Expected more than 2 points after first ingestion, got {initial_count}'

    try:
        yield {
            'client': qdrant_test_service,
            'collection_name': collection_name,
            'user_config': user_config,
            'opt_config': opt_config,
            'initial_count': initial_count,
        }
    finally:
        try:
            qdrant_test_service.delete_collection(collection_name)
        except Exception:
            pass


@pytest.mark.integration
def test_integration_ingestion_with_docker_qdrant_and_deduplication(
    seeded_dense_collection: dict[str, Any],
) -> None:
    """Integration test that ingests documents twice and verifies deduplication."""
    client = seeded_dense_collection['client']
    collection_name = seeded_dense_collection['collection_name']
    user_config = seeded_dense_collection['user_config']
    opt_config = seeded_dense_collection['opt_config']
    initial_count = seeded_dense_collection['initial_count']

    assert isinstance(client, QdrantClient)
    assert isinstance(collection_name, str)
    assert isinstance(user_config, dict)
    assert isinstance(opt_config, dict)
    assert isinstance(initial_count, int)

    ingestion.index(user_config, opt_config)

    final_count = client.count(collection_name).count
    print(f'Points after first ingestion: {initial_count}')
    print(f'Points after second ingestion: {final_count}')

    assert final_count == initial_count, f'Expected {initial_count} points after deduplication, got {final_count}'


@pytest.mark.integration
def test_integration_dense_search_in_seeded_collection(
    seeded_dense_collection: dict[str, Any],
) -> None:
    """Integration test that runs a dense query against the shared seeded collection."""
    client = seeded_dense_collection['client']
    collection_name = seeded_dense_collection['collection_name']

    assert isinstance(client, QdrantClient)
    assert isinstance(collection_name, str)

    query = 'Was ist die Payload in qdrant?'
    dense_vectors = create_embeddings([query], 'BAAI/bge-m3', 'dense')['dense_vecs']
    assert isinstance(dense_vectors, np.ndarray)
    query_vector = dense_vectors[0].tolist()
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using='dense',
        limit=3,
    )

    assert results.points, 'Expected at least one dense search result'
    assert results.points[0].payload is not None, 'Expected payload in the top search result'
    assert 'content' in results.points[0].payload, 'Expected content field in the top search result payload'
    mentions_payload = False
    for point in results.points:
        payload = point.payload
        if payload is None or 'content' not in payload:
            continue
        if 'payload' in payload['content'].lower():
            mentions_payload = True
            break

    assert mentions_payload, 'Expected at least one result mentioning payload'
