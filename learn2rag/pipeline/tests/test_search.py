from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from learn2rag.pipeline import search


def test_search_dense_calls_qdrant_query_points() -> None:
    fake_client = MagicMock()
    fake_response = SimpleNamespace(points=['result'])
    fake_client.query_points.return_value = fake_response

    fake_qdrant = SimpleNamespace(client=fake_client)

    with patch('learn2rag.pipeline.search.Qdrant', return_value=fake_qdrant), \
         patch('learn2rag.pipeline.search.create_embeddings', return_value={'dense_vecs': np.array([[0.1, 0.2]])}):
        user_config = {'collection_name': 'test-collection'}
        opt_config = {
            'embedding_model': 'BAAI/bge-m3',
            'search_mode': 'dense',
            'fusion_mode': 'DBSF',
            'query_mode': 'default',
            'top_k': 3,
            'vector_size': {'BAAI/bge-m3': 2},
            'multi_search': [],
            'prefetch_limit_sparse': 1,
            'prefetch_limit_dense': 1,
            'prefetch_limit_colbert': 1,
        }

        results = search.search('hello world', user_config, opt_config)

    assert results is fake_response
    fake_client.query_points.assert_called_once()
    assert fake_client.query_points.call_args.kwargs['using'] == 'dense'
    assert fake_client.query_points.call_args.kwargs['limit'] == 3


def test_search_multi_builds_multi_vector_query() -> None:
    fake_client = MagicMock()
    fake_response = SimpleNamespace(points=[])
    fake_client.query_points.return_value = fake_response
    fake_qdrant = SimpleNamespace(client=fake_client)

    embeddings_iter = [
        {'dense_vecs': np.array([[0.1, 0.2]])},
        {'dense_vecs': np.array([[0.3, 0.4]])},
    ]

    def fake_create_embeddings(input_sample, model_name, embedding_mode=None):
        return embeddings_iter.pop(0)

    with patch('learn2rag.pipeline.search.Qdrant', return_value=fake_qdrant), \
         patch('learn2rag.pipeline.search.create_embeddings', side_effect=fake_create_embeddings):
        user_config = {'collection_name': 'test-collection'}
        opt_config = {
            'embedding_model': 'BAAI/bge-m3',
            'search_mode': 'dense',
            'fusion_mode': 'DBSF',
            'query_mode': 'multi',
            'top_k': 1,
            'vector_size': {'BAAI/bge-m3': 2},
            'multi_search': ['title'],
            'prefetch_limit_sparse': 1,
            'prefetch_limit_dense': 1,
            'prefetch_limit_colbert': 1,
        }

        results = search.search_multi({'content': 'hello', 'title': 'Greeting'}, user_config, opt_config)

    assert results is fake_response
    fake_client.query_points.assert_called_once()
    assert fake_client.query_points.call_args.kwargs['using'] == 'multi'
    assert fake_client.query_points.call_args.kwargs['limit'] == 1


# Neuer Test für dense_sparse Modus
def test_search_dense_sparse_calls_qdrant_with_prefetch() -> None:
    fake_client = MagicMock()
    fake_response = SimpleNamespace(points=['result'])
    fake_client.query_points.return_value = fake_response

    fake_qdrant = SimpleNamespace(client=fake_client)

    # Mock für create_embeddings mit lexical_weights als Liste von Dicts
    fake_embedding = {
        'dense_vecs': [np.array([0.1, 0.2])],
        'lexical_weights': [{1: 0.5, 2: 0.3}],
        'colbert_vecs': []
    }

    with patch('learn2rag.pipeline.search.Qdrant', return_value=fake_qdrant), \
         patch('learn2rag.pipeline.search.create_embeddings', return_value=fake_embedding):
        user_config = {'collection_name': 'test-collection'}
        opt_config = {
            'embedding_model': 'BAAI/bge-m3',
            'search_mode': 'dense_sparse',
            'fusion_mode': 'RRF',
            'query_mode': 'default',
            'top_k': 5,
            'vector_size': {'BAAI/bge-m3': 2},
            'multi_search': [],
            'prefetch_limit_sparse': 10,
            'prefetch_limit_dense': 10,
            'prefetch_limit_colbert': 1,
        }

        results = search.search('hello world', user_config, opt_config)

    assert results is fake_response
    fake_client.query_points.assert_called_once()
    call_kwargs = fake_client.query_points.call_args.kwargs
    assert 'prefetch' in call_kwargs
    assert len(call_kwargs['prefetch']) == 2  # sparse and dense prefetch
    from qdrant_client import models
    assert call_kwargs['query'].fusion == models.Fusion.RRF


# Neuer Test für reranking_with_flagreranker Modus
def test_search_reranking_with_flagreranker() -> None:
    fake_client = MagicMock()
    fake_response = SimpleNamespace(points=[
        SimpleNamespace(payload={'content': 'test content 1'}),
        SimpleNamespace(payload={'content': 'test content 2'}),
        SimpleNamespace(payload=None)  # Test für None payload
    ])
    fake_client.query_points.return_value = fake_response

    fake_qdrant = SimpleNamespace(client=fake_client)

    # Mock für create_embeddings mit allen nötigen Feldern
    fake_embedding = {
        'dense_vecs': [np.array([0.1, 0.2])],
        'lexical_weights': [{1: 0.5}],
        'colbert_vecs': [[0.1, 0.2]]
    }

    # Mock für FlagReranker
    with patch('learn2rag.pipeline.search.Qdrant', return_value=fake_qdrant), \
         patch('learn2rag.pipeline.search.create_embeddings', return_value=fake_embedding), \
         patch('learn2rag.pipeline.search.FlagReranker') as mock_reranker_class:
        
        mock_reranker = MagicMock()
        mock_reranker.compute_score.return_value = [0.8, 0.6]
        mock_reranker_class.return_value = mock_reranker
        
        user_config = {'collection_name': 'test-collection'}
        opt_config = {
            'embedding_model': 'BAAI/bge-m3',
            'search_mode': 'reranking_with_flagreranker',
            'fusion_mode': 'RRF',
            'query_mode': 'default',
            'top_k': 2,
            'vector_size': {'BAAI/bge-m3': 2},
            'multi_search': [],
            'prefetch_limit_sparse': 20,
            'prefetch_limit_dense': 20,
            'prefetch_limit_colbert': 1,
        }

        results = search.search('hello world', user_config, opt_config)

    assert len(results.points) == 2  # top_k = 2
    # Prüfe, ob Reranker aufgerufen wurde
    mock_reranker.compute_score.assert_called_once()
    # Prüfe, ob Punkte nach reranking_score sortiert sind
    assert results.points[0].payload['reranking_score'] == 0.8
    assert results.points[1].payload['reranking_score'] == 0.6


# Neuer Test für search_multi mit nicht unterstütztem Modell (sollte Exception werfen)
def test_search_multi_unsupported_embedding_model() -> None:
    fake_qdrant = SimpleNamespace(client=MagicMock())
    
    with patch('learn2rag.pipeline.search.Qdrant', return_value=fake_qdrant):
        user_config = {'collection_name': 'test-collection'}
        opt_config = {
            'embedding_model': 'unsupported-model',
            'search_mode': 'dense',
            'query_mode': 'multi',
            'top_k': 1,
            'multi_search': ['title'],
        }

        with pytest.raises(NotImplementedError, match="Embedding model 'unsupported-model' not supported"):
            search.search_multi({'content': 'hello', 'title': 'Greeting'}, user_config, opt_config)
