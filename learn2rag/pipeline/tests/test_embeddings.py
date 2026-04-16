from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from learn2rag.pipeline import embeddings


def test_create_embeddings_dense_sentence_transformers() -> None:
    input_sample = ['hello world']
    fake_array = np.array([[0.1, 0.2, 0.3]])
    fake_model = MagicMock()
    fake_model.encode.return_value = fake_array

    with patch('learn2rag.pipeline.embeddings.SentenceTransformer', return_value=fake_model):
        result = embeddings.create_embeddings(input_sample, 'sentence-transformers/all-mpnet-base-v2')

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 3)
    assert np.array_equal(result, fake_array)
    fake_model.encode.assert_called_once_with(input_sample)


def test_create_embeddings_dense_sparse_colbert_bge() -> None:
    input_sample = ['hello world']
    fake_result = {
        'dense_vecs': np.array([[0.1, 0.2]]),
        'lexical_weights': [{'1': 1.0}],
        'colbert_vecs': [np.array([0.3, 0.4])],
    }
    fake_model = MagicMock()
    fake_model.encode.return_value = fake_result

    with patch('learn2rag.pipeline.embeddings.BGEM3FlagModel', return_value=fake_model):
        result = embeddings.create_embeddings(input_sample, 'BAAI/bge-m3', embedding_mode='dense_sparse_colbert')

    assert isinstance(result, dict)
    assert 'dense_vecs' in result
    assert 'lexical_weights' in result
    assert 'colbert_vecs' in result
    assert result['dense_vecs'].shape == (1, 2)
    assert result['lexical_weights'][0] == {'1': 1.0}
    assert np.array_equal(result['colbert_vecs'][0], np.array([0.3, 0.4]))
    fake_model.encode.assert_called_once_with(
        input_sample,
        batch_size=512,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )
