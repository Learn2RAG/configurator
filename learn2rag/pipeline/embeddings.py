from functools import lru_cache
from FlagEmbedding import BGEM3FlagModel # type: ignore[import-untyped]
from sentence_transformers import SentenceTransformer
from typing import List, Any, Literal
import numpy as np
import warnings


@lru_cache(maxsize=4)
def _get_bge_m3_model() -> BGEM3FlagModel:
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)


@lru_cache(maxsize=4)
def _get_sentence_transformer_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def create_embeddings(
    input_sample: List[str],
    model_name: str = "BAAI/bge-m3",
    embedding_mode: str = "dense",
) -> dict[Literal['dense_vecs', 'lexical_weights', 'colbert_vecs'], np.ndarray[Any, Any] | list[dict[str, float]] | list[np.ndarray[Any, Any]]] | np.ndarray[Any, Any]:
    if model_name == "BAAI/bge-m3":
        model = _get_bge_m3_model()

        if embedding_mode == "dense":
            return model.encode( # type: ignore[no-any-return]
                input_sample,
                batch_size=512,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        elif embedding_mode == "sparse":
            return model.encode( # type: ignore[no-any-return]
                input_sample,
                batch_size=512,
                return_dense=False,
                return_sparse=True,
                return_colbert_vecs=False,
            )
        elif embedding_mode == "dense_sparse":
            return model.encode( # type: ignore[no-any-return]
                input_sample,
                batch_size=512,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
        elif embedding_mode == "colbert":
            return model.encode( # type: ignore[no-any-return]
                input_sample,
                batch_size=32,
                return_dense=False,
                return_sparse=False,
                return_colbert_vecs=True,
            )
        elif embedding_mode == "dense_sparse_colbert":
            return model.encode( # type: ignore[no-any-return]
                input_sample,
                batch_size=8,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True,
            )
        else:
            warnings.warn(
                "Embedding mode unknown or not provided. Using dense embeddings"
            )
            return model.encode( # type: ignore[no-any-return]
                input_sample,
                batch_size=512,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )

    elif model_name == "sentence-transformers/all-mpnet-base-v2":
        model = _get_sentence_transformer_model("sentence-transformers/all-mpnet-base-v2")
        return model.encode(input_sample)  # type: ignore[return-value]

    else:
        warnings.warn("Embedding model unknown or not provided. Using dense embeddings of default model: BAAI/bge-m3")
        model = _get_bge_m3_model()
        return model.encode( # type: ignore[no-any-return]
            input_sample,
            batch_size=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )