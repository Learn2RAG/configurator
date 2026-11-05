import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from qdrant_client import models
from qdrant_client.http.models import ScoredPoint

from learn2rag.pipeline import search as pipeline_search
from learn2rag.pipeline.operators import search as search_operator_module
from learn2rag.pipeline.operators.search import SearchOperator
from learn2rag.pipeline.prov import Prov


def test_default_optimization_config_uses_dense_sparse() -> None:
    config_path = Path(__file__).resolve().parents[1] / "opt_config.json"

    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["search_mode"] == "dense_sparse"
    assert config["embedding_model"] == "BAAI/bge-m3"
    assert config["fusion_mode"] == "DBSF"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_explicit_config_search_reuses_collection_and_authorization(
    monkeypatch: Any,
) -> None:
    denied = ScoredPoint(id=1, version=0, score=0.9, payload={})
    allowed = ScoredPoint(id=2, version=0, score=0.7, payload={})
    request_user_config = {"collection_name": "hybrid-demo"}
    request_opt_config = {"search_mode": "dense_sparse", "top_k": 5}
    captured: dict[str, Any] = {}

    def collect(
        question: str,
        user_config: dict[str, Any],
        opt_config: dict[str, Any],
    ) -> list[Any]:
        captured.update(
            question=question,
            user_config=user_config,
            opt_config=opt_config,
        )
        return [denied, allowed]

    async def authorize(user: str, response: Any) -> list[Any]:
        captured["user"] = user
        captured["authorization_points"] = response.points
        return [allowed]

    monkeypatch.setattr(pipeline_search, "_collect_query_points", collect)
    monkeypatch.setattr(pipeline_search, "filter_authorized", authorize)

    result = await pipeline_search.search_authorized_with_config(
        "Where is Berlin?",
        "ragdemo",
        request_user_config,
        request_opt_config,
    )

    assert result == [allowed]
    assert captured == {
        "question": "Where is Berlin?",
        "user_config": request_user_config,
        "opt_config": request_opt_config,
        "user": "ragdemo",
        "authorization_points": [denied, allowed],
    }


@pytest.mark.anyio
async def test_existing_search_authorized_uses_global_configuration(
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}

    async def explicit(
        question: str,
        user: str,
        request_user_config: dict[str, Any],
        request_opt_config: dict[str, Any],
    ) -> list[Any]:
        captured.update(
            question=question,
            user=user,
            user_config=request_user_config,
            opt_config=request_opt_config,
        )
        return []

    monkeypatch.setattr(pipeline_search, "search_authorized_with_config", explicit)
    global_user_config = getattr(pipeline_search, "user_config")
    global_opt_config = getattr(pipeline_search, "opt_config")

    assert await pipeline_search.search_authorized("question", "user") == []
    assert captured["question"] == "question"
    assert captured["user"] == "user"
    assert captured["user_config"] is global_user_config
    assert captured["opt_config"] is global_opt_config


@pytest.mark.anyio
async def test_search_operator_default_and_explicit_paths_remain_separate(
    monkeypatch: Any,
) -> None:
    default_point = SimpleNamespace(id="default")
    explicit_point = SimpleNamespace(id="explicit")
    calls: list[tuple[Any, ...]] = []

    async def default(question: str, user: str) -> list[Any]:
        calls.append(("default", question, user))
        return [default_point]

    async def explicit(
        question: str,
        user: str,
        request_user_config: dict[str, Any],
        request_opt_config: dict[str, Any],
    ) -> list[Any]:
        calls.append(
            (
                "explicit",
                question,
                user,
                request_user_config,
                request_opt_config,
            )
        )
        return [explicit_point]

    monkeypatch.setattr(search_operator_module, "search_authorized", default)
    monkeypatch.setattr(
        search_operator_module,
        "search_authorized_with_config",
        explicit,
    )
    operator = SearchOperator()

    default_result = await operator.run(
        {"question": "default question", "user": "default user"},
        Prov(),
    )
    user_config = {"collection_name": "hybrid-demo"}
    opt_config = {"search_mode": "dense_sparse"}
    explicit_result = await operator.run_with_config(
        {"question": "hybrid question", "user": "ragdemo"},
        user_config=user_config,
        opt_config=opt_config,
    )

    assert default_result == {"documents": [default_point]}
    assert explicit_result == {"documents": [explicit_point]}
    assert calls == [
        ("default", "default question", "default user"),
        ("explicit", "hybrid question", "ragdemo", user_config, opt_config),
    ]


def test_dense_sparse_search_uses_native_qdrant_fusion_once(monkeypatch: Any) -> None:
    client = SimpleNamespace(query_points=MagicMock(return_value=SimpleNamespace(points=[])))
    embedding_calls: list[tuple[list[str], str, str]] = []

    class FakeQdrant:
        def __init__(self, collection_name: str, opt_config: dict[str, Any]) -> None:
            assert collection_name == "hybrid-demo"
            assert opt_config["search_mode"] == "dense_sparse"
            self.client = client

    def embeddings(
        inputs: list[str], model_name: str, embedding_mode: str
    ) -> dict[str, Any]:
        embedding_calls.append((inputs, model_name, embedding_mode))
        return {
            "dense_vecs": [[0.1, 0.2]],
            "lexical_weights": [{"7": 0.8, "11": 0.4}],
            "colbert_vecs": [],
        }

    monkeypatch.setattr(pipeline_search, "Qdrant", FakeQdrant)
    monkeypatch.setattr(pipeline_search, "create_embeddings", embeddings)
    config = {
        "search_mode": "dense_sparse",
        "embedding_model": "BAAI/bge-m3",
        "fusion_mode": "DBSF",
        "prefetch_limit_dense": 23,
        "prefetch_limit_sparse": 19,
        "top_k": 5,
        "vector_size": {"BAAI/bge-m3": 1024},
        "query_mode": "single",
        "multi_search": [],
    }

    pipeline_search.search(
        "Berlin question",
        {"collection_name": "hybrid-demo"},
        config,
    )

    assert embedding_calls == [
        (["Berlin question"], "BAAI/bge-m3", "dense_sparse")
    ]
    client.query_points.assert_called_once()
    call = client.query_points.call_args.kwargs
    assert call["collection_name"] == "hybrid-demo"
    assert call["limit"] == 5
    assert isinstance(call["query"], models.FusionQuery)
    assert call["query"].fusion == models.Fusion.DBSF
    assert [prefetch.using for prefetch in call["prefetch"]] == ["sparse", "dense"]
    assert [prefetch.limit for prefetch in call["prefetch"]] == [19, 23]
    assert isinstance(call["prefetch"][0].query, models.SparseVector)
    assert call["prefetch"][1].query == [0.1, 0.2]
