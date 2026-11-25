import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
from qdrant_client import models

from ..search import _Embedding, _QueryEmbeddings, _build_prefetch_tree, _read_fusion_params, _retrieve_pool

SEARCH_LOGGER = "learn2rag.pipeline.search"
EXCLUSION = models.Filter(must_not=[models.HasIdCondition(has_id=[1, 2])])


def _embedding(seed: float) -> _Embedding:
    return _Embedding(
        dense=np.full(3, seed),
        sparse=models.SparseVector(indices=[1, 2], values=[seed, seed]),
        colbert=[np.full(3, seed)],
    )


def _embeddings(n_subqueries: int = 0, n_keywords: int = 0) -> _QueryEmbeddings:
    return _QueryEmbeddings(
        question=_embedding(1.0),
        subqueries=[_embedding(10.0 + i) for i in range(n_subqueries)],
        keywords=[_embedding(20.0 + i) for i in range(n_keywords)],
    )


def _opt_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "search_mode": "dense",
        "fusion_mode": "DBSF",
        "prefetch_limit_dense": 25,
        "prefetch_limit_sparse": 20,
        "prefetch_limit_colbert": 15,
        "top_k_subqueries": 5,
        "top_k_keywords": 4,
    }
    config.update(overrides)
    return config


def _tree(
    embeddings: _QueryEmbeddings, opt_config: dict[str, Any], exclusion: models.Filter | None = None,
) -> models.Prefetch:
    return _build_prefetch_tree(embeddings, opt_config, _read_fusion_params(opt_config), exclusion)


def _nodes(node: models.Prefetch) -> list[models.Prefetch]:
    """All nodes of the tree, the root first."""
    children = node.prefetch if isinstance(node.prefetch, list) else []
    return [node, *(descendant for child in children for descendant in _nodes(child))]


def _rrf(node: models.Prefetch) -> models.Rrf:
    assert isinstance(node.query, models.RrfQuery)
    return node.query.rrf


def _children(node: models.Prefetch) -> list[models.Prefetch]:
    assert isinstance(node.prefetch, list)
    return node.prefetch


class QuestionBranchTestCase(unittest.TestCase):
    """Branch A: the vector types of the question for every search_mode."""

    def test_dense_is_a_single_leaf(self) -> None:
        tree = _tree(_embeddings(), _opt_config(search_mode="dense"))

        self.assertIsNone(tree.prefetch)
        self.assertEqual(tree.using, "dense")
        self.assertEqual(tree.limit, 25)

    def test_sparse_is_a_single_leaf(self) -> None:
        tree = _tree(_embeddings(), _opt_config(search_mode="sparse"))

        self.assertIsNone(tree.prefetch)
        self.assertEqual(tree.using, "sparse")
        self.assertEqual(tree.limit, 20)

    def test_dense_sparse_fuses_both_leaves_with_fusion_mode(self) -> None:
        tree = _tree(_embeddings(), _opt_config(search_mode="dense_sparse"))

        self.assertEqual(tree.query, models.FusionQuery(fusion=models.Fusion.DBSF))
        self.assertEqual([(leaf.using, leaf.limit) for leaf in _children(tree)], [("dense", 25), ("sparse", 20)])
        self.assertEqual(tree.limit, 45)

    def test_dense_sparse_colbert_fuses_all_three_leaves(self) -> None:
        tree = _tree(
            _embeddings(), _opt_config(search_mode="dense_sparse_colbert", fusion_mode="RRF"))

        self.assertEqual(tree.query, models.FusionQuery(fusion=models.Fusion.RRF))
        self.assertEqual(
            [(leaf.using, leaf.limit) for leaf in _children(tree)],
            [("dense", 25), ("sparse", 20), ("colbert", 15)],
        )
        self.assertEqual(tree.limit, 60)

    def test_hybrid_mode_requires_valid_fusion_mode(self) -> None:
        for fusion_mode in ("rrf", "", None):
            with self.subTest(fusion_mode=fusion_mode):
                with self.assertRaises(ValueError):
                    _tree(_embeddings(), _opt_config(search_mode="dense_sparse", fusion_mode=fusion_mode))

    def test_single_vector_mode_does_not_need_fusion_mode(self) -> None:
        tree = _tree(_embeddings(), _opt_config(search_mode="dense", fusion_mode=None))

        self.assertEqual(tree.using, "dense")

    def test_leaves_use_the_vectors_of_the_question(self) -> None:
        question = _embedding(7.0)
        tree = _tree(
            _QueryEmbeddings(question, [], []), _opt_config(search_mode="dense_sparse"))

        dense_leaf, sparse_leaf = _children(tree)
        self.assertEqual(dense_leaf.query, [7.0, 7.0, 7.0])
        self.assertEqual(sparse_leaf.query, question.sparse)


class RewriteBranchTestCase(unittest.TestCase):
    """Branch B: RRF over subqueries (B1) and keywords (B2), root over A and B."""

    def test_without_rewrite_the_tree_is_branch_a(self) -> None:
        tree = _tree(_embeddings(), _opt_config())

        self.assertIsNone(tree.prefetch)
        self.assertEqual((tree.using, tree.limit), ("dense", 25))

    def test_root_has_branch_a_and_branch_b_in_this_order(self) -> None:
        tree = _tree(_embeddings(n_subqueries=2), _opt_config())

        branch_a, branch_b = _children(tree)
        self.assertEqual((branch_a.using, branch_a.limit), ("dense", 25))
        self.assertIsInstance(branch_b.query, models.RrfQuery)

    def test_subqueries_only(self) -> None:
        tree = _tree(_embeddings(n_subqueries=2), _opt_config())

        _, branch_b = _children(tree)
        self.assertEqual([(leaf.using, leaf.limit) for leaf in _children(branch_b)], [("dense", 5), ("dense", 5)])
        self.assertEqual(branch_b.limit, 10)
        self.assertEqual(tree.limit, 25 + 10)

    def test_keywords_only(self) -> None:
        tree = _tree(_embeddings(n_keywords=3), _opt_config())

        _, branch_b = _children(tree)
        self.assertEqual(
            [(leaf.using, leaf.limit) for leaf in _children(branch_b)], [("sparse", 4), ("sparse", 4), ("sparse", 4)])
        self.assertEqual(branch_b.limit, 12)
        self.assertEqual(tree.limit, 25 + 12)

    def test_keywords_use_the_sparse_vector_of_the_keyword(self) -> None:
        embeddings = _embeddings(n_keywords=2)
        tree = _tree(embeddings, _opt_config())

        _, branch_b = _children(tree)
        self.assertEqual([leaf.query for leaf in _children(branch_b)], [e.sparse for e in embeddings.keywords])

    def test_subqueries_and_keywords_form_two_groups(self) -> None:
        tree = _tree(_embeddings(n_subqueries=2, n_keywords=3), _opt_config())

        _, branch_b = _children(tree)
        group_subqueries, group_keywords = _children(branch_b)
        self.assertEqual([leaf.using for leaf in _children(group_subqueries)], ["dense", "dense"])
        self.assertEqual(group_subqueries.limit, 10)
        self.assertEqual([leaf.using for leaf in _children(group_keywords)], ["sparse"] * 3)
        self.assertEqual(group_keywords.limit, 12)
        self.assertIsInstance(branch_b.query, models.RrfQuery)
        self.assertEqual(branch_b.limit, 22)
        self.assertEqual(tree.limit, 25 + 22)

    def test_a_single_subquery_is_not_wrapped(self) -> None:
        tree = _tree(_embeddings(n_subqueries=1), _opt_config())

        _, branch_b = _children(tree)
        self.assertIsNone(branch_b.prefetch)
        self.assertEqual((branch_b.using, branch_b.limit), ("dense", 5))
        self.assertEqual(tree.limit, 30)

    def test_a_single_keyword_and_a_single_subquery_form_branch_b(self) -> None:
        tree = _tree(_embeddings(n_subqueries=1, n_keywords=1), _opt_config())

        _, branch_b = _children(tree)
        self.assertEqual([(leaf.using, leaf.limit) for leaf in _children(branch_b)], [("dense", 5), ("sparse", 4)])
        self.assertEqual(branch_b.limit, 9)

    def test_subquery_in_hybrid_mode_is_fused_and_limited_to_top_k_subqueries(self) -> None:
        tree = _tree(_embeddings(n_subqueries=1), _opt_config(search_mode="dense_sparse"))

        _, subquery = _children(tree)
        self.assertEqual(subquery.query, models.FusionQuery(fusion=models.Fusion.DBSF))
        self.assertEqual([(leaf.using, leaf.limit) for leaf in _children(subquery)], [("dense", 25), ("sparse", 20)])
        self.assertEqual(subquery.limit, 5)

    def test_subquery_in_single_vector_mode_uses_top_k_subqueries_as_leaf_limit(self) -> None:
        tree = _tree(_embeddings(n_subqueries=1), _opt_config(search_mode="sparse"))

        branch_a, subquery = _children(tree)
        self.assertEqual(branch_a.limit, 20)
        self.assertEqual((subquery.using, subquery.limit), ("sparse", 5))

    def test_limits_of_fusion_nodes_never_cut_candidates(self) -> None:
        tree = _tree(
            _embeddings(n_subqueries=3, n_keywords=2), _opt_config(search_mode="dense_sparse_colbert"))

        branch_a, branch_b = _children(tree)
        group_subqueries, group_keywords = _children(branch_b)
        self.assertEqual(branch_a.limit, 60)
        self.assertEqual(group_subqueries.limit, 3 * 5)
        self.assertEqual(group_keywords.limit, 2 * 4)
        self.assertEqual(branch_b.limit, 3 * 5 + 2 * 4)
        self.assertEqual(tree.limit, 60 + 3 * 5 + 2 * 4)


class FusionParamsTestCase(unittest.TestCase):
    def test_default_weights_are_one(self) -> None:
        tree = _tree(_embeddings(n_subqueries=2), _opt_config())

        self.assertEqual(_rrf(tree).weights, [1.0, 1.0])
        self.assertIsNone(_rrf(tree).k)

    def test_weights_are_given_as_original_then_rewrite(self) -> None:
        tree = _tree(
            _embeddings(n_subqueries=2), _opt_config(fusion_weights={"original": 2, "rewrite": 0.5}))

        self.assertEqual(_rrf(tree).weights, [2.0, 0.5])
        branch_a, _ = _children(tree)
        self.assertEqual(branch_a.using, "dense")

    def test_missing_weight_defaults_to_one(self) -> None:
        tree = _tree(_embeddings(n_subqueries=2), _opt_config(fusion_weights={"rewrite": 3}))

        self.assertEqual(_rrf(tree).weights, [1.0, 3.0])

    def test_rrf_k_is_set_on_every_rrf_node(self) -> None:
        tree = _tree(_embeddings(n_subqueries=2, n_keywords=2), _opt_config(rrf_k=30))

        _, branch_b = _children(tree)
        group_subqueries, group_keywords = _children(branch_b)
        for node in (tree, branch_b, group_subqueries, group_keywords):
            self.assertEqual(_rrf(node).k, 30)

    def test_rrf_k_does_not_change_the_fusion_mode_of_vectors(self) -> None:
        tree = _tree(_embeddings(), _opt_config(search_mode="dense_sparse", rrf_k=30))

        self.assertEqual(tree.query, models.FusionQuery(fusion=models.Fusion.DBSF))

    def test_invalid_weight_falls_back_to_one_with_a_warning(self) -> None:
        invalid: list[tuple[dict[str, Any], str, list[float]]] = [
            ({"original": 0}, "fusion_weights.original", [1.0, 1.0]),
            ({"original": 2, "rewrite": -1}, "fusion_weights.rewrite", [2.0, 1.0]),
            ({"original": True}, "fusion_weights.original", [1.0, 1.0]),
            ({"rewrite": "1"}, "fusion_weights.rewrite", [1.0, 1.0]),
            ({"original": None, "rewrite": 3}, "fusion_weights.original", [1.0, 3.0]),
        ]
        for weights, key, expected in invalid:
            with self.subTest(fusion_weights=weights):
                with self.assertLogs(SEARCH_LOGGER, "WARNING") as logs:
                    tree = _tree(_embeddings(n_subqueries=2), _opt_config(fusion_weights=weights))

                self.assertEqual(_rrf(tree).weights, expected)
                self.assertIn(key, logs.output[0])
                self.assertIn(repr(weights[key.split(".")[1]]), logs.output[0])

    def test_fusion_weights_that_are_no_object_mean_one_for_both(self) -> None:
        for weights in ([1, 1], "1", 2):
            with self.subTest(fusion_weights=weights):
                with self.assertLogs(SEARCH_LOGGER, "WARNING") as logs:
                    tree = _tree(_embeddings(n_subqueries=2), _opt_config(fusion_weights=weights))

                self.assertEqual(_rrf(tree).weights, [1.0, 1.0])
                self.assertIn("fusion_weights", logs.output[0])
                self.assertIn(repr(weights), logs.output[0])

    def test_invalid_rrf_k_falls_back_to_the_qdrant_default_with_a_warning(self) -> None:
        for rrf_k in (0, -3, 1.5, True, "60"):
            with self.subTest(rrf_k=rrf_k):
                with self.assertLogs(SEARCH_LOGGER, "WARNING") as logs:
                    tree = _tree(_embeddings(n_subqueries=2), _opt_config(rrf_k=rrf_k))

                self.assertIsNone(_rrf(tree).k)
                self.assertIn("rrf_k", logs.output[0])
                self.assertIn(repr(rrf_k), logs.output[0])


class ExclusionTestCase(unittest.TestCase):
    """The exclusion filter of a search round is set on every individual search."""

    def test_filter_is_set_on_every_leaf_and_on_no_fusion_node(self) -> None:
        for search_mode in ("dense", "sparse", "dense_sparse", "dense_sparse_colbert"):
            for n_subqueries, n_keywords in ((0, 0), (2, 0), (0, 2), (2, 2)):
                with self.subTest(search_mode=search_mode, subqueries=n_subqueries, keywords=n_keywords):
                    tree = _tree(
                        _embeddings(n_subqueries, n_keywords), _opt_config(search_mode=search_mode), EXCLUSION)

                    nodes = _nodes(tree)
                    leaves = [node for node in nodes if node.prefetch is None]
                    self.assertTrue(leaves)
                    self.assertTrue(all(leaf.filter == EXCLUSION for leaf in leaves))
                    self.assertTrue(all(node.filter is None for node in nodes if node.prefetch is not None))

    def test_without_exclusion_no_node_has_a_filter(self) -> None:
        tree = _tree(_embeddings(2, 2), _opt_config(search_mode="dense_sparse"))

        self.assertTrue(all(node.filter is None for node in _nodes(tree)))


class RetrievePoolTestCase(unittest.TestCase):
    """`_retrieve_pool` passes the exclusion of a single search to Qdrant."""

    def _query_points_kwargs(self, root: models.Prefetch) -> dict[str, Any]:
        client = MagicMock()
        client.query_points.return_value.points = []
        with patch("learn2rag.pipeline.search.Qdrant") as qdrant:
            qdrant.return_value.get_client.return_value = client
            _retrieve_pool(root, {"collection_name": "test_collection"}, _opt_config())
        return dict(client.query_points.call_args.kwargs)

    def test_single_search_gets_the_exclusion_as_query_filter(self) -> None:
        kwargs = self._query_points_kwargs(_tree(_embeddings(), _opt_config(search_mode="dense"), EXCLUSION))

        self.assertEqual(kwargs["query_filter"], EXCLUSION)

    def test_tree_keeps_the_exclusion_on_its_leaves_only(self) -> None:
        kwargs = self._query_points_kwargs(_tree(_embeddings(), _opt_config(search_mode="dense_sparse"), EXCLUSION))

        self.assertIsNone(kwargs.get("query_filter"))
        self.assertTrue(all(leaf.filter == EXCLUSION for leaf in kwargs["prefetch"]))


if __name__ == "__main__":
    unittest.main()
