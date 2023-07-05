"""Tests of distance and similarity functions
imported from scphylo"""
import anytree
import pytest
import jax

from typing import Callable


import pyggdrasil.distances as dist
from pyggdrasil import TreeNode

import pyggdrasil.tree_inference as ti


@pytest.fixture
def model_tree() -> anytree.Node:
    """Simple tree of the form
    root
    ├── TP53
    │   ├── NRAS
    │   └── BRCA
    └── KRAS
    """
    root = anytree.Node("root")
    tp53 = anytree.Node("TP53", parent=root)
    anytree.Node("KRAS", parent=root)
    anytree.Node("NRAS", parent=tp53)
    anytree.Node("BRCA", parent=tp53)
    return root


@pytest.mark.parametrize(
    "similarity", [dist.MP3Similarity(), dist.AncestorDescendantSimilarity()]
)
def test_self_similarity(similarity, model_tree) -> None:
    """Tests whether similarity(tree, tree) = 1."""

    assert similarity.calculate(model_tree, model_tree) == pytest.approx(1.0)

    # Now we will create the same tree, but in a different order
    # As this should not matter, we expect the same similarity
    root = anytree.Node("root")
    anytree.Node("KRAS", parent=root)
    tp53 = anytree.Node("TP53", parent=root)
    anytree.Node("BRCA", parent=tp53)
    anytree.Node("NRAS", parent=tp53)

    assert similarity.calculate(model_tree, root) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "similarity", [dist.MP3Similarity(), dist.AncestorDescendantSimilarity()]
)
def test_another_is_different_and_symmetric(similarity, model_tree) -> None:
    """Test whether the similarity of different trees is less than 1.0.

    Additionally, we will check whether the function is really symmetric.
    """
    root = anytree.Node("root")
    tp53 = anytree.Node("TP53", parent=root)
    kras = anytree.Node("KRAS", parent=root)
    anytree.Node("NRAS", parent=tp53)
    anytree.Node("BRCA", parent=kras)

    assert similarity.calculate(model_tree, root) < 1.0

    # Check whether the function is symmetric
    if similarity.is_symmetric():
        assert similarity.calculate(model_tree, root) == pytest.approx(
            similarity.calculate(root, model_tree)
        )


def tree_gen(tree_type: str, seed: int) -> Callable[[int], TreeNode]:
    """Return a tree."""
    rng = jax.random.PRNGKey(seed)

    def tree1_(n_nodes: int) -> TreeNode:
        """Return a tree."""
        if tree_type == "r":
            return ti.generate_random_TreeNode(rng, n_nodes)
        elif tree_type == "d":
            return ti.generate_deep_TreeNode(rng, n_nodes)
        else:  # tree_type == "s"
            return ti.generate_star_TreeNode(n_nodes)

    return tree1_


@pytest.mark.parametrize("tree_type1", ["r", "d", "s"])
@pytest.mark.parametrize("seed1", [76, 42])
@pytest.mark.parametrize("tree_type2", ["r", "d", "s"])
@pytest.mark.parametrize("seed2", [13, 42])
@pytest.mark.parametrize("n_nodes", [5, 10])
def test_MP3Similarity(n_nodes: int, tree_type1, seed1: int, tree_type2, seed2: int):
    """Test the MP3Similarity class."""

    tree1_fn = tree_gen(tree_type=tree_type1, seed=seed1)
    tree2_fn = tree_gen(tree_type=tree_type2, seed=seed2)

    tree1 = tree1_fn(n_nodes)
    tree2 = tree2_fn(n_nodes)

    sim = dist.MP3Similarity()

    try:
        result = sim.calculate(
            TreeNode.convert_to_anytree_node(tree1),
            TreeNode.convert_to_anytree_node(tree2),
        )
        print("\n")
        tree1.print_topo()
        tree2.print_topo()
        print(f"MP3: {result}")
    except Exception as e:
        print(e)
        tree1.print_topo()
        tree2.print_topo()
        raise e


# Note: AD metric does not work for star trees, as the first tree
@pytest.mark.parametrize("tree_type1", ["r", "d"])
@pytest.mark.parametrize("seed1", [76])
@pytest.mark.parametrize("tree_type2", ["r", "d", "s"])
@pytest.mark.parametrize("seed2", [13])
@pytest.mark.parametrize("n_nodes", [5])
def test_AncestorDescendantSimilarity(
    n_nodes: int, tree_type1, seed1: int, tree_type2, seed2: int
):
    """Test the AncestorDescendantSimilarity scyphylo class."""

    tree1_fn = tree_gen(tree_type=tree_type1, seed=seed1)
    tree2_fn = tree_gen(tree_type=tree_type2, seed=seed2)

    tree1 = tree1_fn(n_nodes)
    tree2 = tree2_fn(n_nodes)

    sim = dist.AncestorDescendantSimilarity()

    try:
        result = sim.calculate(
            TreeNode.convert_to_anytree_node(tree1),
            TreeNode.convert_to_anytree_node(tree2),
        )
        print("\n")
        tree1.print_topo()
        tree2.print_topo()
        print(f"AD: {result}")
    except Exception as e:
        print("\n")
        tree1.print_topo()
        tree2.print_topo()
        raise e


@pytest.mark.parametrize("tree_type1", ["r", "d", "s"])
@pytest.mark.parametrize("seed1", [76, 42])
@pytest.mark.parametrize("tree_type2", ["r", "d", "s"])
@pytest.mark.parametrize("seed2", [13, 42])
@pytest.mark.parametrize("n_nodes", [5, 10])
def test_AncestorDescendantSimilarity_lq(
    n_nodes: int, tree_type1, seed1: int, tree_type2, seed2: int
):
    """Test the AncestorDescendantSimilarity class of
    Laura Quintas implementation."""

    tree1_fn = tree_gen(tree_type=tree_type1, seed=seed1)
    tree2_fn = tree_gen(tree_type=tree_type2, seed=seed2)

    tree1 = tree1_fn(n_nodes)
    tree2 = tree2_fn(n_nodes)
    sim2 = dist.AncestorDescendantSimilarityInclRoot()

    try:
        result2 = sim2.calculate(
            TreeNode.convert_to_anytree_node(tree1),
            TreeNode.convert_to_anytree_node(tree2),
        )
        print("\n")
        print(f"AD_lq: {result2}")
    except Exception as e:
        print("\n")
        tree1.print_topo()
        tree2.print_topo()
        raise e
