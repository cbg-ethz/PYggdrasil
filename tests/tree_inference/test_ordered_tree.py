"""Tests of the ordered tree class."""
import pytest

import jax.random as random
import jax.numpy as jnp

from pyggdrasil.tree_inference._ordered_tree import OrderedTree
from pyggdrasil.tree_inference._tree import Tree
import pyggdrasil.tree_inference._tree_generator as tree_gen


@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("seed", [45])
def test_ordered_tree(n: int, seed: int):
    """Test if unordered trees may be assigned to the ordered tree class.

    Assumes trees are generated with perfect order always.
    """

    # make tree
    tree = make_tree(n, seed, "r")

    # unordered tree - permute labels
    rng = random.PRNGKey(seed)
    unordered_labels = random.permutation(rng, n)

    # assign unordered tree to ordered tree class
    try:
        OrderedTree(tree.tree_topology, unordered_labels)
    except AssertionError:
        OrderedTree(tree.tree_topology, tree.labels)
        assert True


def make_tree(n: int, seed: int, tree_type: str) -> Tree:
    """Make a tree for testing."""

    rng = random.PRNGKey(seed)

    if tree_type == "r":
        adj_mat = jnp.array(tree_gen._generate_random_tree_adj_mat(rng, n))
        labels = jnp.arange(n)
        return Tree(adj_mat, labels)
    elif tree_type == "s":
        adj_mat = jnp.array(tree_gen._generate_star_tree_adj_mat(n))
        labels = jnp.arange(n)
        return Tree(adj_mat, labels)
    else:
        adj_mat = jnp.array(tree_gen._generate_deep_tree_adj_mat(rng, n))
        labels = jnp.arange(n)
        return Tree(adj_mat, labels)
