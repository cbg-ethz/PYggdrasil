"""Tests of the tree class / utility functions used for the MCMC"""
# _tree.py

import pytest
import jax.random as random
import jax.numpy as jnp
import numpy as np


import pyggdrasil.tree_inference._tree as tr
import pyggdrasil.tree_inference as tree_inf

from pyggdrasil.tree_inference._tree import Tree


@pytest.mark.parametrize("seed", [42, 32, 44])
@pytest.mark.parametrize("n_nodes", [5, 10, 15])
def test_get_descendants(seed: int, n_nodes: int):
    """Test get_descendants against Floyd-Warshall algorithm."""
    # get random numbers keys
    rng = random.PRNGKey(seed)
    rng_tree, rng_nodes, rng_labels = random.split(rng, 3)
    # generate random tree
    adj_mat = tree_inf.generate_random_tree(rng_tree, n_nodes)
    # generate random nodes
    parent = int(random.randint(rng_nodes, shape=(1,), minval=0, maxval=n_nodes)[0])
    # assign labels - randomly sample
    labels = random.permutation(rng_labels, np.arange(n_nodes))
    # make tree
    tree = Tree(adj_mat, labels)  # TODO: make labels random
    # get descendants with matrix exponentiation
    desc01 = tr._get_descendants(
        jnp.array(tree.tree_topology), jnp.array(tree.labels), parent
    )
    # get descendants with Floyd-Warshall
    adj_mat_prime = np.array(tree.tree_topology)
    np.fill_diagonal(adj_mat_prime, 1)
    parent_idx = np.where(tree.labels == parent)[0][0]
    desc02_idx = np.where(tree_inf.get_descendants(adj_mat_prime, parent_idx) == 1)
    desc02 = tree.labels[desc02_idx]
    desc02 = desc02[desc02 != parent]

    # check that descendants are the same
    assert jnp.all(desc01 == desc02)


def test_get_root_label():
    """Test get_root_label. - manual test."""
    adj_mat = jnp.array(
        [
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0],
        ]
    )
    labels = jnp.array([3, 1, 2, 4, 5, 6, 0, 7])

    tree = Tree(adj_mat, labels)

    root_label_test = tr._get_root_label(tree)

    root_label_true = 7

    assert jnp.all(root_label_test == root_label_true)


def test_resort_root_to_end():
    """Test resort_root_to_end. - manual test."""
    adj_mat = jnp.array([[1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 1], [0, 0, 0, 1]])
    labels = jnp.array([1, 4, 2, 3])
    tree = Tree(adj_mat, labels)
    resort_tree = tr._resort_root_to_end(tree, 4)

    assert jnp.all(
        resort_tree.tree_topology
        == jnp.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 1]])
    )
    assert jnp.all(resort_tree.labels == jnp.array([1, 2, 3, 4]))