"""Tests of the tree generator."""
# _tree_generator.py

import pytest
import numpy as np
import jax.random as random

import pyggdrasil.tree_inference as tree_inf


@pytest.mark.parametrize("seed", [42, 32])
@pytest.mark.parametrize("n_nodes", [5, 10])
def test_generate_random_tree(seed: int, n_nodes: int):
    """Test generate_random_tree."""

    # get random numbers key
    rng = random.PRNGKey(seed)
    # get random tree
    adj_matrix = tree_inf.generate_random_tree(rng, n_nodes)
    # check for number of nodes
    shape = adj_matrix.shape
    assert shape[0] == shape[1]
    assert shape[0] == n_nodes
    # total number of edges is n-1
    assert np.sum(adj_matrix) == n_nodes - 1
    # each node has one parent, but the root itself has none
    parenthood = np.ones(n_nodes - 1)
    parenthood = np.append(parenthood, 0)
    assert np.allclose(np.sum(adj_matrix, axis=0), parenthood)


@pytest.mark.parametrize("n_nodes", [5, 10])
def test_generate_star_tree(seed: int, n_nodes: int):
    """Test generate_star_tree."""

    # get tree
    adj_matrix = tree_inf.generate_star_tree(n_nodes)
    # Check the shape of the adjacency matrix
    assert adj_matrix.shape == (n_nodes, n_nodes)
    # Check if the root is the highest index node
    # The last column should contain all zeros
    assert np.sum(adj_matrix[:, -1]) == 0
    # Check if each non-root node is connected to the root
    # Sum of connections to the root should be n_nodes - 1
    assert np.sum(adj_matrix[-1, :-1]) == n_nodes - 1
    # Check if each non-root node is not connected to other non-root nodes
    # Sum of connections between non-root nodes should be zero
    assert np.sum(adj_matrix[:-1, :-1]) == 0


@pytest.mark.parametrize("seed", [42, 32, 44])
@pytest.mark.parametrize("n_nodes", [5, 10, 15])
def test_generate_deep_tree(seed: int, n_nodes: int):
    """Test generate_deep_tree."""
    # TODO: test ancestor relationships, i.e.
    # ancestor matrix rows

    pass
