"""Tests of the MCMC functions."""
# _mcmc.py
import pytest
import jax.random as random
import jax.numpy as jnp


import pyggdrasil.tree_inference._mcmc as mcmc
import pyggdrasil.tree_inference as tree_inf


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_swap_node_labels_move(seed: int):
    """Test swap_node_labels."""
    # get random numbers keys
    rng = random.PRNGKey(seed)
    rng_tree, rng_nodes = random.split(rng, 2)
    # generate random tree
    n_nodes = 10
    adj_mat = tree_inf.generate_random_tree(rng_tree, n_nodes)
    # generate random nodes
    node1, node2 = random.randint(rng_nodes, shape=(2,), minval=0, maxval=n_nodes)
    # assign labels
    tree01 = mcmc.Tree(adj_mat, jnp.arange(n_nodes))
    # swap labels
    tree01_labels = tree01.labels
    tree02 = mcmc._swap_node_labels_move(tree01, node1, node2)
    tree02_labels = tree02.labels
    # check that labels are swapped
    assert tree01_labels[node1] == tree02_labels[node2]
    assert tree01_labels[node2] == tree02_labels[node1]
