"""Tests of the MCMC functions."""
# _mcmc.py
import jax.random as random
import jax.numpy as jnp


import pyggdrasil.tree_inference._mcmc as mcmc
import pyggdrasil.tree_inference as tree_inf


def test_swap_node_labels_move():
    """Test swap_node_labels."""
    rng = random.PRNGKey(42)
    adj_mat = tree_inf.generate_random_tree(rng, 10)
    tree01 = mcmc.Tree(adj_mat, jnp.arange(10))
    tree01_labels = tree01.labels
    tree02 = mcmc._swap_node_labels_move(tree01, 4, 8)
    tree02_labels = tree02.labels
    assert tree01_labels[4] == tree02_labels[8]
