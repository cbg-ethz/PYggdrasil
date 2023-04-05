"""Tests of the MCMC functions."""
# _mcmc.py
import pytest
import jax.random as random
import jax.numpy as jnp

import pyggdrasil.tree_inference._mcmc as mcmc
import pyggdrasil.tree_inference as tree_inf

from pyggdrasil.tree_inference._tree import Tree


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_swap_node_labels_move(seed: int):
    """Test swap_node_labels."""
    # get random numbers keys
    rng = random.PRNGKey(seed)
    rng_tree, rng_nodes = random.split(rng, 2)
    # generate random tree
    n_nodes = 10
    adj_mat = tree_inf.generate_random_tree(rng_tree, n_nodes)
    # generate random nodes - NB: root may not be swapped, hence n_nodes-1
    node1, node2 = random.randint(rng_nodes, shape=(2,), minval=0, maxval=n_nodes - 1)
    # assign labels
    tree01 = mcmc.Tree(adj_mat, jnp.arange(n_nodes))
    # swap labels
    tree01_labels = tree01.labels
    tree02 = mcmc._swap_node_labels_move(tree01, node1, node2)
    tree02_labels = tree02.labels
    # check that labels are swapped
    assert tree01_labels[node1] == tree02_labels[node2]
    assert tree01_labels[node2] == tree02_labels[node1]
    # check that other labels are unchanged
    for i in range(n_nodes):
        if i not in [node1, node2]:
            assert tree01_labels[i] == tree02_labels[i]


# TODO: test prune_and_reattach_move
def test_prune_and_reattach_move():
    """Test prune_and_reattach_move. - manual test"""
    # Original tree
    tree_adj = jnp.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
        ]
    )
    labels = jnp.array([6, 5, 4, 3, 2, 1])
    tree = Tree(tree_adj, labels)

    # new tree
    new_tree = mcmc._prune_and_reattach_move(tree, pruned_node=2, attach_to=3)

    new_tree_corr = Tree(
        jnp.array(
            [
                [0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
            ]
        ),
        jnp.array([6, 3, 1, 5, 4, 2]),
    )

    assert jnp.array_equal(new_tree.tree_topology, new_tree_corr.tree_topology)
    assert jnp.array_equal(new_tree.labels, new_tree_corr.labels)
