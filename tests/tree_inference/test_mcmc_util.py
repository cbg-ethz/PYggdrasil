"""Tests of the MCMC utility functions."""
# _mcmc_util.py

import pytest
import jax.random as random
import jax.numpy as jnp
import numpy as np


import pyggdrasil.tree_inference._mcmc_util as mcmc_util
import pyggdrasil.tree_inference._mcmc as mcmc
import pyggdrasil.tree_inference as tree_inf


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
    tree = mcmc.Tree(adj_mat, labels)  # TODO: make labels random
    # get descendants with matrix exponentiation
    desc01 = mcmc_util._get_descendants(
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


# TODO: add test manual for mcmc_util._prune
def test_prune():
    """Test pruning - manual test."""
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
    tree = mcmc.Tree(adj_mat, labels)
    # tree.to_TreeNode().print_topo()
    # define parent
    parent = 3
    # define subtree - Answer
    subtree_adj_mat = jnp.array(
        [[0, 0, 1, 0], [0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0]]
    )
    subtree_labels = jnp.array([3, 1, 6, 0])
    # define remaining tree
    remaining_adj_mat = jnp.array(
        [[0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]
    )
    remaining_labels = jnp.array([2, 4, 5, 7])
    # do prune - Answer
    subtree_tree, remaining_tree = mcmc_util._prune(tree, parent)

    print(subtree_tree.tree_topology)
    print(subtree_tree.labels)
    print(remaining_tree.tree_topology)
    print(remaining_tree.labels)

    # check that answers are the same
    assert jnp.all(subtree_tree.tree_topology == subtree_adj_mat)
    assert jnp.all(subtree_tree.labels == subtree_labels)
    assert jnp.all(remaining_tree.tree_topology == remaining_adj_mat)
    assert jnp.all(remaining_tree.labels == remaining_labels)
