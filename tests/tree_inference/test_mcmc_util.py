"""Tests of the MCMC utility functions."""
# _mcmc_util.py

import jax.numpy as jnp


import pyggdrasil.tree_inference._mcmc_util as mcmc_util
from pyggdrasil.tree_inference._tree import Tree


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
    tree = Tree(adj_mat, labels)
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


def test_reattach():
    """Test _reattach. - manual test."""
    # Original tree
    # jnp.array(
    #     [
    #         [0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0],
    #         [1, 0, 0, 0, 0, 0],
    #         [0, 1, 1, 0, 0, 0],
    #         [0, 0, 0, 1, 1, 0],
    #     ]
    # )
    # jnp.array([6, 5, 4, 3, 2, 1])

    subtree_corr = Tree(
        jnp.array([[0, 0, 0], [0, 0, 0], [1, 1, 0]]), jnp.array([5, 4, 2])
    )
    r_tree = Tree(jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), jnp.array([6, 3, 1]))
    new_tree = mcmc_util._reattach(r_tree, subtree_corr, 3, 2)

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
    print(new_tree.tree_topology)
    print(new_tree.labels)
    assert jnp.all(new_tree.labels == new_tree_corr.labels)
    assert jnp.all(new_tree.tree_topology == new_tree_corr.tree_topology)
