"""Tests for the logprob_validator module."""

import jax.numpy as jnp


from pyggdrasil.tree_inference import Tree
from pyggdrasil.tree_inference._logprob_validator import _expected


def test_expected():
    """ "Test expected function."""

    # create tree       # 0  1  2  3  4
    adj_mat = jnp.array(
        [
            [0, 0, 0, 0, 0, 0],  # 0
            [0, 0, 0, 0, 0, 0],  # 1
            [0, 0, 0, 0, 0, 0],  # 2
            [1, 0, 0, 0, 0, 0],  # 3
            [0, 1, 1, 0, 0, 0],  # 4
            [0, 0, 0, 1, 1, 0],  # 5
        ]
    )
    labels = jnp.array([0, 1, 2, 3, 4, 5])

    tree = Tree(adj_mat, labels)

    # create cell attachment vector
    # say we have 1 cell and the j=0 cell is attached to node i=2
    cell_attachment = jnp.array([2])

    # cells
    # expected matrix       #  0  1  2  3  4
    expected_mat = jnp.array(
        [
            [0, 0, 0, 0, 0],  # 0  mutations
            [0, 0, 0, 0, 0],  # 1
            [1, 0, 0, 0, 0],  # 2
            [0, 0, 0, 0, 0],  # 3
            [1, 0, 0, 0, 0],  # 4
        ]
    )

    # get expected

    for mutation_i in range(5):  #  forget root
        cell_j = 0
        print("mutation_i", mutation_i)
        fn_value = _expected(tree, mutation_i, cell_j, cell_attachment).__int__()
        expected = expected_mat[mutation_i, cell_j].__int__()
        print("fn_value", fn_value)
        assert fn_value == expected
