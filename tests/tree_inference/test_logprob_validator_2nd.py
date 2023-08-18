"""Tests for the logprob_validator module."""


import jax.numpy as jnp

from pyggdrasil.tree_inference import Tree

import pyggdrasil.tree_inference._tree as tr

from pyggdrasil.tree_inference._log_prob_validator_2nd import (
    _all_attachments,
    _expected,
)


def test_attachments():
    """ "Test all attachments function."""

    mutation_labels = jnp.array([0, 1, 2])

    m_cells = 2

    attachments = _all_attachments(m_cells, mutation_labels)

    # attacments_manual = jnp.array([0,0], [0,1], [0,2], [1,0],
    # [1,1], [1,2], [2,0], [2,1], [2,2])

    assert len(attachments) == 9


def test_expected():
    """ "Test expected function."""

    # define tree
    adj_mat = jnp.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]])
    labels = jnp.array([0, 1, 2, 3])
    tree = Tree(adj_mat, labels)
    tr._get_ancestor_matrix(tree.tree_topology)

    # get the ancestor matrix
    ancestor_mat = tr._get_ancestor_matrix(tree.tree_topology)
    # truncate the last row, which is the root
    ancestor_mat = ancestor_mat[:-1, :]

    # define the cell attachment vector
    cell_attachment = jnp.array([0, 1])

    true_expected = jnp.array([[1, 0], [1, 1], [0, 0]])

    # run the function
    for cell_i in range(2):
        for mutation_i in range(3):
            expected = _expected(cell_i, mutation_i, cell_attachment, ancestor_mat)
            assert expected == true_expected[mutation_i, cell_i]
