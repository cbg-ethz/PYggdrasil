"""Implementation of the log-probability functions.

The log-probability functions are used to calculate the log-probability of a tree.

Implements a dumb version of the log-probability function, which is used for testing.

This version tried to do the literal approach of Eqn 13 in the paper, first line.
"""
import jax

from itertools import product

import pyggdrasil.tree_inference._tree as tr

from pyggdrasil.tree_inference import Tree

from pyggdrasil.tree_inference._config import ErrorRates


def _all_attachments(m_cells: int, node_lables: jax.Array):
    """Returns list of all possible cell attachment vectors.

    elements of cell attachemnt vector  elements are the node labels and
     the index is the cell index
    """

    numbers = list(node_lables)
    permutations = list(product(numbers, repeat=m_cells))

    return permutations


def _expected(
    cell_j: int,
    mutation_i: int,
    attachment_vector: jax.Array,
    ancestor_mat: jax.Array,
) -> int:
    """Element of expected matrix"""

    # child node - get the node that cell_j is attached to
    child_node = attachment_vector[cell_j]

    # is mutation_i a parent or the child node itself where cell_j is attached to?
    expected = ancestor_mat[mutation_i, child_node]

    return int(expected)


def _prob(observation, expected, error_rates: ErrorRates) -> float:
    """Probability of a given observation given the expected value and error rates."""

    fpr, fnr = error_rates.fpr, error_rates.fnr

    if observation == expected == 1:
        return 1 - fnr
    elif observation == expected == 0:
        return 1 - fpr
    elif observation == 1 and expected == 0:
        return fpr
    elif observation == 0 and expected == 1:
        return fnr
    else:
        raise ValueError(
            f"Invalid observation and expected values: {observation}, {expected}"
        )


def _log_prob(prob: float) -> float:
    """Log-probability of a given probability."""

    return float(jax.numpy.log(prob))


def _log_prob_all_mutations(
    data_for_cell_j: jax.Array,
    mutations: jax.Array,
    cell_j,
    attachment,
    tree: Tree,
    error_rates: ErrorRates,
) -> float:
    """Log-probability of all mutations for a given attachment vector and cell sample

    Args:
        data_for_cell_j: data for cell j over all mutations
        mutations: list of mutation labels
        cell_j: cell index
        attachment: attachment vector
        tree: tree
    """

    ancestor_mat = tr._get_ancestor_matrix(tree.tree_topology)

    log_prob_all_mutations = 0

    for mutation_i in mutations:
        expected = _expected(cell_j, mutation_i, attachment, ancestor_mat)
        prob = _prob(data_for_cell_j[mutation_i], expected, error_rates)
        log_prob = _log_prob(prob)
        log_prob_all_mutations += log_prob

    return float(log_prob_all_mutations)


def _log_prob_all_cells_mutations(
    data: jax.Array, tree: Tree, error_rates: ErrorRates, attachment: jax.Array
) -> float:
    """Log-probability of all cells and mutations.

    Args:
        data: data
        tree: tree
        error_rates: error rates
        attachment : attachment vector
    """

    log_prob_all_cells_mutations = 0

    # get no of cells
    m_cells = data.shape[1]

    # loop over all cells
    for cell_j in range(m_cells):
        # get data for cell j
        data_for_cell_j = data[:, cell_j]

        # get all mutations
        mutations = tree.labels

        # get log-probability of all mutations for a given attachment vector and
        # cell sample
        log_prob_all_mutations = _log_prob_all_mutations(
            data_for_cell_j, mutations, cell_j, attachment, tree, error_rates
        )

        log_prob_all_cells_mutations += log_prob_all_mutations

    return float(log_prob_all_cells_mutations)
