"""Implementation of the log-probability functions.

The log-probability functions are used to calculate the log-probability of a tree.

Implements a dumb version of the log-probability function, which is used for testing.

This version tried to do the literal approach of Eqn 13 in the paper, first line.
"""
import jax

from itertools import product

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
