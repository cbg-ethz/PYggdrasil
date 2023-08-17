"""Implementation of the log-probability functions.

The log-probability functions are used to calculate the log-probability of a tree.

Implements a dumb version of the log-probability function, which is used for testing.
"""

import jax.numpy as jnp

import pyggdrasil.tree_inference._tree as tr

from pyggdrasil.tree_inference import Tree, ErrorRates


def _expected(
    tree: Tree,
    mutation_i,
    cell_attachment,
) -> int:
    """Calculates the expected likelihood of a tree given error rates and data.

    Args:
        tree: tree to calculate the expected likelihood of
        cell_attachment: cell attachment vector, for a single cell
        mutation_i : mutation index

    Returns:
        expected likelihood of the tree
    """

    # get the ancestor matrix
    ancestor_mat = tr._get_ancestor_matrix(tree.tree_topology)
    # truncate the last row, which is the root
    ancestor_mat = ancestor_mat[:-1, :]

    # get parent of mutation_i
    # get column of ancestor matrix for a given cell_attachment
    ancestor_col = ancestor_mat[:, cell_attachment]
    # get indices of non-zero elements
    parents_i = []
    for i, val in enumerate(ancestor_col):
        if val != 0:
            parents_i.append(i)

    if mutation_i in parents_i:
        # mutation_i is a parent of cell_j
        # so return 1
        return 1
    else:
        # mutation_i is not a parent of cell_j
        # so return 0
        return 0


def _probability(data: int, expected: int, error_rates: ErrorRates) -> float:
    """Calculates the probability of a given of observation given the expected value,
        and error rates.

    Args:
        data: data to calculate the probability of
        expected: expected data
        error_rates: error rates"""

    fpr, fnr = error_rates

    if data == expected == 0:
        return 1 - fpr
    elif data == expected == 1:
        return 1 - fnr
    elif data == 0 and expected == 1:
        return fnr
    elif data == 1 and expected == 0:
        return fpr
    else:
        raise ValueError(
            f"Invalid data and expected values: data={data}, expected={expected}"
        )


def _marginalize_attachment(
    mutation_i, tree: Tree, data, error_rates: ErrorRates
) -> float:
    """Marginalize attachment over all possible nodes to attach to."""

    list_of_nodes_to_attach_to = (
        tree.labels
    )  # may attach to any nodes even root - though ideally one removes such data

    prob = 0  # prob of tree given a given cell is attached ANYWHERE on given mutation i
    for node in list_of_nodes_to_attach_to:
        expected = _expected(tree, mutation_i, node)
        p_cell_mutation_attachment = _probability(data, expected, error_rates)
        prob += p_cell_mutation_attachment
    return prob


def _combined_log_prob_cells(
    mutation_i, tree: Tree, data_cells_per_mutation, error_rates: ErrorRates
) -> float:
    """Returns the combined log-probability of all cells for ANY attachment,
     for a given mutation.

    Args:
        data_cells_per_mutation: data for all cells for a given mutation,
         in order of mutations
    """

    prob = 0
    for data in data_cells_per_mutation:
        prob += jnp.log(_marginalize_attachment(mutation_i, tree, data, error_rates))

    return float(prob)


def _combine_log_prob_mutations(tree: Tree, data, error_rates: ErrorRates) -> float:
    """Returns the combined log-probability of all mutations, for ANY attachment.

    Args:
        data: data for all cells for all mutations, in order of mutations
    """

    prob = 0
    for mutation in tree.labels:
        prob += _combined_log_prob_cells(mutation, tree, data[mutation], error_rates)

    return prob


def logprobability_fn_validator(tree: Tree, data, error_rates: ErrorRates) -> float:
    """Returns the log-probability of a tree given data and error rates."""

    return _combine_log_prob_mutations(tree, data, error_rates)
