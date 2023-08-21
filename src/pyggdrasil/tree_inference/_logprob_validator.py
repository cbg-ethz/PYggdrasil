"""Implementation of the log-probability functions.

The log-probability functions are used to calculate the log-probability of a tree.

Implements a dumb version of the log-probability function, which is used for testing.

This version attempts to do it by intuition without looking ta the paper.
"""
import jax
import jax.numpy as jnp

import pyggdrasil.tree_inference._tree as tr

from pyggdrasil.tree_inference import Tree, ErrorRates

# set up logging
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _expected(
    tree: Tree,
    mutation_i: int,
    cell_attachment: int,
) -> int:
    """Calculates the expected likelihood of a tree given error rates and data.

    Args:
        tree: tree to calculate the expected likelihood of
        cell_attachment: cell attachment index (node index attached to),
            for a single cell
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
        logger.debug(
            f"mutation_i={mutation_i} should be present in"
            f" cell attached to node={cell_attachment}, so E=1"
        )
        # so return 1
        return 1
    else:
        # mutation_i is not a parent of cell_j
        logger.debug(
            f"mutation_i={mutation_i} should be present in"
            f" cell attached to node={cell_attachment}, so E=0"
        )
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
        logger.debug(f"data={data}, expected={expected}, so P=1-fpr={1-fpr}")
        return 1 - fpr
    elif data == expected == 1:
        logger.debug(f"data={data}, expected={expected}, so P=1-fnr={1-fnr}")
        return 1 - fnr
    elif data == 0 and expected == 1:
        logger.debug(f"data={data}, expected={expected}, so P=fnr={fnr}")
        return fnr
    elif data == 1 and expected == 0:
        logger.debug(f"data={data}, expected={expected}, so P=fpr={fpr}")
        return fpr
    else:
        raise ValueError(
            f"Invalid data and expected values: data={data}, expected={expected}"
        )


def _log_probability(
    mutation_i, tree: Tree, data: int, error_rates: ErrorRates, cell_attachment: int
) -> float:
    """Get slog probability of a cell carring a mutation given and attachment."""

    expected = _expected(tree, mutation_i, cell_attachment)
    p_cell_mutation_attachment = _probability(data, expected, error_rates)
    log_prob = jnp.log(p_cell_mutation_attachment)
    logger.debug(f"log_prob={log_prob}")

    return float(log_prob)


def _exp_sum_mutations(
    tree: Tree, data: jax.Array, error_rates: ErrorRates, cell_attachment: int
) -> float:
    """Returns the exponentiated sum of the log-probabilities of all
        mutations for a given cell and attachment.

    Args:
        data is a column of the data matrix, for a given cell

    """

    # all but the last label, last is root
    mutations = tree.labels[:-1]

    sum = 0
    for mutation in mutations:
        logger.debug(f"For mutation={mutation}")
        data_mutation = int(data[mutation])
        log_prob = _log_probability(
            mutation, tree, data_mutation, error_rates, cell_attachment
        )
        sum += log_prob

    exp_sum = jnp.exp(sum)
    logger.debug(f"exp_sum={exp_sum}")

    return float(exp_sum)


def _marginalize_attachments(
    tree: Tree, data: jax.Array, error_rates: ErrorRates
) -> float:
    """Sums over all possible attachments for a given cell.

    Thi is the total probability of a cell given the data and all attachments.

    Args:
        data is a column of the data matrix, for a given cell

    """

    attachments = tree.labels

    sum = 0
    for attachment in attachments:
        logger.debug(f"For attachment={attachment}")
        sum += _exp_sum_mutations(tree, data, error_rates, attachment)

    logger.debug(f"sum={sum}")
    return float(sum)


def _sum_cell_log(tree: Tree, data: jax.Array, error_rates: ErrorRates) -> float:
    """Returns the sum of the log-probabilities of all cells/mutation
        for ANY attachment.

    Args:
        data is here the whole data matrix
    """
    cells = data.shape[1]
    sum = 0
    for cell in range(cells):
        logger.debug(f"For cell={cell}")
        data_cell = data[:, cell]
        sum += jnp.log(_marginalize_attachments(tree, data_cell, error_rates))
    logger.debug(f"sum={sum}")
    return float(sum)


def logprobability_fn(data: jax.Array, tree: Tree, error_rates: ErrorRates) -> float:
    """Returns the log-probability function."""
    return _sum_cell_log(tree, data, error_rates)


def mutation_likelihood():
    """Returns the mutation likelihood tensor."""
    raise NotImplementedError(
        "Consider implementing a tensor comparison of the validator "
        "to the fast version. Thi is just the expected tensor."
    )
