"""Implementation of the log-probability functions.

The log-probability functions are used to calculate the log-probability of a tree.
"""

import pyggdrasil.tree_inference._tree as tr
from pyggdrasil.tree_inference._tree import Tree

from pyggdrasil.tree_inference._interface import MutationMatrix


def logprobability_fn(
    data: MutationMatrix, tree: tr.Tree, theta: tuple[float, float]
) -> float:
    """Returns a function that calculates the log-probability of a tree.

    Args:
        data: observed mutation matrix to calculate the log-probability of
        tree: tree to calculate the log-probability of
        theta: \theta = (\alpha, \beta) error rates

    Returns:
        log-probability of the tree
        Implements the log-probability function from the paper: Equation 13
        P(D|T,\theta) = \frac{T,\theta | D}{P(T,\theta)}
    """

    # TODO: consider using jsps.special.logsumexp
    # TODO: consider using jnp.einsum

    # likelihood = None
    #
    # # P( \sigma_j | T, \theta)
    # attch_prior = jnp.array([0.5, 0.5])  # TODO: implement this
    # # prod_{i=1}^{n} P(D_{ij} | A(T)_{i˜sigma(j)})
    # prod_cells_likelihood = jnp.prod(likelihood)
    # # [prod_cells_likelihood] * attch_prior
    # prod_cells_likelihood_attch_prior = prod_cells_likelihood * attch_prior
    # # log sum_{\simga_j = 1}^{n+1}
    # log_prob_cells = jsp.special.logsumexp(prod_cells_likelihood_attch_prior)
    # # sum_{j=1}^{m}
    # jnp.sum(log_prob_cells)

    # result = jnp.einsum("ij,ij->i", likelihood, attch_prior)

    raise NotImplementedError("Not implemented yet.")


def _mutation_likelihood(
    cell: int,
    mutation: int,
    sigma: int,
    tree: Tree,
    mutation_mat: MutationMatrix,
):
    """Returns the log-likelihood of a cell / mutaiton.

    Args:
        cell: cell index
        mutation: mutation index
        sigma: mutation node the cell is attached to
        tree: tree object contains the tree topology, labels

    Returns:
        likelihood of the cell / mutation - see Equation 13
        P(D_{ij} | A(T)_{i˜sigma_j})

    Note:
        Notation to SCITE paper:
        i = cell
        j = mutation
        \\simga_j = sigma
        D = mutation_mat
    """

    def _compute_mutation_likelihood(mutation_status: int, ancestor: int) -> float:
        """Returns the mutation likelihood.

        Args:
            mutation_status: mutation status of the cell - D_{ij}
            ancestor: ancestor of the cell attached to the mutation node
                    - A(T)_{i˜sigma_j}

        Returns:
            mutation likelihood - P(D_{ij} | A(T)_{i˜sigma_j})
        """
        # TODO: is this just a boolean?
        # if mutation_status == 0 and ancestor == 0: 1
        # if mutation_status == 1 and ancestor == 1: 1
        # else: 0
        # error rates don't matter here ?
        raise NotImplementedError("Not implemented yet.")

    # A(T) - get ancestor matrix
    ancestor_mat = tr._get_ancestor_matrix(tree.tree_topology)
    # D_{ij} - mutation status
    mutation_status = mutation_mat[cell, mutation]
    # A(T)_{i˜sigma_j} - ancestor of cell i attached to node sigma_j
    ancestor = ancestor_mat[cell, sigma]
    # P(D_{ij} | A(T)_{i˜\delta_j})
    mutation_likelihood = _compute_mutation_likelihood(mutation_status, ancestor)

    return mutation_likelihood