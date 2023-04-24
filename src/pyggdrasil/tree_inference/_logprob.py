"""Implementation of the log-probability functions.

The log-probability functions are used to calculate the log-probability of a tree.
"""
import jax
import jax.numpy as jnp
import jax.scipy as jsp

import pyggdrasil.tree_inference._tree as tr
from pyggdrasil.tree_inference._tree import Tree

from pyggdrasil.tree_inference._interface import MutationMatrix
from pyggdrasil.tree_inference._interface import AncestorMatrix

# Array of floats of shape (n, m, n+1) where n is the number of mutations,
# m is the number of cells, and n+1 is the number of nodes in the tree.
# The axis are (i=mutation, j=cell, k=attachment node)
Mutation_Likelihood = jax.Array


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

    #  get log P(D_{ij} | A(T)_i~~sigma_j) i.e. the log mutation likelihood
    log_mutation_likelihood = _log_mutation_likelihood(tree, data, theta)

    # sum_{i=1}^{n} of prior
    lse_arg = jnp.einsum("ijk->jk", log_mutation_likelihood)

    # log sum_{k=1}^{n+1} exp or prior
    # log sum exp operation over sigma_j = k
    log_sum_exp = jsp.special.logsumexp(lse_arg, axis=-1)

    # sum_{j=1}^{m}
    log_prob = float(jnp.einsum("j->", log_sum_exp))

    return log_prob


def _log_mutation_likelihood(
    tree: Tree,
    mutation_mat: MutationMatrix,
    theta: tuple[float, float],
) -> Mutation_Likelihood:
    """Returns the log-likelihood of a cell / mutation /attachment.

    Args:
        cell: cell index
        mutation: mutation index
        sigma: mutation node the cell is attached to
        tree: tree object contains the tree topology, labels
        mutation_mat: mutation matrix
        theta: \theta = (\alpha, \beta) error rates

    Returns:
        likelihood of the cell / mutation - see Equation 13
        log (P(D_{ij} | A(T)_{i˜sigma_j}) )
            i / axis 0 has n dimensions sum over mutation nodes (n)
                - each mutation
            j / axis 1 has m dimensions sum over cells (m)
                - each cell
            k / axis 2 has n+1 dimensions sum over nodes (n+1)
                - attachment to mutation and root
    """

    # A(T) - get ancestor matrix
    ancestor_mat = tr._get_ancestor_matrix(tree.tree_topology)
    # get mutation likelihood
    mutation_likelihood = _mutation_likelihood(mutation_mat, ancestor_mat, theta)
    # get log of it
    log_mutation_likelihood = jnp.log(mutation_likelihood)

    return log_mutation_likelihood


def _mutation_likelihood(
    mutation_matrix: MutationMatrix,
    ancestor_matrix: AncestorMatrix,
    theta: tuple[float, float],
) -> Mutation_Likelihood:
    """Returns the mutation likelihood given the data and expected mutation matrix.

    Currently, only implements non-homozygous mutations.

    Args:
        mutation_matrix: mutation matrix
        ancestor_matrix: ancestor matrix
    Returns:
        mutation likelihood - P(D_{ij} | A(T)_{i˜sigma_j})
            i / axis 0 has n dimensions sum over mutation nodes (n)
                - each mutation
            j / axis 1 has m dimensions sum over cells (m)
                - each cell
            k / axis 2 has n+1 dimensions sum over nodes (n+1)
                - attachment to mutation and root
    Note:
        let k = sigma_j
    """

    # m = number of cells, n-1 = number of mutations, ex root
    n, m = mutation_matrix.shape
    alpha, beta = theta

    # truncate ancestor matrix
    ancestor_matrix = ancestor_matrix[:-1]

    # repeat the ancestor matrix  - tensor of dimensions (n, m, n+1)
    ancestor_tensor = jnp.repeat(ancestor_matrix[:, jnp.newaxis, :], m, axis=1)

    # repeat the mutation matrix  - tensor of dimensions (n, m, n+1)
    mutation_tensor = jnp.repeat(mutation_matrix[:, :, jnp.newaxis], n + 1, axis=2)

    # compute the likelihood
    mutation_likelihood = jnp.zeros((n, m, n + 1))
    # P(D_{ij} = 0| E_{ik} = 0) = 1- alpha
    mask = (mutation_tensor == 0) & (ancestor_tensor == 0)
    mutation_likelihood = jnp.where(mask, 1 - alpha, mutation_likelihood)
    # P(D_{ij} = 0| E_{ik} = 1) = beta
    mask = (mutation_tensor == 0) & (ancestor_tensor == 1)
    mutation_likelihood = jnp.where(mask, beta, mutation_likelihood)
    # P(D_{ij} = 1| E_{ik} = 0) = alpha
    mask = (mutation_tensor == 1) & (ancestor_tensor == 0)
    mutation_likelihood = jnp.where(mask, alpha, mutation_likelihood)
    # P(D_{ij} = 1| E_{ik} = 1) = 1 - beta
    mask = (mutation_tensor == 1) & (ancestor_tensor == 1)
    mutation_likelihood = jnp.where(mask, 1 - beta, mutation_likelihood)

    return mutation_likelihood
