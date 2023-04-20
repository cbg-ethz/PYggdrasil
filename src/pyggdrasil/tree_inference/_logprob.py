"""Implementation of the log-probability functions.

The log-probability functions are used to calculate the log-probability of a tree.
"""
import jax
import jax.numpy as jnp

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
    n, m = data.shape  # m = number of cells, n = number of mutations

    raise NotImplementedError("logprobability_fn not implemented yet")
    # return float(log_prob)


def _mutation_likelihood(
    cell: int,
    mutation: int,
    sigma: int,
    tree: Tree,
    mutation_mat: MutationMatrix,
    theta: tuple[float, float],
) -> Mutation_Likelihood:
    """Returns the log-likelihood of a cell / mutaiton.

    Args:
        cell: cell index
        mutation: mutation index
        sigma: mutation node the cell is attached to
        tree: tree object contains the tree topology, labels
        mutation_mat: mutation matrix
        theta: \theta = (\alpha, \beta) error rates

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

    # A(T) - get ancestor matrix
    ancestor_mat = tr._get_ancestor_matrix(tree.tree_topology)
    # D_{ij} - mutation status
    int(mutation_mat[cell, mutation])
    # A(T)_{i˜sigma_j} - ancestor of cell i attached to node sigma_j
    int(ancestor_mat[cell, sigma])
    # P(D_{ij} | A(T)_{i˜\delta_j})
    # mutation_likelihood = #_compute_mutation_likelihood(mutation_status, ancestor)

    raise NotImplementedError("Not implemented yet.")


def _compute_mutation_likelihood(
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

    Note:
        let k = sigma_j
    """

    # m = number of cells, n-1 = number of mutations, ex root
    n_red, m = mutation_matrix.shape
    # n = number of nodes, including root
    n = n_red + 1

    alpha, beta = theta

    # truncate ancestor matrix
    ancestor_matrix = ancestor_matrix[:-1]

    # repeat the ancestor matrix  - tensor of dimensions (n-1, m, n)
    ancestor_tensor = jnp.repeat(ancestor_matrix[:, jnp.newaxis, :], m, axis=1)

    # repeat the mutation matrix  - tensor of dimensions (n-1, m, n)
    mutation_tensor = jnp.repeat(mutation_matrix[:, :, jnp.newaxis], n, axis=2)

    # compute the likelihood
    mutation_likelihood = jnp.zeros((n - 1, m, n))
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
