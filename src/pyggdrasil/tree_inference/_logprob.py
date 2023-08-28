"""Implementation of the log-probability functions.

The log-probability functions are used to calculate the log-probability of a tree.
"""
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from typing import Callable

import pyggdrasil.tree_inference._tree as tr
from pyggdrasil.tree_inference._tree import Tree
from pyggdrasil.tree_inference._ordered_tree import OrderedTree


from pyggdrasil.tree_inference._interface import (
    MutationMatrix,
    AncestorMatrix,
    ErrorRates,
)

# Array of floats of shape (n, m, n+1) where n is the number of mutations,
# m is the number of cells, and n+1 is the number of nodes in the tree.
# The axis are (i=mutation, j=cell, k=attachment node)
MutationLikelihood = jax.Array


def create_logprob(data: MutationMatrix, rates: ErrorRates) -> Callable:
    """Returns a function that calculates the log-probability of a tree,
    with given error rates and data.

    Curries the data and rates into the log-probability function.

    Args:
        data (MutationMatrix): observed mutation matrix
        rates (ErrorRates): \theta = (\fpr, \fnr) error rates

    Returns:
        logprob_ (Callable): function that calculates the log-probability of a tree
    """

    def logprob_(tree: OrderedTree) -> float:
        """Calculates the log-probability of a tree given error rates and data.

        Args:
            tree (Tree): tree to calculate the log-probability of

        Returns:
            log-probability of the tree
        """
        return logprobability_fn(data, tree, rates)

    return logprob_


def logprobability_fn(
    data: MutationMatrix, tree: OrderedTree, theta: ErrorRates
) -> float:
    """Calculates the log-probability of a tree given error rates and data.

    Args:
        data: observed mutation matrix to calculate the log-probability of
        tree: tree to calculate the log-probability of, must be OrderedTree
        theta: \theta = (\fpr, \fnr) error rates

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
    theta: ErrorRates,
) -> MutationLikelihood:
    """Returns the log-likelihood of a cell / mutation /attachment.

    Args:
        cell: cell index
        mutation: mutation index
        sigma: mutation node the cell is attached to
        tree: tree object contains the tree topology, labels
        mutation_mat: mutation matrix
        theta: \theta = (\fpr, \fnr) error rates

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
    theta: ErrorRates,
) -> MutationLikelihood:
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
    fpr, fnr = theta

    # truncate ancestor matrix
    ancestor_matrix = ancestor_matrix[:-1]

    # repeat the ancestor matrix  - tensor of dimensions (n, m, n+1)
    ancestor_tensor = jnp.repeat(ancestor_matrix[:, jnp.newaxis, :], m, axis=1)

    # repeat the mutation matrix  - tensor of dimensions (n, m, n+1)
    mutation_tensor = jnp.repeat(mutation_matrix[:, :, jnp.newaxis], n + 1, axis=2)

    # compute the likelihood
    mutation_likelihood = jnp.zeros((n, m, n + 1))
    # P(D_{ij} = 0| E_{ik} = 0) = 1- fpr
    mask = (mutation_tensor == 0) & (ancestor_tensor == 0)
    mutation_likelihood = jnp.where(mask, 1 - fpr, mutation_likelihood)
    # P(D_{ij} = 0| E_{ik} = 1) = fnr
    mask = (mutation_tensor == 0) & (ancestor_tensor == 1)
    mutation_likelihood = jnp.where(mask, fnr, mutation_likelihood)
    # P(D_{ij} = 1| E_{ik} = 0) = fpr
    mask = (mutation_tensor == 1) & (ancestor_tensor == 0)
    mutation_likelihood = jnp.where(mask, fpr, mutation_likelihood)
    # P(D_{ij} = 1| E_{ik} = 1) = 1 - fnr
    mask = (mutation_tensor == 1) & (ancestor_tensor == 1)
    mutation_likelihood = jnp.where(mask, 1 - fnr, mutation_likelihood)

    # TODO: implement homozygous mutations / missing data

    return mutation_likelihood


def _logprobability_fn_verify(
    data: MutationMatrix, tree: tr.Tree, theta: ErrorRates
) -> float:
    """Calculates the log-probability of a tree given error rates and data.
        Uses basic numpy sums and exp /log to verify einsum /
         log sum exp implementation.
    Args:
        data: observed mutation matrix to calculate the log-probability of
        tree: tree to calculate the log-probability of
        theta: \theta = (\fpr, \fnr) error rates
    Returns:
        log-probability of the tree
        Implements the log-probability function from the paper: Equation 13
        P(D|T,\theta) = \frac{T,\theta | D}{P(T,\theta)}
    """

    #  get log P(D_{ij} | A(T)_i~~sigma_j) i.e. the log mutation likelihood
    #         log (P(D_{ij} | A(T)_{i˜sigma_j}) )
    #             i / axis 0 has n dimensions sum over mutation nodes (n)
    #                 - each mutation
    #             j / axis 1 has m dimensions sum over cells (m)
    #                 - each cell
    #             k / axis 2 has n+1 dimensions sum over nodes (n+1)
    #                 - attachment to mutation and root
    # TODO: replace with _log_mutation_likelihood_verify
    log_mutation_likelihood = _log_mutation_likelihood(tree, data, theta)
    print(f"log-mutation likelihood: {log_mutation_likelihood.shape}")

    # verify that the first dimension is the number of mutations
    assert log_mutation_likelihood[:, 1, 1].shape[0] == tree.labels.shape[0] - 1

    # sum over the n cells  - axis 1 / i
    carrier = jnp.sum(log_mutation_likelihood, axis=0)

    # verify the dimensions are correct
    print(f"carrier after sum over mutations: {carrier.shape}")
    assert carrier.shape == (
        data.shape[1],
        tree.tree_topology.shape[0],
    )  # (m cells, k=n+1 not needed as adjacency included root)

    # exp the carrier
    carrier = jnp.exp(carrier)
    # check the dimensions of the attachment axis next to be summed over
    print(f"carrier after exp: {carrier.shape}")
    assert carrier.shape[1] == tree.tree_topology.shape[0]

    # sum over the n+1 nodes - axis 2 / k
    carrier = jnp.sum(carrier, axis=1)  # axis 1 is now k, then (m, )

    # verify the dimensions are correct
    print(f"carrier after sum over nodes: {carrier.shape}")
    assert carrier.shape == (data.shape[1],)  # (m, )

    # log the carrier
    carrier = jnp.log(carrier)
    # sum over the m cells - axis 0 / j
    carrier = jnp.sum(carrier, axis=0)  # axis 0 is now j, then (1, )

    return float(carrier)
