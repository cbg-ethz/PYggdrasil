"""Simulations of mutation matrices from mutation trees."""
from enum import Enum

import numpy as np
from jax import random
import jax.numpy as jnp

import pyggdrasil.tree_inference._interface as interface
from typing import Union
from jax import Array

# Mutation matrix without noise
PerfectMutationMatrix = Union[np.ndarray, Array]


def _add_false_positives(
    rng: interface.JAXRandomKey,
    matrix: PerfectMutationMatrix,
    noisy_mat: interface.MutationMatrix,
    false_positive_rate: float,
) -> interface.MutationMatrix:
    """adds false positives to  mutation matrix

    Args:
        rng: JAX random key
        matrix: perfect mutation matrix
        noisy_mat: matrix to modify, accumulated changes
        false_positive_rate: false positive rate :math:`\\alpha`

    Returns:
        Mutation matrix of size and entries as noisy_mat given
         with false positives at rate given
    """

    # P(D_{ij} = 1 |E_{ij}=0)=alpha
    # Generate a random matrix of the same shape as the original
    rand_matrix = random.uniform(key=rng, shape=matrix.shape)
    # Create a mask of elements that satisfy the condition
    # (original value equals y and random value is less than p)
    mask = (matrix == 0) & (rand_matrix < false_positive_rate)
    # Use the mask to set the corresponding elements of the matrix to x
    noisy_mat = jnp.where(mask, 1, noisy_mat)

    return noisy_mat


def _add_false_negatives(
    rng: interface.JAXRandomKey,
    matrix: PerfectMutationMatrix,
    noisy_mat: interface.MutationMatrix,
    false_negative_rate: float,
    observe_homozygous: bool,
) -> interface.MutationMatrix:
    """adds false negatives to mutation matrix

    Args:
        rng: JAX random key
        matrix: perfect mutation matrix
        noisy_mat: matrix to modify, accumulated changes
        false_negative_rate: false positive rate :math:`\\alpha`

    Returns:
        Mutation matrix of size and entries as noisy_mat given
        with false negatives at rate given
    """

    # P(D_{ij}=0|E_{ij}=1) = beta if non-homozygous
    # P(D_{ij}=0|E_{ij}=1) = beta / 2 if homozygous
    rand_matrix = random.uniform(key=rng, shape=matrix.shape)
    mask = matrix == 1
    mask_homozygous = observe_homozygous & (rand_matrix < false_negative_rate / 2)
    mask_non_homozygous = (not observe_homozygous) & (rand_matrix < false_negative_rate)
    mask = mask & np.logical_or(mask_homozygous, mask_non_homozygous)
    noisy_mat = jnp.where(mask, 0, noisy_mat)

    return noisy_mat


def _add_homozygous_errors(
    rng_neg: interface.JAXRandomKey,
    rng_pos: interface.JAXRandomKey,
    matrix: PerfectMutationMatrix,
    noisy_mat: interface.MutationMatrix,
    false_negative_rate: float,
    false_positive_rate: float,
    observe_homozygous: bool,
) -> interface.MutationMatrix:
    """Adds both homozygous errors to mutation matrix, if observe_homozygous.

    Args:
        rng_neg: Jax random key for given E=0
        rng_pos: Jax random key for given E=1
        matrix: perfect mutation matrix
        noisy_mat: matrix to modify, accumulated changes
        false_negative_rate: false negative rate :math:`\\beta`
        false_positive_rate: false positive rate :math:`\\alpha`
        observe_homozygous: is homozygous or not

    Returns:
        Mutation matrix of size and entries as noisy_mat given
        with false homozygous calls at rates given.
    """

    # Add Homozygous False Un-mutated
    # # P(D_{ij} = 2 | E_{ij} = 0) = alpha*beta / 2
    rand_matrix = random.uniform(key=rng_neg, shape=matrix.shape)
    mask = (
        (matrix == 0)
        & observe_homozygous
        & (rand_matrix < (false_negative_rate * false_positive_rate / 2))
    )
    noisy_mat = jnp.where(mask, 2, noisy_mat)

    # Add Homozygous False Mutated
    # P(D_{ij} = 2| E_{ij} = 1) = beta / 2
    rand_matrix = random.uniform(key=rng_pos, shape=matrix.shape)
    mask = (
        (matrix == 1) & observe_homozygous & (rand_matrix < (false_negative_rate / 2))
    )
    noisy_mat = jnp.where(mask, 2, noisy_mat)

    return noisy_mat


def _add_missing_entries(
    rng: interface.JAXRandomKey,
    matrix: PerfectMutationMatrix,
    noisy_mat: interface.MutationMatrix,
    missing_entry_rate: float,
) -> interface.MutationMatrix:
    """Adds missing entries

    Args:
        rng: Jax random key
        matrix: perfect mutation matrix
        noisy_mat: matrix to modify, accumulated changes

    Returns:
        Mutation matrix of size and entries as noisy_mat given
        with missing entries e=3 at rate given.
    """

    # Add missing data
    # P(D_{ij} = 3) = missing_entry_rate
    rand_matrix = random.uniform(key=rng, shape=matrix.shape)
    mask = rand_matrix < missing_entry_rate
    noisy_mat = jnp.where(mask, 3, noisy_mat)

    return noisy_mat


def add_noise_to_perfect_matrix(
    rng: interface.JAXRandomKey,
    matrix: PerfectMutationMatrix,
    false_positive_rate: float = 1e-5,
    false_negative_rate: float = 1e-2,
    missing_entry_rate: float = 1e-2,
    observe_homozygous: bool = False,
) -> interface.MutationMatrix:
    """

    Args:
        rng: JAX random key
        matrix: binary matrix with 1 at sites where a mutation is present.
          Shape (n_cells, n_sites).
        false_positive_rate: false positive rate :math:`\\alpha`.
          Should be in the half-open interval [0, 1).
        false_negative_rate: false negative rate :math:`\\beta`.
          Should be in the half-open interval [0, 1).
        missing_entry_rate: fraction os missing entries.
          Should be in the half-open interval [0, 1).
          If 0, no missing entries are present.
        observe_homozygous: if true, some homozygous mutations will be observed
          due to noise. See Eq. (8) on p. 5 of the original SCITE paper.

    Returns:
        array with shape (n_cells, n_sites)
          with the observed mutations and ``int`` data type.
          Entries will be:
            - 0 (no mutation)
            - 1 (mutation present)
            - ``HOMOZYGOUS_MUTATION`` if ``observe_homozygous`` is true
            - ``MISSING ENTRY`` if ``missing_entry_rate`` is non-zero
    """
    # RNGs for false positives, false negatives, and missing data
    rng_false_pos, rng_false_neg, rng_miss, rng_homo_pos, rng_homo_neg = random.split(
        rng, 5
    )
    # make matrix to edit and keep unchanged
    noisy_mat = matrix.copy()

    # Add False Positives - P(D_{ij} = 1 |E_{ij}=0)=alpha
    noisy_mat = _add_false_positives(
        rng_false_pos, matrix, noisy_mat, false_positive_rate
    )

    # Add False Negatives
    # P(D_{ij}=0|E_{ij}=1) = beta if non-homozygous
    # P(D_{ij}=0|E_{ij}=1) = beta / 2 if homozygous
    noisy_mat = _add_false_negatives(
        rng_false_neg, matrix, noisy_mat, false_negative_rate, observe_homozygous
    )

    # Add Homozygous Errors if applicable
    noisy_mat = _add_homozygous_errors(
        rng_homo_neg,
        rng_homo_pos,
        matrix,
        noisy_mat,
        false_negative_rate,
        false_positive_rate,
        observe_homozygous,
    )

    # Add missing entries
    noisy_mat = _add_missing_entries(rng_miss, matrix, noisy_mat, missing_entry_rate)

    return noisy_mat


class CellAttachmentStrategy(Enum):
    """Enum representing valid strategies for attaching
    cells to the mutation tree.

    Allowed values:
      - UNIFORM_INCLUDE_ROOT: each node in the tree has equal probability
          of being attached a cell
      - UNIFORM_EXCLUDE_ROOT: each non-root node in the tree has equal probability
          of being attached a cell
    """

    UNIFORM_INCLUDE_ROOT = "UNIFORM_INCLUDE_ROOT"
    UNIFORM_EXCLUDE_ROOT = "UNIFORM_EXCLUDE_ROOT"


def attach_cells_to_tree(
    rng: interface.JAXRandomKey,
    tree: interface.TreeAdjacencyMatrix,
    n_cells: int,
    strategy: CellAttachmentStrategy,
) -> PerfectMutationMatrix:
    """Attaches cells to the mutation tree.

    Args:
        rng: JAX random key
        tree: matrix representing mutation tree
        n_cells: number of cells to sample
        strategy: cell attachment strategy.
          See ``CellAttachmentStrategy`` for more information.

    Returns:
        binary matrix of shape ``(n_cells, n_sites)``,
          where ``n_sites`` is determined from the ``tree``
    """
    if n_cells < 1:
        raise ValueError(f"Number of sampled cells {n_cells} cannot be less than 1.")

    if strategy == CellAttachmentStrategy.UNIFORM_INCLUDE_ROOT:
        tree.shape[0]
    elif strategy == CellAttachmentStrategy.UNIFORM_EXCLUDE_ROOT:
        tree.shape[0] - 1
    else:
        raise ValueError(f"CellAttachmentStrategy {strategy} is not valid.")

    # uniform attachment prior
    # p = 1/n_sites

    # cells_on_tree = random.bernoulli(rng, p, [n_cells, n_sites])

    raise NotImplementedError("This function needs to be implemented.")


def sample_cell_attachment(
    rng: interface.JAXRandomKey,
    n_cells: int,
    n_nodes: int,
    strategy: CellAttachmentStrategy,
):
    """Samples the node attachment for each cell given a uniform prior.

    Args:
        rng: JAX random key
        n_cells: number of cells
        n_nodes: number of nodes including root,
            nodes counted from 1, root = n_nodes
        strategy: ell attachment strategy.
          See ``CellAttachmentStrategy`` for more information.

    Returns:
        \\sigma - sample/cell attachment vector
            where elements are sampled node indices
            (index+1) of \\sigma corresponds to cell/sample number
    """

    return NotImplementedError("This function needs to be implemented.")
