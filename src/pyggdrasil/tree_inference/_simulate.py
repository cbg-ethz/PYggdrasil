"""Simulations of mutation matrices from mutation trees."""
from enum import Enum

import numpy as np
from jax import random

import pyggdrasil.tree_inference._interface as interface

# Mutation matrix without noise
PerfectMutationMatrix = np.ndarray


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

    # Now the matrix can be adjusted according to formula (2) and (8)
    shape = np.shape(matrix)
    perfect_matrix = matrix
    # P(D_{ij} = 1 |E_{ij}=0)=alpha
    fp_mat = random.bernoulli(rng_false_pos, false_positive_rate, shape)
    matrix = np.add(np.multiply((perfect_matrix == 0), fp_mat), matrix)

    if not observe_homozygous:
        # P(D_{ij}=0|E_{ij}=1) = beta
        fn_mat = random.bernoulli(rng_false_pos, false_negative_rate, shape)
        matrix = np.add(-np.multiply((perfect_matrix == 1), fn_mat), matrix)
    else:
        # P(D_{ij}=0|E_{ij}=1) = beta / 2
        fn_mat = random.bernoulli(rng_false_neg, false_negative_rate / 2, shape)
        matrix = np.add(np.multiply((perfect_matrix == 1), fn_mat), matrix)
        # P(D_{ij} = 2 | E_{ij} = 0) = alpha*beta / 2
        f_hom_neg_mat = random.bernoulli(
            rng_homo_neg, false_positive_rate * false_negative_rate / 2, shape
        )
        matrix = np.add(2 * np.multiply((perfect_matrix == 0), f_hom_neg_mat), matrix)
        # P(D_{ij} = 2| E_{ij} = 1)
        f_hom_pos_mat = random.bernoulli(rng_homo_pos, false_negative_rate / 2, shape)
        matrix = np.add(2 * np.multiply((perfect_matrix == 1), f_hom_pos_mat), matrix)

    # missing data
    miss_mat = random.bernoulli(rng_miss, missing_entry_rate, shape)
    matrix = np.where(miss_mat == 1, np.nan, matrix)

    return matrix


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

    raise NotImplementedError("This function needs to be implemented.")
