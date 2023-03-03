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
    rng_false_pos, rng_false_neg, rng_miss = random.split(rng, 3)

    # TODO(Pawel, Gordon): Now probably it's the easiest to draw from random.bernoulli
    #   matrices corresponding to the false positives and false negatives
    #   and adjust the mutation matrix accordingly

    # Now the matrix can be adjusted according to missing entry simulation

    raise NotImplementedError("This function needs to be implemented.")


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
    tree: interface.MutationTreeMatrix,
    n_cells: int,
    strategy: CellAttachmentStrategy,
) -> PerfectMutationMatrix:
    """Attaches cells to the mutation tree.

    Args:
        rng: JAX random key
        tree: mutation tree
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
