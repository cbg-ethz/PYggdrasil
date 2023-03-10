"""Subpackage-specific interfaces.

Note:
    This submodule should not import other submodules,
    so we do not introduce circular imports.
"""
from typing import Union

import jax
from jax.random import PRNGKeyArray
import numpy as np

# JAX random key
JAXRandomKey = PRNGKeyArray

# Type annotation for a generic array.
Array = Union[jax.Array, np.ndarray]

# A matrix of shape (n_nodes, n_nodes).
# Represents the adjacency matrix of a tree, i.e. A[k, m] = 1
# if node k is the parent of m.
# Note that we do not include self-loops, so that the diagonal A[k, k] = 0.
# Additionally, we will assume that 0 is the root.
# In particular, A[:, 0] is the zero vector.
TreeAdjacencyMatrix = Array

# Represents mutations in sampled cells (n_cells, n_sites)
MutationMatrix = np.ndarray
# Apart from 0 (no mutation) and 1 (mutation present) we can observe these values
# in the experimentally obtained matrices:
HOMOZYGOUS_MUTATION: int = (
    2  # Homozygous mutation observed. See Eq. (8) on p. 5 of the SCITE paper.
)
MISSING_ENTRY: int = 3  # Missing entry.
