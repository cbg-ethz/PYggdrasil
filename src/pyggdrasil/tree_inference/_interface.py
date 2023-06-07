"""Subpackage-specific interfaces.

Note:
    This submodule should not import other submodules,
    so we do not introduce circular imports.
"""
from typing import Union

import jax
import numpy as np


# Type annotation for a generic array.
Array = Union[jax.Array, np.ndarray]

# A matrix of shape (n_nodes, n_nodes).
# Represents the adjacency matrix of a tree, i.e. A[k, m] = 1
# if node k is the parent of m.
# Note that all nodes have self-loops, so that the diagonal A[k, k] = 1.
# Additionally, we will assume that highest index node is the root.
# In particular, A[:, -1] is a zero vector all but the last element,
# i.e. [0,0,...0,0,1]
TreeAdjacencyMatrix = Array

# Represents mutations in sampled cells (n_mutations, n_cells)
# with n_cell columns, and n_site rows
# (i.e. each row is a site, no root), and its cells attached
MutationMatrix = Array
# Apart from 0 (no mutation) and 1 (mutation present) we can observe these values
# in the experimentally obtained matrices:
HOMOZYGOUS_MUTATION: int = (
    2  # Homozygous mutation observed. See Eq. (8) on p. 5 of the SCITE paper.
)
MISSING_ENTRY: int = 3  # Missing entry.

# A vector containing the sampled cell attachment to nodes of the tree.
# entries refer to mutations/nodes with root as the highest
# entries count from 1 to n_nodes (if root included in sampling, else n_nodes-1)
# indices counted from 0 / pythonic index refer to cell (sample numbers)
CellAttachmentVector = Array

# A matrix of shape (n_mutations+1, n_mutations+1), defining relatedness
# as in SCITE paper without the last row truncated
# Additionally, we will assume that highest index node is the root.
# In particular, A[:, -1] is a zero vector all but the last element,
# i.e. [0,0,...0,0,1]
# A node is its own ancestor.
AncestorMatrix = Array

# Observational Error rates
# tuple of (fpr, fnr)
ErrorRates = tuple[float, float]
