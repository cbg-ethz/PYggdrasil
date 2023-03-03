"""Subpackage-specific interfaces.

Note:
    This submodule should not import other submodules,
    so we do not introduce circular imports.
"""
from typing import Any
from jax.random import PRNGKeyArray
import numpy as np

# JAX random key
JAXRandomKey = PRNGKeyArray


# TODO(Pawel): This class for mutation trees
# needs to be made precise later.
# Probably, it'll be a subtype of a general labeled tree.
MutationTreeMatrix = Any

# Mutation matrix type. It should be a numpy array of a specified format.
MutationMatrix = np.ndarray

# Apart from 0 (no mutation) and 1 (mutation present) we can observe these values
# in the experimentally obtained matrices:
HOMOZYGOUS_MUTATION: int = (
    2  # Homozygous mutation observed. See Eq. (8) on p. 5 of the SCITE paper.
)
MISSING_ENTRY: int = 3  # Missing entry.
