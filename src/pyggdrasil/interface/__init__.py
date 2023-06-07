"""Commonly used interfaces, used across different subpackages.

Note:
  - Subpackage-specific interfaces should be
    in a subpackage's version of such module.
  - This subpackage should not depend on other subpackages, so we
    do not introduce circular imports.
"""

import xarray as xr
import jax.numpy as jnp
import jax

from dataclasses import dataclass

from pyggdrasil import TreeNode

# MCMC sample in xarray format.
# example:
# <xarray.Dataset>
# Dimensions:          (from_node_k: 10, to_node_k: 10)
# Coordinates:
#  * from_node_k      (from_node_k) int64 8 2 3 1 4 7 6 0 5 9
#  * to_node_k        (to_node_k) int64 8 2 3 1 4 7 6 0 5 9
# Data variables:
#    iteration        int64 12
#    tree             (from_node_k, to_node_k) float64 0.0 0.0 0.0 ... 0.0 0.0
#    log-probability  float64 -121.6
MCMCSample = xr.Dataset


@dataclass
class PureMcmcData:
    """Pure MCMC data

    Attributes:
        iterations: jax.Array
                iteration numbers
        trees: list[TreeNode]
                list of TreeNode objects
        log_probabilities: jax.Array
                log-probabilities of the trees
    """

    iterations: jax.Array
    trees: list[TreeNode]
    log_probabilities: jax.Array

    def get_sample(self, iteration: int) -> tuple[int, TreeNode, float]:
        """Return a sample from the MCMC chain.

        Args:
            iteration: iteration number
        Returns:
            tree: TreeNode object
            log_probability: log-probability of the tree
        """
        # get index of iteration
        iteration_idx = jnp.where(self.iterations == iteration)[0][0]

        return (
            iteration,
            self.trees[iteration_idx],
            self.log_probabilities[iteration_idx].item(),
        )

    def append(self, iteration: int, tree: TreeNode, log_probability: float):
        """Append a sample to the MCMC chain.

        Args:
            iteration: iteration number
            tree: TreeNode object
            log_probability: log-probability of the tree
        """
        self.iterations = jnp.append(self.iterations, iteration)
        self.trees.append(tree)
        self.log_probabilities = jnp.append(self.log_probabilities, log_probability)
