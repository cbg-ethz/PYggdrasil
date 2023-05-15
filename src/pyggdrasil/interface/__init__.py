"""Commonly used interfaces, used across different subpackages.

Note:
  - Subpackage-specific interfaces should be
    in a subpackage's version of such module.
  - This subpackage should not depend on other subpackages, so we
    do not introduce circular imports.
"""

from dataclasses import dataclass
import xarray as xr


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
    """Pure MCMC data - easy to plot."""

    iteration: xr.DataArray
    tree: xr.DataArray
    log_probability: xr.DataArray


@dataclass
class AugmentedMcmcData:
    """Augmented MCMC data - easy to plot. Contains distance."""

    iteration: xr.DataArray
    tree: xr.DataArray
    log_probability: xr.DataArray
    distance: xr.DataArray
