"""Commonly used interfaces, used across different subpackages.

Note:
  - Subpackage-specific interfaces should be
    in a subpackage's version of such module.
  - This subpackage should not depend on other subpackages, so we
    do not introduce circular imports.
"""

import xarray as xr


# MCMC sample in xarray format.
# example:
# array.Dataset {
# dimensions:
# 	from_node_k = 5 ;
# 	to_node_k = 5 ;
# 	rng_key_run = 2 ;
# variables:
# 	int64 iteration() ;
# 	int32 from_node_k(from_node_k) ;
# 	int32 to_node_k(to_node_k) ;
# 	int32 tree(from_node_k, to_node_k) ;
# 	float64 log-probability() ;
# 	uint32 rng_key_run(rng_key_run) ;
# // global attributes:
# }
MCMCSample = xr.Dataset
