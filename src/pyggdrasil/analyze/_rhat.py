"""Implements Gelman-Rubin convergence diagnostics - R-hat."""

import arviz as az
import xarray as xr
import numpy as np

# load utils
from pyggdrasil.analyze._utils import truncate_arrays


def rhats(chains: np.ndarray) -> np.ndarray:
    """Compute estimate of rank normalized split R-hat for a set of chains.

    Sometimes referred to as the potential scale reduction factor /
    Gelman-Rubin statistic.

    Used the “rank” method recommended by Vehtari et al. (2019)

    The rank normalized R-hat diagnostic tests for lack of convergence by
    comparing the variance between multiple chains to the variance within
    each chain.
    If convergence has been achieved, the between-chain and within-chain
    variances should be identical.

    Args:
        chains: array of arrays to calculate R-hat for
                (minimum of 2 chains, minimum of 4 draws)

    Returns:
        R-hat for given chains from index 4 to length,
        returns list that is 4 shorter than the length of the chains

    Note:
        - May return NaN if the chains are too short and all values are the same
    """

    # minimal length of chains
    min_length = 4

    # Generate all possible truncation lengths
    max_length = min(len(array) for array in chains)
    truncation_lengths = range(min_length, max_length + 1)

    # TODO(Gordon): potential memory bottelneck, calculate on the fly
    # Truncate arrays to all possible lengths
    truncated_chains = [
        truncate_arrays(chains, length) for length in truncation_lengths
    ]

    # make sure that the arrays are in the correct format
    truncated_chains = [az.convert_to_dataset(arr) for arr in truncated_chains]

    # Calculate R-hat for all possible truncation lengths
    rhats = [az.rhat(az.convert_to_dataset(arr)) for arr in truncated_chains]

    # Return R-hat for all possible truncation lengths
    combined_dataset = xr.concat(rhats, dim="")  # type: ignore

    # Convert the combined dataset to a NumPy array
    rhats = combined_dataset["x"].to_series().to_numpy()

    return rhats
