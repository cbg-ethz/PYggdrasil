"""Implements Effective-Sample-Size convergence diagnostics - ESS."""

import arviz as az
import xarray as xr
import numpy as np


def truncate_arrays(arrays: np.ndarray, length: int) -> np.ndarray:
    """Truncate arrays to given length.

    Args:
        arrays: array of arrays to truncate
        length: length to truncate arrays to

    Returns:
        truncated arrays
    """
    truncated_arrays = [arr[:length] for arr in arrays]

    return np.array(truncated_arrays)


def ess(chains: np.ndarray) -> np.ndarray:
    """Calculates the effective sample size of a set of chains.


    used the “bulk” method recommended by Vehtari et al. (2019)
    rank normalized draws are used to calculate the effective sample size

    Args:
    chains: array of arrays to calculate ESS for
            (minimum of 2 chains, minimum of 4 draws)

    Returns:
       ess: effective sample size of chains
    """

    # minimal length of chains
    min_length = 4

    # Generate all possible truncation lengths
    max_length = min(len(array) for array in chains)
    truncation_lengths = range(min_length, max_length + 1)

    # Truncate arrays to all possible lengths
    truncated_chains = [
        truncate_arrays(chains, length) for length in truncation_lengths
    ]

    # make sure that the arrays are in the correct format
    truncated_chains = [az.convert_to_dataset(arr) for arr in truncated_chains]

    # Calculate ESS for all possible truncation lengths
    ess_v = [az.ess(az.convert_to_dataset(arr)) for arr in truncated_chains]

    # Return ESS for all possible truncation lengths
    combined_dataset = xr.concat(ess_v, dim="")  # type: ignore

    # Convert the combined dataset to a NumPy array
    ess_v = combined_dataset["x"].to_series().to_numpy()

    return ess_v
