"""Analyze trees from MCMC runs."""

import jax.numpy as jnp
import logging
import numpy as np

from pyggdrasil.tree_inference import unpack_sample

from pyggdrasil.interface import MCMCSample, PureMcmcData


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def to_pure_mcmc_data(mcmc_samples: list[MCMCSample]) -> PureMcmcData:
    """Converts McmcRunData to PureMcmcData.

    Takes a list of MCMCSamples
    converts it into a xarray easy to plot.

    Args:
        mcmc_samples : list[MCMCSample] - list of MCMC samples
    Returns:
        PureMcmcData
    """

    # length of the list of samples
    mcmc_samples_len = len(mcmc_samples)
    # unpack each sample into a list of tuples
    iterations = jnp.empty(mcmc_samples_len)
    trees = []
    log_probabilities = jnp.empty(mcmc_samples_len)

    for index, sample in enumerate(mcmc_samples):
        logger.debug(f"converting sample of index: {index}")
        iteration, tree, logprobability = unpack_sample(sample)
        iterations = iterations.at[index].set(iteration)
        trees.append(tree.to_TreeNode())
        log_probabilities = log_probabilities.at[index].set(logprobability)

    # convert to PureMcmcData
    pure_data = PureMcmcData(iterations, trees, log_probabilities)

    return pure_data


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
