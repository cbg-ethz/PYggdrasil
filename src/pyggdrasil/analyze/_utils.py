"""Analyze trees from MCMC runs."""

import jax.numpy as jnp
import logging

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


# TODO (Gordon): Consider using below classes to analyze trees from MCMC runs,
#  to calculate metrics at once.
# class Scorer:
#     """Provide a set of callable metrics to score trees, given curried metrics."""
#
#     def __init__(self, metrics: dict[str, Callable[[TreeNode], float]]) -> None:
#         """Initialize Scorer with a set of metrics curried with a TreeNode."""
#         self.metrics = metrics
#
#     def score(self, t: TreeNode) -> dict[str, float]:
#         """Score a tree with the metrics."""
#         return {name: fun(t) for name, fun in self.metrics.items()}
#
#
# class Analyzer:
#     """Analyze trees from MCMC runs given a Scorer."""
#
#     def __init__(self, scorer: Scorer) -> None:
#         """Initialize Analyzer with a Scorer."""
#         self.scorer = scorer
#
#     def analyze(self, mcmc_samples: PureMcmcData) -> dict[str, list[float]]:
#         """Analyze trees from MCMC runs given a Scorer."""
#         scores = {name: [] for name in self.scorer.metrics.keys()}
#         for tree in mcmc_samples.trees:
#             for name, score in self.scorer.score(tree).items():
#                 scores[name].append(score)
#         return scores
