"""Analyze trees from MCMC runs."""

import jax.numpy as jnp
from typing import Union, Callable
import logging

from pyggdrasil import TreeNode, compare_trees
from pyggdrasil.tree_inference import Tree, unpack_sample

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


def check_run_for_tree(
    desired_tree: Union[Tree, TreeNode], mcmc_samples: PureMcmcData
) -> Union[PureMcmcData, bool]:
    """Check if a tree is in an MCMC run.

    Returns of list of tuples of (iteration, tree, log-probability), or False.
    Goes through entire chain to find all instances of the tree.

    Args:
        desired_tree : Tree
        mcmc_samples : McmcRunData
    Returns:
        bool or list(tuple[int, Tree, float])
    """

    # check type of desired_tree
    if isinstance(desired_tree, Tree):
        # convert to TreeNode
        desired_tree = desired_tree.to_TreeNode()

    # Check if the desired tree is in the MCMC run
    results = PureMcmcData(jnp.empty(0), [], jnp.empty(0))
    for i, tree in enumerate(mcmc_samples.trees):
        if compare_trees(desired_tree, tree):  # type: ignore
            iteration, _, log_probability = mcmc_samples.get_sample(i)
            results.append(iteration, tree, log_probability)

    if results.iterations.size > 0:
        return results

    return False


class Scorer:
    """Provide a set of callable metrics to score trees, given curried metrics."""

    def __init__(self, metrics: dict[str, Callable[[TreeNode], float]]) -> None:
        """Initialize Scorer with a set of metrics curried with a TreeNode."""
        self.metrics = metrics

    def score(self, t: TreeNode) -> dict[str, float]:
        """Score a tree with the metrics."""
        return {name: fun(t) for name, fun in self.metrics.items()}


class Analyzer:
    """Analyze trees from MCMC runs given a Scorer."""

    def __init__(self, scorer: Scorer) -> None:
        """Initialize Analyzer with a Scorer."""
        self.scorer = scorer

    def analyze(self, mcmc_samples: PureMcmcData) -> dict[str, list[float]]:
        """Analyze trees from MCMC runs given a Scorer."""
        scores = {name: [] for name in self.scorer.metrics.keys()}
        for tree in mcmc_samples.trees:
            for name, score in self.scorer.score(tree).items():
                scores[name].append(score)
        return scores


def analyze_mcmc_run(mcmc_data: PureMcmcData,
                     metrics: list[Callable[[TreeNode, TreeNode], float]],
                     tree: TreeNode
) -> None:
    """Analyze a MCMC run.

    Args:
        mcmc_data : PureMcmcData
                    MCMC run data to analyze of iteration no., tree, and log-probability
        metrics : list[Callable[[TreeNode, TreeNode], float]]
                    List of metrics to apply to the trees.
        tree : TreeNode
                    Tree to compare all applicable metrics to.
    """

    # TODO: consider moving all analysis to a new module - tree inference gets too big
