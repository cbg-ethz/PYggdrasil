"""Analyze trees from MCMC runs."""

import jax.numpy as jnp
from typing import Union, Callable
import logging

from pyggdrasil import TreeNode, compare_trees
from pyggdrasil.tree_inference import Tree

from pyggdrasil.interface import PureMcmcData


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def analyze_mcmc_run(
    mcmc_data: PureMcmcData,
    metric: Callable[[TreeNode, TreeNode], float],
    base_tree: TreeNode,
) -> tuple[list[int], list[float]]:
    """Analyze a MCMC run.

    Args:
        mcmc_data : PureMcmcData
                    MCMC run data to analyze of iteration no.,
                    tree, and log-probability
        metric :  Callable[[TreeNode, TreeNode], float]
                     metric to apply to the trees.
        base_tree : TreeNode
                    Tree to compare all applicable metrics to.

    Returns:
        list[int], list[float]
        Iteration number and results of the metric.
    """

    # initialize list of results
    results = []
    iterations = []

    # curried metric function
    def metric_curried(tree: TreeNode) -> float:
        """Return curried metric function."""
        return metric(base_tree, tree)

    # iterate through the trees
    for i, t in enumerate(mcmc_data.trees):
        # compare the trees
        results.append(metric_curried(t))
        iterations.append(mcmc_data.iterations[i].__int__())

    return iterations, results


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
