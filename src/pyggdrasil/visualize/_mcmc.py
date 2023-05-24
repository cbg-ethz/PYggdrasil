"""Methods for visualizing mcmc samples."""


from numpy import ndarray

from pyggdrasil import TreeNode
from pyggdrasil.distances import TreeDistance, calculate_distance_matrix


def save_log_p_iteration():
    """Save plot of log probability vs iteration number to disk."""
    pass


def _ax_log_p_iteration():
    """Make Axes of log probability vs iteration number for all given runs."""
    pass


def save_dist_iteration():
    """Save plot of distance to true tree vs iteration number to disk."""
    pass


def _ax_dist_iteration():
    """Make Axes of distance to true tree vs iteration number for all given runs."""
    pass


def _calc_distances_to_true_tree(
    true_tree: TreeNode, distance: TreeDistance, trees: list[TreeNode]
) -> ndarray:
    """Calculate distances to true tree for all samples.

    Args:
        distance : TreeDistance
        trees : list[TreeNode]
    Returns:
        list[float]
    """

    # make list of true tree objects as long as the list of samples
    true_trees = [true_tree] * len(trees)

    # calculate distances
    distances = calculate_distance_matrix(true_trees, trees, distance=distance)

    return distances


def _save_dist_to_disk():
    """Calculate log probabilities for all samples."""

    # TODO: consider moving this to serialize

    pass


def _save_top_trees_plots():
    """Save top trees by log probability to disk."""
    pass


def make_mcmc_run_panel():
    """Make panel of MCMC run plots.
    - log probability vs iteration number
    - distance to true tree vs iteration number
    - top 3 trees by log probability, with distances to true tree
    """
    pass
