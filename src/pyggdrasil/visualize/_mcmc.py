"""Methods for visualizing mcmc samples."""


import matplotlib.pyplot as plt

from numpy import ndarray
from pathlib import Path
from typing import Union

from pyggdrasil import TreeNode
from pyggdrasil.interface import PureMcmcData
from pyggdrasil.distances import TreeDistance, TreeSimilarity, calculate_distance_matrix


def save_log_p_iteration(iterations: ndarray, log_p: ndarray, output_dir: Path) -> None:
    """Save plot of log probability vs iteration number to disk."""

    # make matplotlib figure, given the axes

    fig, ax = plt.subplots()
    ax = fig.add_subplot(111)
    ax.plot(iterations, log_p, color="blue")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("log P")
    ax.tick_params(axis="y", labelcolor="red")
    ax.legend(loc="upper right")
    fullpath = output_dir / "logP_iteration.svg"
    fig.savefig(fullpath, format="svg")  # type: ignore


def _ax_log_p_iteration(ax: plt.Axes, data: PureMcmcData) -> plt.Axes:
    """Make Axes of log probability vs iteration number for all given runs."""

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Distance")
    ax.plot(data.iterations, data.trees, color="blue", label="Distance")
    ax.tick_params(axis="y", labelcolor="blue")
    ax.legend(loc="upper left")

    return ax


def save_dist_iteration(
    iterations: ndarray, dist_simi: ndarray, ylabel: str, output_dir: Path
) -> None:
    """Save plot of distance to true tree vs iteration number to disk."""

    # make matplotlib figure, given the axes

    fig, ax = plt.subplots()
    ax = fig.add_subplot(111)
    ax.plot(iterations, dist_simi, color="red")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="y", labelcolor="red")
    ax.legend(loc="upper right")
    fullpath = output_dir / "dist_iteration.svg"
    fig.savefig(fullpath, format="svg")  # type: ignore


def _ax_dist_iteration(ax: plt.Axes, data: PureMcmcData) -> plt.Axes:
    """Make Axes of distance to true tree vs iteration number for all given runs."""

    ax.set_ylabel("Probability")
    ax.plot(data.iterations, data.log_probabilities, color="red", label="Probability")
    ax.tick_params(axis="y", labelcolor="red")
    ax.legend(loc="upper right")

    return ax


def _calc_distances_to_true_tree(
    true_tree: TreeNode,
    similarity_measure: Union[TreeDistance, TreeSimilarity],
    trees: list[TreeNode],
) -> ndarray:
    """Calculate distances to true tree for all samples.

    Args:
        distance : TreeDistance
        trees : list[TreeNode]
    Returns:
        ndarray
    """

    # make list of true tree objects as long as the list of samples
    true_trees = [true_tree]

    # calculate distances
    distances = calculate_distance_matrix(
        true_trees, trees, distance=similarity_measure
    )

    # flatten distances
    distances = distances.flatten()

    return distances


def _save_dist_to_disk(distances: ndarray, fullpath: Path) -> None:
    """Calculate log probabilities for all samples."""

    # TODO: consider moving this to a serializer

    # save distances to disk as json
    with open(fullpath, "w") as f:
        f.write(str(distances))


def _save_top_trees_plots():
    """Save top trees by log probability to disk."""
    return NotImplementedError


def make_mcmc_run_panel(
    data: PureMcmcData,
    true_tree: TreeNode,
    similarity_measure: Union[TreeDistance, TreeSimilarity],
    out_dir: Path,
) -> None:
    """Make panel of MCMC run plots, save to disk.

    Choose distance to use for calculating distances to true tree.

    Plots:
    - log probability vs iteration number
    - distance to true tree vs iteration number
    - top 3 trees by log probability, with distances to true tree

    Args:
        data : mcmc samples
        true_tree : true tree to compare samples to
        similarity_measure : similarity or distance measure
        out_dir : path to output directory
    Returns:
        None
    """

    # check if true_tree is not None
    if true_tree is None:
        raise ValueError("true_tree must not be None")

    # make output dir path
    path = Path(out_dir)

    # calculate distances to true tree
    distances = _calc_distances_to_true_tree(true_tree, similarity_measure, data.trees)

    # save distances to disk
    tree_distance = similarity_measure.__class__.__name__
    fullpath = path / f"{tree_distance}_s_to_true_tree.csv"
    _save_dist_to_disk(distances, fullpath)

    # Start building figure
    fig, ax = plt.subplots()

    # Plot distances
    _ax_log_p_iteration(ax, data)

    # Create a secondary axis for probabilities
    ax[1] = ax[0].twinx()
    _ax_dist_iteration(ax[1], data)

    plt.title("Distances and Probabilities over Iterations")
    plt.savefig(path / "dist_prob_over_iterations.svg")
    plt.close()
