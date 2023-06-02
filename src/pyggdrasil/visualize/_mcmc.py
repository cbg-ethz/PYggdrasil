"""Methods for visualizing mcmc samples."""


import matplotlib.pyplot as plt

from numpy import ndarray
from pathlib import Path

from pyggdrasil import TreeNode
from pyggdrasil.interface import PureMcmcData
from pyggdrasil.distances import TreeSimilarityMeasure, calculate_distance_matrix
from pyggdrasil.visualize import plot_tree_mcmc_sample, plot_tree_no_print


def save_log_p_iteration(data: PureMcmcData, output_dir: Path) -> None:
    """Save plot of log probability vs iteration number to disk."""

    # make matplotlib figure, given the axes

    fig, ax = plt.subplots()
    ax = fig.add_subplot(111)
    _ax_log_p_iteration(ax, data)
    fullpath = output_dir / "logP_iteration.svg"
    fig.savefig(fullpath, format="svg")  # type: ignore


def _ax_log_p_iteration(ax: plt.Axes, data: PureMcmcData) -> plt.Axes:
    """Make Axes of log probability vs iteration number for all given runs."""

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\log(P(D|T,\theta))$")
    ax.plot(data.iterations, data.log_probabilities, color="blue", label="logP")
    ax.tick_params(axis="y", labelcolor="blue")
    ax.legend(loc="upper left")

    return ax


def save_dist_iteration(
    data: PureMcmcData,
    output_dir: Path,
    distances: ndarray,
    similarityMeasure: TreeSimilarityMeasure,
) -> None:
    """Save plot of distance to true tree vs iteration number to disk.

    Args:
        data: PureMcmcData
            Data from MCMC runs.
        output_dir: Path
            Output directory to save the plot.
        distances: ndarray
            Distances to true tree for each iteration.
    """

    # make matplotlib figure, given the axes

    fig, ax = plt.subplots()
    ax = fig.add_subplot(111)
    _ax_dist_iteration(ax, data, distances, similarityMeasure)
    fullpath = output_dir / "dist_iteration.svg"
    fig.savefig(fullpath, format="svg")  # type: ignore


def _ax_dist_iteration(
    ax: plt.Axes,
    data: PureMcmcData,
    distances: ndarray,
    similarityMeasure: TreeSimilarityMeasure,
) -> plt.Axes:
    """Make Axes of distance to true tree vs iteration number for all given runs."""

    ax.set_xlabel("Iteration")
    # get name of distance measure
    dist_name = similarityMeasure.__class__.__name__
    ax.set_ylabel("Distance / Similarity")
    ax.plot(data.iterations, distances, color="red", label=dist_name)
    ax.tick_params(axis="y", labelcolor="red")
    ax.legend(loc="upper right")

    return ax


def _calc_distances_to_true_tree(
    true_tree: TreeNode,
    similarity_measure: TreeSimilarityMeasure,
    trees: list[TreeNode],
) -> ndarray:
    """Calculate distances to true tree for all samples.

    Args:
        similarity_measure : similarity or distance function class
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

    # check if directory and file exist, if not create them
    if not fullpath.parent.exists():
        fullpath.parent.mkdir(parents=True)
        fullpath.touch()

    # save distances to disk as json
    with open(fullpath, "w") as f:
        f.write(str(distances))


def _save_top_trees_plots(data: PureMcmcData, output_dir: Path) -> None:
    """Save top trees by log probability to disk."""

    # get indices of top three logP samples
    top_indices = data.log_probabilities.argsort()[-3:][::-1]

    # get iteration numbers of top three logP samples
    top_iterations = data.iterations[top_indices]

    # get samples of top three logP samples
    top_samples = []
    for iteration in top_iterations:
        sample = data.get_sample(iteration)
        top_samples.append(sample)

    # plot top three trees
    rank = 1
    for each in top_samples:
        plot_tree_mcmc_sample(each, output_dir, save_name=f"top_tree_{rank}")
        rank += 1


def make_mcmc_run_panel(
    data: PureMcmcData,
    true_tree: TreeNode,
    similarity_measure: TreeSimilarityMeasure,
    out_dir: Path,
) -> None:
    """Make panel of MCMC run plots, save to disk.
    ^
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
    fullpath = path / f"{tree_distance}_to_true_tree.csv"
    _save_dist_to_disk(distances, fullpath)

    # FIGURE 1: shared iteration axis: logP, distance
    # Start building figure
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    # Plot distances
    ax1 = _ax_log_p_iteration(ax1, data)
    # Create a secondary axis for probabilities
    ax2 = ax1.twinx()
    ax2 = _ax_dist_iteration(ax2, data, distances, similarity_measure)
    plt.title("Distances and Probabilities over Iterations")
    plt.savefig(path / "dist_logP_iter.svg")
    plt.close()

    # FIGURE 2:  iteration axis: logP
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    # Plot plot logP
    ax1 = _ax_log_p_iteration(ax1, data)
    plt.title("Log Probability over Iterations")
    plt.savefig(path / "logP_iter.svg")
    plt.close()

    # FIGURE 3:  iteration axis: distance
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    # Plot plot distance
    ax1 = _ax_dist_iteration(ax1, data, distances, similarity_measure)
    plt.title("Distance over Iterations")
    plt.savefig(path / "dist_iter.svg")
    plt.close()

    # FIGURE 4:  top 3 trees by logP
    _save_top_trees_plots(data, path)

    # FIGURE 5:  plot true tree
    plot_tree_no_print(true_tree, "true_tree", path)
