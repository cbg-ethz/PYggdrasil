"""Methods for visualizing mcmc samples."""
import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path

from pyggdrasil.interface import PureMcmcData
from pyggdrasil.visualize import plot_tree_mcmc_sample

# use SVG backend for matplotlib
matplotlib.use("SVG")


def save_log_p_iteration(
    iterations: list[int], log_probs: list[float], output_fp: Path
) -> None:
    """Save plot of log probability vs iteration number to disk.

    Args:
        output_dir: Path
            Output directory to save the plot.
        iterations: list[int]
            Iteration numbers.
        log_probs: list[float]
            Log probabilities.
    """

    # make matplotlib figure, given the axes

    fig, ax = plt.subplots()
    _ax_log_p_iteration(ax, iterations, log_probs)  # type: ignore
    # ensure the output directory exists
    # strip the filename from the output path
    output_dir = output_fp.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    # save the figure
    fig.savefig(output_fp, format="svg")  # type: ignore


def _ax_log_p_iteration(
    ax: plt.Axes, iterations: list[int], log_probs: list[float]
) -> plt.Axes:
    """Make Axes of log probability vs iteration number for all given runs."""

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\log(P(D|T,\theta))$")
    ax.plot(iterations, log_probs, color="black")
    ax.tick_params(axis="y", labelcolor="black")

    return ax


def save_metric_iteration(
    iteration: list[int],
    distances: list[float],
    metric_name: str,
    out_fp: Path,
) -> None:
    """Save plot of distance to true tree vs iteration number to disk.

    Args:
        iteration: list[int]
            Iteration numbers.
        out_fp: Path
            Output file path.
        distances: ndarray
            Distances to true tree for each iteration.
        metric_name: str
            Name of distance metric.
    """

    # make matplotlib figure, given the axes

    fig, ax = plt.subplots()
    _ax_dist_iteration(ax, iteration, distances, metric_name)  # type: ignore
    # ensure the output directory exists
    # strip the filename from the output path
    output_dir = out_fp.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    # save the figure
    fig.savefig(out_fp, format="svg")  # type: ignore


def _ax_dist_iteration(
    ax: plt.Axes,
    iteration: list[int],
    distances: list[float],
    metric_name: str,
) -> plt.Axes:
    """Make Axes of distance to true tree vs iteration number for all given runs."""

    ax.set_xlabel("Iteration")
    # get name of distance measure
    ax.set_ylabel(metric_name)
    ax.plot(iteration, distances, color="black", label=metric_name)
    ax.tick_params(axis="y", labelcolor="black")
    # ax.legend(loc="upper right")

    return ax


def _save_top_trees_plots(data: PureMcmcData, output_dir: Path) -> None:
    """Save plots of top three trees by log probability to disk."""

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


# def make_mcmc_run_panel(
#     data: PureMcmcData,
#     true_tree: TreeNode,
#     similarity_measure: TreeSimilarityMeasure,
#     out_dir: Path,
# ) -> None:
#     """Make panel of MCMC run plots, save to disk.
#     ^
#         Choose distance to use for calculating distances to true tree.
#
#         Plots:
#         - log probability vs iteration number
#         - distance to true tree vs iteration number
#         - distance and log probability vs iteration number
#         - top 3 trees by log probability, with distances to true tree
#
#         Args:
#             data : mcmc samples
#             true_tree : true tree to compare samples to
#             similarity_measure : similarity or distance measure
#             out_dir : path to output directory
#         Returns:
#             None
#     """
#
#     # check if true_tree is not None
#     if true_tree is None:
#         raise ValueError("true_tree must not be None")
#
#     # make output dir path
#     path = Path(out_dir)
#
#     # calculate distances to true tree
#     distances = _calc_distances_to_true_tree(true_tree,
#                                              similarity_measure, data.trees)
#
#     # FIGURE 1: shared iteration axis: logP, distance
#     # Start building figure
#     fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
#     # Plot distances
#     ax1 = _ax_log_p_iteration(
#         plt.Axes(ax1), data.iterations.tolist(), data.log_probabilities.tolist()
#     )
#     # Create a secondary axis for probabilities
#     ax2 = ax1.twinx()
#     ax2 = _ax_dist_iteration(ax2, data.iterations.tolist(), list(distances), "blast")
#     plt.title("Distances and Probabilities over Iterations")
#     plt.savefig(path / "dist_logP_iter.svg")
#     plt.close()
#
#     # FIGURE 2:  iteration axis: logP
#     fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
#     # Plot plot logP
#     ax1 = _ax_log_p_iteration(
#         plt.Axes(ax1), data.iterations.tolist(), data.log_probabilities.tolist()
#     )
#     plt.title("Log Probability over Iterations")
#     plt.savefig(path / "logP_iter.svg")
#     plt.close()
#
#     # FIGURE 3:  iteration axis: distance
#     fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
#     # Plot plot distance
#     ax1 = _ax_dist_iteration(
#         plt.Axes(ax1), data.iterations.tolist(), list(distances), "blast"
#     )
#     plt.title("Distance over Iterations")
#     plt.savefig(path / "dist_iter.svg")
#     plt.close()
#
#     # FIGURE 4:  top 3 trees by logP
#     _save_top_trees_plots(data, path)
#
#     # FIGURE 5:  plot true tree
#     plot_tree_no_print(true_tree, "true_tree", path)
