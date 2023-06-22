"""Methods for visualizing mcmc samples."""

import matplotlib
import matplotlib.pyplot as plt

import json
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
            Iteration numbers, 1-indexed.
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
            Iteration numbers, 1-indexed.
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
    """Make Axes of distance to true tree vs iteration number for all given runs.

    Args:
        ax: plt.Axes
            Matplotlib axes.
        iteration: list[int]
            Iteration numbers, 1-indexed.
        distances: list[float]
            Distances to true tree for each iteration.
        metric_name: str
            Name of distance metric.

    Returns:
        ax: plt.Axes
    """

    ax.set_xlabel("Iteration")
    # get name of distance measure
    ax.set_ylabel(metric_name)
    ax.plot(iteration, distances, color="black", label=metric_name)
    ax.tick_params(axis="y", labelcolor="black")
    # ax.legend(loc="upper right")

    return ax


def save_top_trees_plots(data: PureMcmcData, output_dir: Path) -> None:
    """Save plots of top three trees by log probability to disk,
     with accompanying json to node the iterations.

    Args:
        data : mcmc samples
        output_dir : path to output directory

    Returns:
        None

    Saves:
        top_tree_1.svg
        top_tree_2.svg
        top_tree_3.svg
        top_trees_iterations.json
    """

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

    # save iteration numbers of top three trees
    top_iterations = top_iterations.tolist()
    info = {
        "iteration numbers": top_iterations,
        "log probabilities": data.log_probabilities[top_indices].tolist(),
    }

    with open(output_dir / "top_tree_info.json", "w") as f:
        json.dump(info, f)
