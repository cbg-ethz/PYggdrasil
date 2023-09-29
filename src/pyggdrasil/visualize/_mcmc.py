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
    ax: plt.Axes, iterations: list[int], log_probs: list[float]  # type: ignore
) -> plt.Axes:  # type: ignore
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
    ax: plt.Axes,  # type: ignore
    iteration: list[int],
    distances: list[float],
    metric_name: str,
) -> plt.Axes:  # type: ignore
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


def save_rhat_iteration(
    iteration: list[int],
    rhats: list[float],
    out_fp: Path,
) -> None:
    """Save plot of rhat vs iteration number to disk.

    Args:
        iteration: list[int]
            Iteration numbers.
        out_fp: Path
            Output file path.
        rhats: ndarray
            R hat values for each iteration.
    """

    # make matplotlib figure, given the axes

    fig, ax = plt.subplots()
    ax.set_xlabel("Iteration")  # type: ignore
    # get name of distance measure
    ax.set_ylabel(r"$\hat{R}$")  # type: ignore
    ax.plot(iteration, rhats, color="black")  # type: ignore
    # specifying horizontal line type
    # see limits https://arxiv.org/pdf/1903.08008.pdf
    plt.axhline(y=1.1, color="b", linestyle="--", linewidth=0.5)  # type: ignore
    plt.axhline(y=1.01, color="r", linestyle="-", linewidth=0.5)  # type: ignore
    ax.tick_params(axis="y", labelcolor="black")  # type: ignore
    # ensure the output directory exists
    # strip the filename from the output path
    output_dir = out_fp.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    # save the figure
    fig.savefig(out_fp, format="svg")  # type: ignore


def save_rhat_iteration_AD_DL(
    iteration: list[int],
    rhats_AD: list[float],
    rhats_DL: list[float],
    out_fp: Path,
) -> None:
    """Save plot of rhat vs iteration number to disk.

    Args:
        iteration: list[int]
            Iteration numbers.
        out_fp: Path
            Output file path.
        rhats_AD: ndarray
            R hat values for each iteration for AD.
        rhats_DL: ndarray
            R hat values for each iteration for DL.
    """

    # make matplotlib figure, given the axes

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlabel("Iteration")  # type: ignore
    # get name of distance measure
    ax.set_ylabel(r"$\hat{R}$")  # type: ignore
    ax.set_ylim(0.8, 5.0)  # type: ignore
    ax.plot(iteration, rhats_AD, color="darkgreen", label="AD")  # type: ignore
    ax.plot(iteration, rhats_DL, color="darkorange", label="DL")  # type: ignore
    # specifying horizontal line type
    # see limits https://arxiv.org/pdf/1903.08008.pdf
    plt.axhline(y=1.1, color="b", linestyle="--", linewidth=0.5)  # type: ignore
    plt.axhline(y=1.01, color="r", linestyle="-", linewidth=0.5)  # type: ignore
    ax.tick_params(axis="y", labelcolor="black")  # type: ignore
    ax.set_yticks([1, 2, 3, 4, 5])  # type: ignore
    ax.legend(loc="upper right")  # type: ignore
    # ensure the output directory exists
    # strip the filename from the output path
    output_dir = out_fp.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    # save the figure
    fig.savefig(out_fp, format="svg", bbox_inches="tight")  # type: ignore


def save_ess_iteration_AD_DL(
    iteration: list[int],
    ess_bulk_AD: list[float],
    ess_bulk_DL: list[float],
    ess_tail_AD: list[float],
    ess_tail_DL: list[float],
    out_fp: Path,
) -> None:
    """Save plot of ess vs iteration number to disk.

    Args:
        iteration: list[int]
            Iteration numbers.
        out_fp: Path
            Output file path.

        TODO: add description
    """

    # make matplotlib figure, given the axes

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlabel("Iteration")  # type: ignore
    # get name of distance measure
    ax.set_ylabel(r"$ESS$")  # type: ignore
    # bulk
    ax.plot(iteration, ess_bulk_AD, color="darkgreen", linestyle="-")  # type: ignore
    ax.plot(iteration, ess_bulk_DL, color="darkorange", linestyle="-")  # type: ignore
    # tail
    ax.plot(iteration, ess_tail_AD, color="darkgreen", linestyle="--")  # type: ignore
    ax.plot(iteration, ess_tail_DL, color="darkorange", linestyle="--")  # type: ignore

    # artificial legend
    # add solid line black line to ledgend as bulk
    ax.plot([], [], color="black", label="bulk", linestyle="-")  # type: ignore
    # add dashed line black line to ledgend as tail
    ax.plot([], [], color="black", label="tail", linestyle="--")  # type: ignore
    # add darkgreen marker to ledgend as AD
    ax.plot(  # type: ignore
        [], [], color="darkgreen", label="AD", marker="o", linestyle=""
    )
    # add darkorange marker to ledgend as DL
    ax.plot(  # type: ignore
        [], [], color="darkorange", label="DL", marker="o", linestyle=""
    )

    # specifying horizontal line type
    # see limits https://arxiv.org/pdf/1903.08008.pdf
    # 400 at least
    plt.axhline(y=400, color="r", linestyle="-", linewidth=0.5)  # type: ignore
    ax.tick_params(axis="y", labelcolor="black")  # type: ignore
    ax.legend(loc="upper right")  # type: ignore
    # ensure the output directory exists
    # strip the filename from the output path
    output_dir = out_fp.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    # save the figure
    fig.savefig(out_fp, format="svg", bbox_inches="tight")  # type: ignore
