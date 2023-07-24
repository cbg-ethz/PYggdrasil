"""Methods for visualizing trees."""

import os
from pathlib import Path

import numpy as np
from anytree.exporter import DotExporter
import pydot
import matplotlib
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import networkx as nx
import logging
from typing import Union, Optional

from pyggdrasil import TreeNode

# setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# use SVG backend for matplotlib
matplotlib.use("SVG")

NodeLabel = Union[str, int, float]


def plot_tree(
    tree: TreeNode,
    save_name: str,
    save_dir: Path,
    print_options: dict,
    rename_labels: Optional[dict[str, NodeLabel]] = None,
) -> None:
    """Plot a tree and save it to a file.

    Args:
        tree: TreeNode
            tree to plot, given the root node
                tree.data: may contain info to print for tree / node e.g.
                    "log-likelihood": float
                        likelihood of the tree
                tree.name: name of tree, may use a title of plot
        save_name: str
            name of the file to save the plot to
        save_dir: Path
            directory to save the plot to
        print_options: dict
            options for printing the tree
                "title": bool
                    whether to print the name/title of the tree/root
                "data_tree": dict
                    what attributes to print of the tree
        rename_labels: dict
            dictionary of labels to rename
                {0:1, 1:2, 3:"IPC4", 4:"IPC5"}
                {old_label:new_label, ...}
                pass empty dict to not rename labels
    Returns:
        None

    Note:
        see tests/visualize/test_tree.py for example usage
    """

    # check if tree-name is none and throw log - and error
    if tree.name is None and print_options["title"] is True:
        logger.error("Tree name is None, cannot print title")
        raise ValueError("Tree name is None, cannot print title")

    # make full path
    fullpath = os.path.join(save_dir, save_name)
    # make output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # convert to networkX graph
    # convert to dot graph
    dot_string = ""
    for line in DotExporter(tree):
        dot_string += line
    # convert to networkX graph
    graphs = pydot.graph_from_dot_data(dot_string)
    graph = graphs[0]  # type: ignore
    # convert to networkX graph
    nx_graph = nx.nx_pydot.from_pydot(graph)

    # dynamically set fig size
    # Calculate the depth and width of the tree
    depth = nx.dag_longest_path_length(nx_graph)
    width = max(len(list(nx.descendants(nx_graph, node))) for node in nx_graph.nodes())

    # Calculate an appropriate figure size
    node_width = 0.5  # Width of each node in the figure
    node_height = 1.3  # Height of each node in the figure
    figure_width = width * node_width
    figure_height = 1.5 + depth * node_height

    # plot
    fig = plt.figure(figsize=(figure_width, figure_height))

    # LaTeX preamble
    latex_preamble = r"""
    \usepackage{lmodern}
    \usepackage[T1]{fontenc}
    \usepackage[utf8]{inputenc}
    """

    # Update the font settings in matplotlib
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "text.usetex": True,
            "text.latex.preamble": latex_preamble,
        }
    )

    # relabel Root node
    mapping = {str(tree.name): "R"}
    nx_graph = nx.relabel_nodes(nx_graph, mapping)

    # relabel nodes - if rename_labels is not empty
    if rename_labels:
        # check that all nodes are in rename_labels
        mapping = {}
        for node in nx_graph.nodes:
            if node in rename_labels:
                mapping[node] = rename_labels[node]
            else:
                logging.warning(f"Node {node} not found in rename_labels")
        nx_graph = nx.relabel_nodes(nx_graph, mapping)

    # top down tree layout
    pos = graphviz_layout(nx_graph, prog="dot")
    # as subplots
    ax1 = fig.add_subplot()

    # calculate the label sizes
    node_sizes = np.array([])
    for element in nx_graph.nodes:
        node_sizes = np.append(node_sizes, len(str(element)) * 520)

    # plot graph
    nx.draw(
        nx_graph,
        with_labels=True,
        ax=ax1,
        pos=pos,
        node_color="w",
        edgecolors="k",
        font_size=20,
        font_weight="bold",
        node_size=node_sizes,
        node_shape="o",  # s for square, o for circle
    )

    # make title
    if print_options["title"]:
        fig.suptitle(tree.data["tree-name"], fontsize=20)

    description = ""
    for detail in print_options["data_tree"]:
        if detail == "log-likelihood":
            description += (
                r"$\log(P(D|T,\theta)):$ " + f"{tree.data['log-likelihood']:.2f}" + "\n"
            )
        else:
            description += detail + ": " + str(tree.data[detail]) + "\n"

    plt.text(1, 0.97, description, dict(size=15), transform=ax1.transAxes, va="top")

    # check if fullpath already has .svg extension, else add it
    if not fullpath.endswith(".svg"):
        fullpath += ".svg"
    # save
    plt.savefig(fullpath, bbox_inches="tight", format="svg")
    plt.close()
    logger.info(f"Saved tree plot to {fullpath}")


# TODO: Consider creating MCMCSample class
def plot_tree_mcmc_sample(
    sample: tuple[int, TreeNode, float], save_dir: Path, save_name: str = ""
) -> None:
    """Takes input of get_sample of PureMcmcData and ,
    Plot a tree and save it to a file with just its iteration
    number and log probability,
    in the savename."""

    print_options = dict()
    print_options["title"] = False
    print_options["data_tree"] = dict()
    print_options["data_tree"]["log-likelihood"] = True

    # get the iteration number
    i = sample[0]
    # get the tree
    tree = sample[1]
    tree.data = dict()
    tree.data["log-likelihood"] = sample[2]
    # get the log probability, and round to 2 decimal places
    log_prob = sample[2]
    log_prob = round(log_prob, 2)
    # make save name from iteration number
    if save_name == "":
        save_name = "iter_" + str(i) + "_log_prob_" + str(log_prob)

    # plot tree
    plot_tree(tree, save_name, save_dir, print_options)


def plot_tree_no_print(tree: TreeNode, save_name: str, save_dir: Path) -> None:
    """Takes input of get_sample of PureMcmcData and ,
    Plot a tree and save it to a file with just its iteration
     number and log probability,
    in the savename."""

    print_options = dict()
    print_options["title"] = False
    print_options["data_tree"] = dict()

    # get the iteration number

    tree.data = dict()

    # plot tree
    plot_tree(tree, save_name, save_dir, print_options)
