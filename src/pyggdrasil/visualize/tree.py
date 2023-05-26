"""Methods for visualizing trees."""

import os
from pathlib import Path

import numpy as np
from anytree.exporter import DotExporter
import pydot
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import networkx as nx
import logging
from typing import Union, Optional

from pyggdrasil import TreeNode


NodeLabel = Union[str, int, float]


def plot(
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

    # plot
    fig = plt.figure()

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
    print(node_sizes)

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

    plt.savefig(fullpath + ".svg", bbox_inches="tight")
    plt.close()
