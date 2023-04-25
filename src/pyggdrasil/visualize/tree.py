"""Methods for visualizing trees."""

import os
from pathlib import Path

from anytree.exporter import DotExporter
import pydot
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import networkx as nx


from pyggdrasil import TreeNode


def plot(tree: TreeNode, save_name: str, save_dir: Path, print_options: dict) -> None:
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
    graph = graphs[0]
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

    # plt.rcParams["text.usetex"] = True

    # relabel Root node
    mapping = {str(tree.name): "R"}
    nx_graph = nx.relabel_nodes(nx_graph, mapping)

    # top down tree layout
    pos = graphviz_layout(nx_graph, prog="dot")
    # as subplots
    ax1 = fig.add_subplot()

    # plot graph
    nx.draw(
        nx_graph,
        with_labels=True,
        ax=ax1,
        pos=pos,
        node_color="w",
        edgecolors="k",
        node_size=1000,
        font_size=20,
        font_weight="bold",
    )

    # make title
    if print_options["title"]:
        fig.suptitle(tree.data["tree-name"], fontsize=20)

    description = ""
    for detail in print_options["data_tree"]:
        if detail == "log-likelihood":
            description += (
                r"$\log(P(D|T,\theta)$: " + str(tree.data["log-likelihood"]) + "\n"
            )
        else:
            description += detail + ": " + str(tree.data[detail]) + "\n"

    plt.text(1, 0.97, description, dict(size=15), transform=ax1.transAxes, va="top")

    plt.savefig(fullpath + ".svg", bbox_inches="tight")
    plt.close()
