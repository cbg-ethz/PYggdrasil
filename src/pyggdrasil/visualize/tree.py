"""Methods for visualizing trees."""

from pathlib import Path
import jax.numpy as jnp
from anytree.exporter import DotExporter
import pydot
import os
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

import networkx as nx

from pyggdrasil.tree_inference._tree import Tree


from pyggdrasil import TreeNode


def plot(tree: TreeNode, save_name: str, save_dir: Path, print_options: dict) -> None:
    """Plot a tree.

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
                "data_node": dict
                    what attributes to print of the nodes
    Returns:
        None
    """

    # tree.print_topo()

    print(print_options)

    fullpath = os.path.join(save_dir, save_name)
    # make output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # DotExporter(tree).to_dotfile(fullpath + ".dot")

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
    plt.ioff
    fig = plt.figure()
    plt.rcParams["text.usetex"] = True

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
        fig.suptitle(tree.data["tree-name"], fontsize=15)

    description = ""
    if "log-likelihood" in print_options["data_tree"]:
        description += (
            r"$\log(P(D|T,\theta))$: " + str(tree.data["log-likelihood"]) + "\n"
        )

    plt.text(0.7, 0.9, description, dict(size=15), transform=ax1.transAxes)

    plt.savefig(fullpath + ".svg", bbox_inches="tight")

    # Parse the DOT string into a Graphviz Source object
    # source = graphviz.Source(dot_string)

    # Set graph attributes
    # gv_graph.graph_attr(label='My Graph Title')
    # gv_graph.graph_attr['rankdir'] = 'TB'

    # Create a subgraph for the text
    # with graph.subgraph() as sub:
    #      # Set subgraph attributes
    #      sub.attr(rank='same')
    #      sub.attr(label='')
    #
    #      # Add text to the subgraph
    #      my_var = 'log_prob'  # your variable containing the text
    #      sub.node('text', label=my_var, shape='plaintext', pos='r', fontsize='20')

    # gv_graph.render(fullpath + ".svg")  # Render the graph as a SVG file


################################################################################
if __name__ == "__main__":
    adj_mat = jnp.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]])
    labels = jnp.array([0, 1, 2, 3])

    simple_tree = Tree(adj_mat, labels)

    root = simple_tree.to_TreeNode()
    root.data = dict()
    root.data["log-likelihood"] = -4.3
    root.data["tree-name"] = "Maximum Likelihood Tree"

    print_options = dict()
    print_options["title"] = True
    print_options["data_tree"] = dict()
    print_options["data_tree"]["log-likelihood"] = True
    print_options["data_node"] = dict()
    # empty dict means print nothing - of node data

    plot(root, "test10", Path("../../../data/trees/"), print_options)
