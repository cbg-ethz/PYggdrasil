"""Methods for visualizing trees."""

from pathlib import Path
import jax.numpy as jnp
from anytree.exporter import DotExporter
import os

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

    tree.print_topo()

    print(print_options)

    fullpath = os.path.join(save_dir, save_name)
    # make output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Assuming 'tree' is an instance of the TreeNode class

    # DotExporter(tree, graph_attr=graph_attr).to_picture(fullpath + ".svg")

    DotExporter(tree).to_dotfile(fullpath + ".dot")

    # dotTree.graph_attr['label'] = 'My Graph Label'
    # Add the label argument to the graph

    # dotTree.render(fullpath + ".svg", view=True)  # Render the graph as a SVG file


################################################################################
if __name__ == "__main__":
    adj_mat = jnp.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]])
    labels = jnp.array([0, 1, 2, 3])

    simple_tree = Tree(adj_mat, labels)

    root = simple_tree.to_TreeNode()

    print_options = dict()
    print_options["title"] = True
    print_options["data_tree"] = dict()
    print_options["data_tree"]["log-likelihood"] = True
    print_options["data_node"] = dict()
    # empty dict means print nothing - of node data

    plot(root, "test02", Path("../../../data/trees/"), print_options)
