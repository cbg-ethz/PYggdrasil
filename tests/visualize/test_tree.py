"""Tests the visualization functions for trees."""
# _tree.py


import jax.numpy as jnp
import os
from pathlib import Path

from pyggdrasil.tree_inference._tree import Tree

import pyggdrasil.visualize.tree as viz


def test_plot():
    """Test plot_tree. - check for output."""
    adj_mat = jnp.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
        ]
    )
    labels = jnp.array([0, 1, 2, 3, 4])

    simple_tree = Tree(adj_mat, labels)

    root = simple_tree.to_TreeNode()
    root.data = dict()
    root.data["log-likelihood"] = -4.3
    root.data["Data type"] = "sim data"
    root.data["Run"] = "3"
    root.data["tree-name"] = "Maximum Likelihood Tree"

    print_options = dict()
    print_options["title"] = True
    print_options["data_tree"] = dict()
    print_options["data_tree"]["log-likelihood"] = True
    print_options["data_tree"]["Data type"] = False
    print_options["data_tree"]["Run"] = False

    # make full path
    save_dir = Path("../data/trees/")
    save_name = "unit_test_tree"

    check_dir = Path("../data/trees/")
    fullpath = os.path.join(check_dir, save_name + ".svg")

    # delete file if it exists
    if os.path.isfile(fullpath):
        os.remove(fullpath)
    # plot and save action

    viz.plot(root, save_name, save_dir, print_options)

    # check if file exists
    assert os.path.isfile(fullpath)
