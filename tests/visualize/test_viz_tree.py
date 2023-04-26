"""Tests the visualization functions for trees."""
# _tree.py

import jax.numpy as jnp
import os
import pytest

from pyggdrasil.tree_inference._tree import Tree
import pyggdrasil.visualize.tree as viz

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Skipping this test in GitHub Actions.")
def test_plot(tmp_path):
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
    save_dir = tmp_path / "trees"
    save_dir.mkdir()
    save_name = "unit_test_tree"

    check_dir = save_dir
    fullpath = os.path.join(check_dir, save_name + ".svg")

    # delete file if it exists
    if os.path.isfile(fullpath):
        os.remove(fullpath)
    # plot and save action

    viz.plot(root, save_name, save_dir, print_options)

    # check if file exists
    assert os.path.isfile(fullpath)
