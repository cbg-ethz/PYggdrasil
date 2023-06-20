"""Tests of the tree class / utility functions used for the MCMC"""
# _tree.py

import pytest
import jax.random as random
import jax.numpy as jnp
import numpy as np
import json


import pyggdrasil.tree_inference._tree as tr
import pyggdrasil.tree_inference._tree_generator as tree_gen
import pyggdrasil.tree_inference as tree_inf
import pyggdrasil.serialize as serialize

from pyggdrasil.tree_inference._tree import Tree

import logging

LOGGER = logging.getLogger(__name__)


@pytest.mark.parametrize("seed", [42, 32, 44])
@pytest.mark.parametrize("n_nodes", [5, 10, 15])
def test_get_descendants(seed: int, n_nodes: int):
    """Test get_descendants against Floyd-Warshall algorithm."""
    # get random numbers keys
    rng = random.PRNGKey(seed)
    rng_tree, rng_nodes, rng_labels = random.split(rng, 3)
    # generate random tree
    adj_mat = jnp.array(tree_gen._generate_random_tree_adj_mat(rng_tree, n_nodes))
    # generate random nodes
    parent = int(random.randint(rng_nodes, shape=(1,), minval=0, maxval=n_nodes)[0])
    # assign labels - randomly sample
    labels = random.permutation(rng_labels, np.arange(n_nodes))
    # make tree
    tree = Tree(adj_mat, labels)  # TODO: make labels random
    # get descendants with matrix exponentiation
    desc01 = tr.get_descendants(
        jnp.array(tree.tree_topology), jnp.array(tree.labels), parent
    )
    # get descendants with Floyd-Warshall
    adj_mat_prime = np.array(tree.tree_topology)
    np.fill_diagonal(adj_mat_prime, 1)
    parent_idx = np.where(tree.labels == parent)[0][0]
    desc02_idx = np.where(tree_inf.get_descendants_fw(adj_mat_prime, parent_idx) == 1)
    desc02 = tree.labels[desc02_idx]
    desc02 = desc02[desc02 != parent]

    # check that descendants are the same
    assert jnp.all(desc01 == desc02)


def test_get_root_label():
    """Test get_root_label. - manual test."""
    adj_mat = jnp.array(
        [
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0],
        ]
    )
    labels = jnp.array([3, 1, 2, 4, 5, 6, 0, 7])

    tree = Tree(adj_mat, labels)

    root_label_test = tr._get_root_label(tree)

    root_label_true = 7

    assert jnp.all(root_label_test == root_label_true)


def test_resort_root_to_end():
    """Test resort_root_to_end. - manual test."""
    adj_mat = jnp.array([[1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 1], [0, 0, 0, 1]])
    labels = jnp.array([1, 4, 2, 3])
    tree = Tree(adj_mat, labels)
    resort_tree = tr._resort_root_to_end(tree, 4)

    assert jnp.all(
        resort_tree.tree_topology
        == jnp.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 1]])
    )
    assert jnp.all(resort_tree.labels == jnp.array([1, 2, 3, 4]))


def test_tree_node_to_tree():
    """Test _tree_from_node_to_tree. - manual test."""
    # load tree
    json_str = """
    {"name": 9,
    "data": null,
    "children": [
                {"name": 5,
                "data": null,
                "children": [
                            {"name": 7,
                             "data": null,
                            "children": []}, 
                            {"name": 0,
                            "data": null,
                            "children": [
                                            {"name": 8,
                                            "data": null,
                                            "children": []},
                                            {"name": 4,
                                            "data": null,
                                            "children": [
                                                            {"name": 2,
                                                            "data": null,
                                                            "children": []},
                                                            {"name": 1,
                                                            "data": null,
                                                            "children": []}]},
                                            {"name": 6,
                                            "data": null,
                                            "children": [
                                                        {"name": 3,
                                                        "data": null,
                                                        "children": []}]}]}]}]}"""
    json_obj = json.loads(json_str)
    tree_node = serialize.deserialize_tree_from_dict(
        json_obj, deserialize_data=lambda x: x
    )
    # convert to tree
    tree = tr.tree_from_tree_node(tree_node)

    # check that tree is correct
    assert jnp.all(
        tree.tree_topology
        == jnp.array(
            [
                [0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )
    )


def test_is_same_tree():
    """Test is_same_tree function.
    same tree with different label order
    """

    adj_mat1 = jnp.array(
        [
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0],
        ]
    )
    labels1 = jnp.array([3, 1, 2, 4, 5, 6, 0, 7])
    tree1 = Tree(adj_mat1, labels1)

    adj_mat2 = jnp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
        ]
    )
    labels2 = jnp.array([0, 1, 2, 3, 4, 5, 6, 7])
    tree2 = Tree(adj_mat2, labels2)

    assert tr.is_same_tree(tree1, tree2)


def test_is_not_same_tree():
    """Test is_same_tree function.
    same tree with different label order
    """

    adj_mat1 = jnp.array(
        [
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0],
        ]
    )
    labels1 = jnp.array([3, 1, 2, 4, 5, 6, 0, 7])
    tree1 = Tree(adj_mat1, labels1)

    adj_mat2 = jnp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
        ]
    )
    labels2 = jnp.array([0, 1, 2, 5, 4, 3, 6, 7])
    tree2 = Tree(adj_mat2, labels2)

    assert tr.is_same_tree(tree1, tree2) is False


def make_tree(n: int, seed: int, tree_type: str) -> Tree:
    """Make a tree for testing."""

    rng = random.PRNGKey(seed)

    if tree_type == "r":
        adj_mat = jnp.array(tree_gen._generate_random_tree_adj_mat(rng, n))
        labels = jnp.arange(n)
        return Tree(adj_mat, labels)
    elif tree_type == "s":
        adj_mat = jnp.array(tree_gen._generate_star_tree_adj_mat(n))
        labels = jnp.arange(n)
        return Tree(adj_mat, labels)
    else:
        adj_mat = jnp.array(tree_gen._generate_deep_tree_adj_mat(rng, n))
        labels = jnp.arange(n)
        return Tree(adj_mat, labels)


@pytest.mark.parametrize("n", [5, 10])
@pytest.mark.parametrize("seed", [34, 424])
@pytest.mark.parametrize("tree_type", ["r", "s", "d"])
def test_is_valid_tree(n: int, seed: int, tree_type: str):
    """Test is_valid_tree function."""

    tree = make_tree(n, seed, tree_type)
    assert tr.is_valid_tree(tree) is True
