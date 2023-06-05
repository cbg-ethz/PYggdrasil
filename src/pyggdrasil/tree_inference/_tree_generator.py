"""This submodule defines functions to generate trees.

Note: this is part of the private API.
"""
import jax

import numpy as np

from pyggdrasil.tree_inference import JAXRandomKey


def generate_star_tree(n_nodes: int) -> np.ndarray:
    """Generate a star tree of n_nodes nodes
       Root is the highest index node.

    Args:
        n_nodes: Number of nodes, including root, in the tree.

    Returns:
        Adjacency matrix of a star tree with n_nodes nodes.
        A tree with n_nodes nodes, each node descending from the root.
        I.e. matrix with last row all ones.
    """
    tree = np.zeros((n_nodes, n_nodes), dtype=int)
    tree[-1, :] = 1

    return tree


def generate_deep_tree(rng: JAXRandomKey, n_nodes: int) -> np.ndarray:
    """Generate a deep tree of n_nodes nodes.

    Args:
        rng: JAX random key.
        n_nodes: Number of nodes in the tree.

    Returns:
        Adjacency matrix of a deep tree with n_nodes nodes.
    """
    tree = np.zeros((n_nodes, n_nodes), dtype=int)

    # generate a random permutation of the nodes
    nodes_except_root = n_nodes - 1
    permutation = jax.random.permutation(rng, nodes_except_root)

    # add edges from root to each node in the permutation
    root = n_nodes - 1
    parent = root
    for child in permutation:
        tree[parent, child] = 1
        parent = child

    return tree
