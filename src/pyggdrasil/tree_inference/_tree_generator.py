"""This submodule defines functions to generate trees.

Note: this is part of the private API.
"""
import jax

import jax.numpy as jnp


def generate_star_tree(n_nodes: int) -> jax.Array:
    """Generate a star tree of n_nodes nodes
       Root is the highest index node.

    Args:
        n_nodes: Number of nodes, including root, in the tree.

    Returns:
        Adjacency matrix of a star tree with n_nodes nodes.
        A tree with n_nodes nodes, each node descending from the root.
        I.e. matrix with last row all ones.
    """
    tree = jnp.zeros((n_nodes, n_nodes), dtype=int)
    tree[-1, :] = 1

    return tree
