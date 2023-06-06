"""This submodule defines functions to generate trees.

All adjacency matrices are assumed to have the highest index node as the root.
All Adjacency matrices are assumed to be directed, and to have no self-connections.

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
        I.e. matrix with last row all ones, but the last as root.
        Nodes are not self-connected.
    """
    tree = np.zeros((n_nodes, n_nodes), dtype=int)
    tree[-1, :] = 1
    tree[-1, -1] = 0  # root is not self-connected

    return tree


def generate_deep_tree(rng: JAXRandomKey, n_nodes: int) -> np.ndarray:
    """Generate a deep tree of n_nodes nodes.

    Args:
        rng: JAX random key.
        n_nodes: Number of nodes in the tree.

    Returns:
        Adjacency matrix of a deep tree with n_nodes nodes.
        Root is the highest index node.
        Nodes are not self-connected.
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


def generate_random_tree(rng: JAXRandomKey, n_nodes: int) -> np.ndarray:
    """
    Generates a random tree with n nodes, where the root is the highest index node.
    Nodes are not self-connected.

    Args:
        rng: JAX random number generator
        n_nodes: int number of nodes in the tree

    Returns:
        adj_matrix: np.ndarray
            adjacency matrix: adj_matrix[i, j] means an edge "i->j"
            Note 1: nodes are here not self-connected
            Note 2: the root is the last node
    """
    # Generate a random tree
    adj_matrix = _generate_random_tree(rng, n_nodes)
    # Adjust the node order to convention
    adj_matrix = _reverse_node_order(adj_matrix)

    return adj_matrix


def _generate_random_tree(rng: JAXRandomKey, n_nodes: int) -> np.ndarray:
    """
    Generates a random tree with n nodes, where the root is the first node.

    Args:
        rng: JAX random number generator
        n_nodes: int number of nodes in the tree

    Returns:
        adj_matrix: np.ndarray
            adjacency matrix: adj_matrix[i, j] means an edge "i->j"
            Note 1: nodes are here not self-connected
            Note 2: the root is the first node
    """
    # Initialize adjacency matrix with zeros
    adj_matrix = np.zeros((n_nodes, n_nodes))
    # Generate random edges for the tree
    for i in range(1, n_nodes):
        # Select a random parent node from previously added nodes
        parent = jax.random.choice(rng, i)
        # Add an edge from the parent to the current node
        adj_matrix[parent, i] = 1
    # Return the adjacency matrix
    return adj_matrix


def _reverse_node_order(adj_matrix: np.ndarray) -> np.ndarray:
    """
    Reverses the order of the nodes in the tree adjacency matrix.

    Args:
        adj_matrix: np.ndarray
            adjacency matrix

    Returns:
        adj_matrix: np.ndarray
            adjacency matrix
    """
    # Reverse the order of the nodes
    adj_matrix = adj_matrix[::-1, ::-1]
    # Return the adjacency matrix
    return adj_matrix