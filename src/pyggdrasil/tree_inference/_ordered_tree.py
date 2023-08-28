"""This submodule defines our data format for ordered trees, for use in MCMC kernel.

This subclas of the Tree class ensures that the tree is ordered,
forces the usage or ordered Trees in the mcmc kernel.

Note: this is part of the private API.
"""

import jax
import jax.numpy as jnp

from pyggdrasil.tree_inference._tree import Tree, _reorder_tree


class OrderedTree(Tree):
    """As Tree, but with the additional requirement that the tree is ordered, i.e.
        labels are fixed to ``0,... ,N``, where ``N`` is the number of mutations.

        Uses the same assignment as Tree, even though labels are fixed.
        Performs a check that the labels are ordered.

    For ``N`` mutations we use a tree with ``N+1`` nodes,
    where the nodes at positions ``0, ..., N-1`` are "blank"
    and can be mapped to any of the mutations.
    The node ``N`` is the root node and should always be mapped
    to the wild type.

    Attrs:
        tree_topology: the topology of the tree
          encoded in the adjacency matrix.
          No self-loops, i.e. diagonal is all zeros.
          Shape ``(N+1, N+1)``
        labels: Fixed to ``0,...,N`` - maps nodes in the tree topology
          to the actual mutations.
          Note: the last position always maps to itself,
          as it's the root, and we use the convention
          that root has the largest index.
          Shape ``(N+1,)``
    """

    tree_topology: jax.Array
    labels: jax.Array

    def __init__(self, tree_topology: jax.Array, labels: jax.Array):
        """Initialize the OrderedTree. Performs some checks on the labels."""
        # check that labels are ordered
        assert jnp.all(labels == jnp.arange(labels.shape[0]))
        super().__init__(tree_topology, labels)

    @staticmethod
    def from_tree(tree: Tree) -> "OrderedTree":
        """Converts a Tree to an OrderedTree.

        Args:
            tree: Tree to convert to OrderedTree

        Returns:
            OrderedTree
        """
        # reorder tree
        ordered_tree = _reorder_tree(tree, jnp.arange(tree.labels.shape[0]))
        # return ordered tree type
        return OrderedTree(ordered_tree.tree_topology, ordered_tree.labels)
