"""This submodule defines our data format for trees, for use in MCMC,
and provides some utility functions.

Note: this is part of the private API.
"""

import jax
import dataclasses
import numpy as np
import jax.numpy as jnp
from jax import Array

from pyggdrasil.tree import TreeNode
import pyggdrasil.tree_inference as tree_inf


@dataclasses.dataclass(frozen=True)
class Tree:
    """For ``N`` mutations we use a tree with ``N+1`` nodes,
    where the nodes at positions ``0, ..., N-1`` are "blank"
    and can be mapped to any of the mutations.
    The node ``N`` is the root node and should always be mapped
    to the wild type.

    Attrs:
        tree_topology: the topology of the tree
          encoded in the adjacency matrix.
          No self-loops, i.e. diagonal is all zeros.
          Shape ``(N+1, N+1)``
        labels: maps nodes in the tree topology
          to the actual mutations.
          Note: the last position always maps to itself,
          as it's the root, and we use the convention
          that root has the largest index.
          Shape ``(N+1,)``
    """

    tree_topology: jax.Array
    labels: jax.Array

    def to_TreeNode(self) -> TreeNode:
        """Converts this Tree to a TreeNode.
        Returns the root node of the tree."""

        root = None
        # check that the last node is the root
        root_label = _get_root_label(self)
        if root_label != self.labels[-1]:
            print("Root was not the last node in the adjacency matrix.")
            reorder_tree = _resort_root_to_end(self, root_label)
            print("Tree has been reordered - placing the root at the end.")
            print("This does not change the Tree object.")
            root = tree_inf.adjacency_to_root_dfs(
                adj_matrix=np.array(reorder_tree.tree_topology),
                labels=np.array(reorder_tree.labels),
            )
        else:
            root = tree_inf.adjacency_to_root_dfs(
                adj_matrix=np.array(self.tree_topology), labels=np.array(self.labels)
            )
        return root

    def print_topo(self):
        """Prints the tree in a human-readable format."""
        return self.to_TreeNode().print_topo()


def _resort_root_to_end(tree: Tree, root: int) -> Tree:
    """Resorts tree so that root is at the end of the adjacency matrix.

    Args:
        root: int
            root label of the tree
        tree: Tree
            tree to resort

    Returns:
        tree: Tree
    """
    # get root index
    root_idx = int(jnp.where(tree.labels == root)[0])
    # get all nodes which are not root
    non_root_idx = jnp.where(tree.labels != root)[0]
    # get new reduced adjacency matrix
    reduced_adj = tree.tree_topology[non_root_idx, :][:, non_root_idx]
    # resort row of root, so that it is at the end - excluding root
    root_row = tree.tree_topology[root_idx, non_root_idx]
    # resort column of root, so that it is at the end
    root_col = jnp.append(
        tree.tree_topology[non_root_idx, root_idx],
        tree.tree_topology[root_idx, root_idx],
    )
    # get new adjacency matrix
    # TODO: use matrix assignment not append
    new_adj = jnp.append(reduced_adj, jnp.array([root_row]), axis=0)
    new_adj = jnp.append(new_adj, jnp.swapaxes(jnp.array([root_col]), 0, 1), axis=1)
    # get new labels
    new_labels = jnp.append(tree.labels[non_root_idx], tree.labels[root_idx])
    # get new tree
    resorted_tree = Tree(new_adj, new_labels)

    return resorted_tree


def _get_descendants(
    adj_matrix: Array, labels: Array, parent: int, includeParent: bool = False
) -> Array:
    """
    Returns a list of labels representing the descendants of node parent.
    Used boolean matrix exponentiation to find descendants.

    Complexity:
        Naive: O(n^3 * (n-1)) where n is the number of nodes in the tree including root.
        TODO: - Consider implementing 'Exponentiation by Squaring Algorithm'
                    for  O(n^3 * log(m)
              - fix conditional exponentiation for exponent < n-1
    Args:
    - tree (Tree):  a Tree object
    - parent: an integer representing
        the label of the node whose descendants we want to find

    Returns:
    - a JAX array of integers representing the labels of the descendants of node parent
      in order of nodes in the adjacency matrix, i.e. the order of the labels
      if includeParent is True, the parent is included in the list of descendants
    """
    # get number of nodes
    n = adj_matrix.shape[0]
    # get ancestor matrix
    ancestor_mat = _get_ancestor_matrix(adj_matrix, n)
    # get index of parent
    parent_idx = int(jnp.where(labels == parent)[0])
    # get descendants
    desc = jnp.where(ancestor_mat[parent_idx, :])[0]
    # get labels correspond to indices
    desc_labels = labels[desc]
    # remove parent - as self-looped
    if not includeParent:
        desc_labels = desc_labels[desc_labels != parent]
    return desc_labels


def _expon_adj_mat(adj_matrix: Array, exp: int, cond: bool = False):
    """Exponentiation of adjacency matrix.

    Complexity: O(n^3 * m) where n is the size of the square matrix and m the exponent
            if cond= True the 'm' < n-1 hence speedup

    TODO: Consider implementing 'Exponentiation by Squaring Algorithm'
            for  O(n^3 * log(m)

    Args:
        adj_matrix : adjacency matrix
        exp: exponent
        cond: if to use conditional exponentiation,
                stop if matrix does not change anymore
    """
    bool_mat = jnp.where(adj_matrix == 1, True, False)
    adj_matrix_exp = bool_mat
    if not cond:

        def body(carry, _):
            return jnp.dot(carry, bool_mat), None

        adj_matrix_exp = jax.lax.scan(body, bool_mat, jnp.arange(exp))[0]
    elif cond:
        # TODO: fix this - this is not working
        raise NotImplementedError("Conditional exponentiation not implemented yet.")

        exp_counter = 0

        def loop_cond_fn(carry):
            prev_matrix, curr_matrix = carry
            nonlocal exp_counter
            exp_counter = exp_counter + 1
            return ~(jnp.array_equal(prev_matrix, curr_matrix)) and (exp_counter <= exp)

        def loop_body_fn(carry):
            prev_matrix, curr_matrix = carry
            new_matrix = jnp.dot(curr_matrix, bool_mat)
            return curr_matrix, new_matrix

        (_, adj_matrix_exp), _ = jax.lax.while_loop(
            loop_cond_fn, loop_body_fn, (bool_mat, bool_mat)
        )

    return adj_matrix_exp


def _get_ancestor_matrix(adj_matrix: Array, n: int):
    """Returns the ancestor matrix.

    Complexity: O(n^3 * (n-1)) where n is the number
            of nodes in the tree including root.

    Args:
        adj_matrix: adjacency matrix
        n: number of nodes in the tree including root
    Returns:
        ancestor_matrix: boolean matrix where the (i,j)
        entry is True if node i is an ancestor of node j.
    """

    # ensure is jax array
    adj_matrix = jnp.array(adj_matrix)
    # get adjacency matrix
    n = adj_matrix.shape[0]
    # add self-loops
    diag_idx = jnp.diag_indices_from(adj_matrix)
    adj_matrix = adj_matrix.at[diag_idx].set(1)
    # boolean matrix exponentiation
    ancestor_matrix = _expon_adj_mat(adj_matrix, n - 1)
    return ancestor_matrix


def _get_root_label(tree: Tree) -> int:
    """Returns the root label of a tree

    Args:
        tree: Tree
            tree to get root label of

    Returns:
        root_label: int
            root label of the tree
    """
    # get ancestor matrix of tree
    ancestor_matrix = _get_ancestor_matrix(tree.tree_topology, tree.labels.shape[0])
    # find row which has all ones in ancestor_matrix
    root_idx = jnp.where(jnp.all(ancestor_matrix == 1, axis=1))[0]
    if len(root_idx) > 1:
        raise ValueError("More than one root found - not a tree")
    # get root label
    root_label = int(tree.labels[root_idx])

    return root_label


def _reorder_tree(tree: Tree, from_labels, to_labels):
    """Reorders tree from current labels to new labels

    Args:
        tree: Tree
            tree to reorder
        from_labels: Array
            current labels of tree
        to_labels: Array
            new labels of tree
    Returns:
        reordered_tree: Tree

    """
    size = tree.tree_topology.shape[0]
    new_adj = jnp.zeros((size, size))
    for row in range(size):
        prior__row_label = from_labels[row]
        new_row_idx = int(jnp.where(to_labels == prior__row_label)[0])
        for column in range(size):
            prior_col_label = from_labels[column]
            new_col_idx = int(jnp.where(to_labels == prior_col_label)[0])
            value = tree.tree_topology[row, column]
            new_adj = new_adj.at[new_row_idx, new_col_idx].set(value)

    reordered_tree = Tree(tree_topology=new_adj, labels=to_labels)
    return reordered_tree
