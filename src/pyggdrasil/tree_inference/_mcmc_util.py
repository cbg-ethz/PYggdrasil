"""Utility Functions for _mcmc.py
"""
from jax import Array
import jax.numpy as jnp
import jax

from pyggdrasil.tree_inference._mcmc import Tree  # type: ignore


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
    # get corresponding root label
    root_label = tree.labels[root_idx]

    return root_label


def _prune(tree: Tree, parent: int) -> tuple[Tree, Tree]:
    """Prune subtree, by cutting edge leading to node parent
    to obtain subtree of descendants desc and the remaining tree.

    Note: may return subtrees/remaining tree with root not at the
            last index of the adjacency matrix

    Args:
        tree : Tree
             tree to prune from
        parent : int
             label of root node of subtree to prune
    Returns:
        tuple of [remaining tree, subtree]
    """
    # get subtree labels
    subtree_labels = _get_descendants(
        tree.tree_topology, tree.labels, parent, includeParent=True
    )
    # get subtree indices - assumes labels of tree and subtree are in the sane order
    subtree_idx = jnp.where(jnp.isin(tree.labels, subtree_labels))[0].tolist()
    # get subtree adjacency matrix
    subtree_adj = tree.tree_topology[subtree_idx, :][:, subtree_idx]
    subtree = Tree(subtree_adj, subtree_labels)

    # get remaining tree labels
    remaining_idx = jnp.where(~jnp.isin(tree.labels, subtree_labels))[0]
    # get remaining tree adjacency matrix
    remaining_adj = tree.tree_topology[remaining_idx, :][:, remaining_idx]
    # get remaining tree labels
    remaining_labels = tree.labels[remaining_idx]
    # get remaining tree
    remaining_tree = Tree(remaining_adj, remaining_labels)

    return (subtree, remaining_tree)


def _reattach(tree: Tree, subtree: Tree, node: int) -> Tree:
    """Reattach subtree to tree, by adding edge between parent and child.

    Args:
        tree : Tree
             tree to reattach to
        subtree : Tree
             subtree to reattach
        node : int
             label of node to attach subtree to
      Returns:
         tree with subtree reattached
    """
    raise NotImplementedError
