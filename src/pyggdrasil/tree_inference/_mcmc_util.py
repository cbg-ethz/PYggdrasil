"""Utility Functions for _mcmc.py
"""
from jax import Array
import jax.numpy as jnp
import jax

from pyggdrasil.tree_inference._mcmc import Tree  # type: ignore


def _get_descendants(adj_matrix: Array, labels: Array, parent: int) -> Array:
    """
    Returns a list of indices representing the descendants of node parent.
    Used boolean matrix exponentiation to find descendants.

    Args:
    - tree (Tree):  a Tree object
    - parent: an integer representing
        the index of the node whose descendants we want to find

    Returns:
    - a JAX array of integers representing the indices of the descendants of node parent
    """
    # get adjacency matrix
    n = adj_matrix.shape[0]
    # ass self-loops
    diag_idx = jnp.diag_indices_from(adj_matrix)
    adj_matrix = adj_matrix.at[diag_idx].set(1)
    # boolean matrix exponentiation

    adj_matrix_exp = _expon_adj_mat(adj_matrix, n - 1)

    # bool_mat = jnp.where(adj_matrix == 1, True, False)
    #
    # def body(carry, _):
    #     return jnp.dot(carry, bool_mat), None
    #
    # adj_matrix_exp = jax.lax.scan(body, bool_mat, jnp.arange(n - 1))[0]
    # get descendants
    desc = jnp.where(adj_matrix_exp[parent, :])[0]
    # get labels
    desc_labels = labels[desc]
    # remove parent - as self-looped
    desc_labels = desc_labels[desc_labels != parent]
    return desc_labels


def _expon_adj_mat(adj_matrix: Array, exp: int, cond: bool = False):
    """Exponentiation of adjacency matrix.

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
        exp_counter = 0

        def loop_cond_fn(carry):
            prev_matrix, curr_matrix = carry
            nonlocal exp_counter
            exp_counter = exp_counter + 1
            return ~(jnp.array_equal(prev_matrix, curr_matrix)) or (exp_counter <= exp)

        def loop_body_fn(carry):
            prev_matrix, curr_matrix = carry
            new_matrix = jnp.dot(curr_matrix, bool_mat)
            return curr_matrix, new_matrix

        (_, adj_matrix_exp), _ = jax.lax.while_loop(
            loop_cond_fn,
            loop_body_fn,
            (jnp.zeros_like(bool_mat), jnp.zeros_like(bool_mat)),
        )

    return adj_matrix_exp


def _prune(tree: Tree, parent: int) -> tuple[Tree, Tree]:
    """Prune subtree, by cutting edge leading to node parent
    to obtain subtree of descendants desc and the remaining tree.

    Args:
        tree : Tree
             tree to prune from
        parent : int
             label of root node of subtree to prune
    Returns:
        tuple of [remaining tree, subtree]
    """

    raise NotImplementedError


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
