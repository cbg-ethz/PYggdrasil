"""Utility Functions for _mcmc.py
"""
import jax.numpy as jnp
import jax.scipy as jsp

from pyggdrasil.tree_inference._tree import Tree
import pyggdrasil.tree_inference._tree as tr


def _prune(tree: Tree, pruned_node: int) -> tuple[Tree, Tree]:
    """Prune subtree, by cutting edge leading to node parent
    to obtain subtree of descendants desc and the remaining tree.

    Note: may return subtrees/remaining tree with root not at the
            last index of the adjacency matrix

    Args:
        tree : Tree
             tree to prune from
        pruned_node : int
             label of root node of subtree to prune
    Returns:
        tuple of [remaining tree, subtree]
    """
    # get subtree labels
    subtree_labels = tr._get_descendants(
        tree.tree_topology, tree.labels, pruned_node, includeParent=True
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


def _reattach(tree: Tree, subtree: Tree, attach_to: int, pruned_node: int) -> Tree:
    """Reattach subtree to tree, by adding edge between parent and child.

    Args:
        tree : Tree
             tree to reattach to
        subtree : Tree
             subtree to reattach
        attach_to : int
             label of node to attach subtree to
        pruned_node : int
              label of root node of subtree
      Returns:
         tree with subtree reattached, via a connection from parent to child
    """
    # get root index label of subtree
    child_idx = jnp.where(subtree.labels == pruned_node)[0]
    # get root index label of tree
    parent_idx = jnp.where(tree.labels == attach_to)[0]

    new_tree_adj = jsp.linalg.block_diag(tree.tree_topology, subtree.tree_topology)
    new_tree_adj = new_tree_adj.at[parent_idx, tree.labels.shape[0] + child_idx].set(1)

    return Tree(new_tree_adj, jnp.append(tree.labels, subtree.labels))
