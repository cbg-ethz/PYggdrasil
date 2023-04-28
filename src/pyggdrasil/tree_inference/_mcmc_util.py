"""Utility Functions for _mcmc.py
"""
import jax.numpy as jnp
import jax.scipy as jsp
import xarray as xr

from pyggdrasil.tree_inference._tree import Tree
import pyggdrasil.tree_inference._tree as tr
from pyggdrasil.tree_inference._interface import JAXRandomKey, Array


# TODO: consider moving all these 3 function to tests only
def _prune(tree: Tree, pruned_node: int) -> tuple[Tree, Tree]:
    """Prune subtree, by cutting edge leading to node parent
    to obtain subtree of descendants desc and the remaining tree.

    LEGACY CODE - Only for visualization/testing purposes

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

    LEGACY CODE - Only for visualization/testing purposes

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


def _prune_and_reattach_move(tree: Tree, *, pruned_node: int, attach_to: int) -> Tree:
    """Prune a node from tree topology and attach it to another one.

    LEGACY CODE - Only for visualization/testing purposes

    Returns:
        new tree, with node ``pruned_node`` pruned and reattached to ``attach_to``.

    Note:
        This is a *pure function*, i.e., the original ``tree`` should not change.
    """
    # Prune Step
    subtree, remaining_tree = _prune(tree=tree, pruned_node=pruned_node)
    # Reattach Step
    new_tree = _reattach(
        tree=remaining_tree,
        subtree=subtree,
        attach_to=attach_to,
        pruned_node=pruned_node,
    )
    return new_tree


def _pack_sample(
    iteration: int, tree: Tree, logprobability: float, rng_key_run: JAXRandomKey
) -> xr.Dataset:
    """Pack MCMC sample to xarray to be dumped."""
    adj_mat: (Array) = tree.tree_topology
    labels: (Array) = tree.labels

    tree_xr = xr.DataArray(
        adj_mat,
        dims=("from_node_k", "to_node_k"),
        coords={"from_node_k": labels, "to_node_k": labels},
    )

    rng_key_run_ls = jnp.asarray(rng_key_run)

    data_vars = {
        "iteration": iteration,
        "tree": tree_xr,
        "log-probability": logprobability,
        "rng_key_run": rng_key_run_ls,
    }

    ds = xr.Dataset(data_vars=data_vars)

    return ds
