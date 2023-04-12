"""Markov Chain Monte Carlo inference for mutation trees according to the SCITE model.

Note:
    This implementation assumes that the false positive
    and false negative rates are known and provided as input.
"""
from typing import Callable, Optional
import math
from jax import random
import jax.numpy as jnp
import dataclasses


from pyggdrasil.tree_inference._tree import Tree
import pyggdrasil.tree_inference._tree as tr


def _prune_and_reattach_move(tree: Tree, *, pruned_node: int, attach_to: int) -> Tree:
    """Prune a node from tree topology and attach it to another one.

    Returns:
        new tree, with node ``pruned_node`` pruned and reattached to ``attach_to``.

    Note:
        This is a *pure function*, i.e., the original ``tree`` should not change.
    """
    # get tree
    new_adj_mat = tree.tree_topology
    # get nodes
    pruned_node_idx = jnp.where(tree.labels == pruned_node)[0]
    attach_to_idx = jnp.where(tree.labels == attach_to)[0]
    # Prune Step
    new_adj_mat = new_adj_mat.at[:, pruned_node_idx].set(
        0
    )  # cut all connections of pruned node
    # Attach Step
    new_adj_mat = new_adj_mat.at[attach_to_idx, pruned_node_idx].set(1)
    # make new tree
    new_tree = Tree(tree_topology=new_adj_mat, labels=tree.labels)

    return new_tree


def _prune_and_reattach_proposal(
    key: random.PRNGKeyArray, tree: Tree
) -> tuple[Tree, float]:
    """Samples a new proposal using the "prune and reattach" move.

    Args:
        key: JAX random key
        tree: original tree from which we will build a new sample

    Returns:
        new tree
        float, representing the correction factor
          :math`\\log q(new tree | old tree) - \\log q(old tree | new tree)`.
          As this move is reversible with identical probability,
          this number is always 0.

    Note:
        1. This is a *pure function*, i.e., the original ``tree`` should not change.
        2.
    """
    # get random keys
    rng_prune, rng_reattach = random.split(key)
    # pick a random non-root node to prune
    pruned_node = int(random.choice(rng_prune, tree.labels[:-1]))
    # get descendants of pruned node
    descendants = tr._get_descendants(tree.tree_topology, tree.labels, pruned_node)
    # possible nodes to reattach to - including pruned node for aperiodic case
    possible_nodes = jnp.setdiff1d(tree.labels, descendants)
    # pick a random node to reattach to
    attach_to = int(random.choice(rng_reattach, possible_nodes))
    return (
        _prune_and_reattach_move(
            tree=tree, pruned_node=pruned_node, attach_to=attach_to
        ),
        0.0,
    )


def _swap_node_labels_move(tree: Tree, node1: int, node2: int) -> Tree:
    """Swaps labels between ``node1`` and ``node2`` leaving the tree topology
    untouched."""
    label1 = tree.labels[node1]
    label2 = tree.labels[node2]

    # Copy all the labels
    new_labels = tree.labels
    # Assign label2 to node1...
    new_labels = new_labels.at[node1].set(label2)
    # ... and label1 to node2
    new_labels = new_labels.at[node2].set(label1)

    return Tree(
        tree_topology=tree.tree_topology,
        labels=new_labels,
    )


def _swap_node_labels_proposal(
    key: random.PRNGKeyArray, tree: Tree
) -> tuple[Tree, float]:
    """Samples a new proposal using the "swap labels" move.

    Args:
        key: JAX random key
        tree: original tree from which we will build a new sample

    Returns:
        new tree
        float, representing the correction factor
          :math`\\log q(new tree | old tree) - \\log q(old tree | new tree)`.
          As this move is reversible with identical probability,
          this number is always 0.

    Note:
        This is a *pure function*, i.e., the original ``tree`` should not change.
    """
    # Sample two distinct non-root labels
    node1, node2 = random.choice(key, tree.labels[:-1], shape=(2,), replace=False)

    return _swap_node_labels_move(tree=tree, node1=node1, node2=node2), 0.0


def _swap_subtrees_move(tree: Tree, node1: int, node2: int) -> Tree:
    """Swaps subtrees rooted at ``node1`` and ``node2``.

    Args:
        tree: original tree from which we will build a new sample
        node1: root of the first subtree
        node2: root of the second subtree
    Returns:
        new tree
    """
    # get node indices
    node1_idx = jnp.where(tree.labels == node1)[0]
    node2_idx = jnp.where(tree.labels == node2)[0]
    # get parent of node1
    parent1_idx = jnp.where(tree.tree_topology[:, node1_idx] == 1)[0]
    # get parent of node2
    parent2_idx = jnp.where(tree.tree_topology[:, node2_idx] == 1)[0]
    # detach subtree 1
    new_adj_mat = tree.tree_topology.at[parent1_idx, node1_idx].set(0)
    # detach subtree 2
    new_adj_mat = new_adj_mat.at[parent2_idx, node2_idx].set(0)
    # attach subtree 1 to parent of node2
    new_adj_mat = new_adj_mat.at[parent2_idx, node1_idx].set(1)
    # attach subtree 2 to parent of node1
    new_adj_mat = new_adj_mat.at[parent1_idx, node2_idx].set(1)
    # make new tree
    new_tree = Tree(tree_topology=new_adj_mat, labels=tree.labels)

    return new_tree


def _swap_subtrees_proposal(key: random.PRNGKeyArray, tree: Tree) -> tuple[Tree, float]:
    """Samples a new proposal using the "swap subtrees" move.
    Args:
        key: JAX random key
        tree: original tree from which we will build a new sample
    Returns:
        new tree
        float, representing the correction factor
            :math`\\log q(new tree | old tree) - \\log q(old tree | new tree)`.
    """
    # Sample two distinct non-root labels
    node1, node2 = random.choice(key, tree.labels[:-1], shape=(2,), replace=False)
    # make sure we swap the descendants first
    # (in case the nodes are in the same lineage)
    same_lineage = False
    desc_node1 = tr._get_descendants(tree.tree_topology, tree.labels, node1)
    if node2 in desc_node1:
        # swap, to make node 2 the descendant
        node1, node2 = node2, node1
        same_lineage = True
    # new tree
    # simple case - swap two nodes that are not in the same lineage
    if not same_lineage:
        # Note: correction factor is zero as, move as equal proposal probability
        return _swap_subtrees_move(tree, node1, node2), 0.0
        # nodes are in same lineage - avoid cycles
    else:  # node 2 is descendant
        desc_node2 = tr._get_descendants(tree.tree_topology, tree.labels, node2)
        corr = float(jnp.log(desc_node2 + 1.0) - jnp.log(desc_node1 + 1.0))
        return _swap_subtrees_move(tree, node1, node2), corr


@dataclasses.dataclass
class MoveProbabilities:
    """Move probabilities. The default values were taken from
    the paragraph **Combining the three MCMC moves** of page 14
    of the SCITE paper supplement.
    """

    prune_and_reattach: float = 0.1
    swap_node_labels: float = 0.65
    swap_subtrees: float = 0.25


def _validate_move_probabilities(move_probabilities: MoveProbabilities, /) -> None:
    """Validates if ``move_probabilities`` are valid.

    Raises:
        ValueError, if the probabilities are wrong
    """
    tup = (
        move_probabilities.prune_and_reattach,
        move_probabilities.swap_node_labels,
        move_probabilities.swap_subtrees,
    )

    if min(tup) <= 0 or max(tup) >= 1:
        raise ValueError(
            f"Probabilities must be in the open interval (0, 1). "
            f"Were: {move_probabilities}."
        )
    if not math.isclose(sum(tup), 1.0):
        raise ValueError(
            f"Probabilities must sum up to one. " f"Are: {move_probabilities}"
        )


def _mcmc_kernel(
    key: random.PRNGKeyArray,
    tree: Tree,
    move_probabilities: MoveProbabilities,
    logprobability_fn: Callable[[Tree], float],
    logprobability: Optional[float] = None,
) -> tuple[Tree, float]:
    """

    Args:
        key: JAX random key
        tree: the last tree
        move_probabilities: probabilities of making different moves
        logprobability_fn: function taking a tree and returning its log-probability
          (up to the additive (log-)normalization constant).
        logprobability: log-probability of the tree, :math:`\\log p(tree)`.
          If ``None``, it will be calculated using ``logprobability_fn``, what however
          may increase the number of function evaluations.

    Returns:
        new tree sampled
        log-probability of the tree, can be used at the next iteration

    Note:
        - proposal: refers to the proposal tree in this functions naming/comments
        - tree: refers to the current/old tree
        - log_q_diff: is the log ratio or the proposal probabilities
          called 'correction term' here are 0 for all cases but the swap subtrees move
    """
    # Validate whether move probabilities are right
    _validate_move_probabilities(move_probabilities)

    # Calculate log-probability of the current sample, if not provided
    # log p(old tree)
    logprobability = (
        logprobability_fn(tree) if logprobability is None else logprobability
    )

    # Decide which move to use
    key_which, key_move, key_acceptance = random.split(key, 3)
    move_type = random.choice(
        key_which,
        3,
        p=jnp.asarray(
            [
                move_probabilities.prune_and_reattach,
                move_probabilities.swap_node_labels,
                move_probabilities.swap_subtrees,
            ]
        ),
    )

    # Generate the proposal and the correction term:
    # log q(proposal | old tree) - log q(old tree | proposal)
    if move_type == 0:
        proposal, log_q_diff = _prune_and_reattach_proposal(key, tree)
    elif move_type == 1:
        proposal, log_q_diff = _swap_node_labels_proposal(key, tree)
    else:
        proposal, log_q_diff = _swap_subtrees_proposal(key, tree)

    # log p(proposal)
    logprob_proposal = logprobability_fn(proposal)

    # This is the logarithm of the famous Metropolis-Hastings ratio:
    # log A (proposal | tree) = log p(proposal) - log p(tree)
    #                           + (log q(proposal | tree) - log q(proposal | tree))
    log_ratio = logprob_proposal - logprobability + log_q_diff
    # We want to have A = min(1, r). For numerical stability, we can do
    # log(A) = min(0, log(r)), and log(r) is above
    acceptance_ratio = jnp.exp(min(0.0, log_ratio))

    u = random.uniform(key_acceptance)
    if u <= acceptance_ratio:
        return proposal, logprob_proposal
    else:
        return tree, logprobability
