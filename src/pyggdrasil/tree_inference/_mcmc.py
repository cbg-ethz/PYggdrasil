"""Markov Chain Monte Carlo inference for mutation trees according to the SCITE model.

Note:
    This implementation assumes that the false positive
    and false negative rates are known and provided as input.
"""
from typing import Callable, Optional
import jax
import math
from jax import random
import jax.numpy as jnp
import dataclasses
import numpy as np

from pyggdrasil.tree import TreeNode
import pyggdrasil.tree_inference as tree_inf


@dataclasses.dataclass(frozen=True)
class Tree:
    """For ``N`` mutations we use a tree with ``N+1`` nodes,
    where the nodes at positions ``0, ..., N-1`` are "blank"
    and can be bijectively mapped to any of the mutations.
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
        root = tree_inf.adjacency_to_root_dfs(
            adj_matrix=np.array(self.tree_topology), labels=np.array(self.labels)
        )
        return root

    def __str__(self):
        """Prints the tree in a human-readable format."""
        return self.to_TreeNode().print_topo()


def _prune_and_reattach_move(tree: Tree, pruned_node: int, attach_to: int) -> Tree:
    """Prune a node from tree topology and attach it to another one.

    Returns:
        new tree, with node ``pruned_node`` pruned and reattached to ``attach_to``.

    Note:
        This is a *pure function*, i.e., the original ``tree`` should not change.
    """
    raise NotImplementedError


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

    Note:
        1. This is a *pure function*, i.e., the original ``tree`` should not change.
        2.
    """
    raise NotImplementedError


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
          As this move is reversible, this number is always 0.

    Note:
        This is a *pure function*, i.e., the original ``tree`` should not change.
    """
    # Sample two distinct non-root labels
    node1, node2 = 0, 1
    # TODO: jax.random.choice with replace=False should suffice.
    #   It's however very easy to have "off by one" bug here, so unit test
    #   is a good idea.
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
    raise NotImplementedError


def _swap_subtrees_proposal(key: random.PRNGKeyArray, tree: Tree) -> tuple[Tree, float]:
    """Samples a new proposal using the "swap subtrees" move.
    Args:
        key: JAX random key
        tree: original tree from which we will build a new sample
    Returns:
        new tree
        float, representing the correction factor
            :math`\\log q(new tree | old tree) - \\log q(old tree | new tree)`.
            TODO: add the formula
    """
    raise NotImplementedError


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
        logprobability_fn: function taking a tree and returning it's log-probability
          (up to the additive (log-)normalization constant).
        logprobability: log-probability of the tree, :math:`\\log p(tree)`.
          If ``None``, it will be calculated using ``logprobability_fn``, what however
          may increase the number of function evaluations.

    Returns:
        new tree sampled
        log-probability of the tree, can be used at the next iteration
    """
    # Validate whether move probabilities are right
    _validate_move_probabilities(move_probabilities)

    # Calculate log-probability of the current sample, if not provided
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

    logprob_proposal = logprobability_fn(proposal)

    # This is the logarithm of the famous Metropolis-Hastings ratio:
    # https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm#Formal_derivation
    # TODO(Pawel, Gordon): Triple check this.
    log_ratio = logprob_proposal - logprobability - log_q_diff
    # We want to have A = min(1, r). For numerical stability, we can do
    # log(A) = min(0, log(r)), and log(r) is above
    acceptance_ratio = jnp.exp(min(0.0, log_ratio))

    u = random.uniform(key_acceptance)
    if u <= acceptance_ratio:
        return proposal, logprob_proposal
    else:
        return tree, logprobability
