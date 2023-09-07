"""Implements Tree operations that rely on _mcmc.py"""

from pyggdrasil import TreeNode

from pyggdrasil.interface import JAXRandomKey

from pyggdrasil.tree_inference._interface import MoveProbabilities
from pyggdrasil.tree_inference._config import MoveProbConfigOptions
import pyggdrasil.tree_inference._mcmc as mcmc
from pyggdrasil.tree_inference._tree import Tree

import jax.random as random


def evolve_tree_mcmc(
    init_tree: TreeNode,
    n_moves: int,
    rng: JAXRandomKey,
    move_probs: MoveProbabilities = MoveProbConfigOptions.DEFAULT.value,  # type: ignore
) -> TreeNode:
    """Evolves a tree using the SCITE MCMC moves, assumes default move weights.

    Args:
        init_tree: TreeNode
            tree to evolve
        n_moves: int
            number of moves to perform
        rng: JAXRandomKey
            random number generator
        move_probs: MoveProbabilities
            move probabilities to use

    Returns:
        tree_ev: TreeNode
            evolved tree
    """

    tree = Tree.tree_from_tree_node(init_tree)

    tree_ev = mcmc._evolve_tree_mcmc(tree, n_moves, rng, move_probs)

    return tree_ev.to_TreeNode()


def evolve_tree_mcmc_all(
    init_tree: TreeNode,
    n_moves: int,
    rng: JAXRandomKey,
    move_probs: MoveProbabilities = MoveProbConfigOptions.DEFAULT.value,  # type: ignore
) -> list[TreeNode]:
    """Evolves a tree using the SCITE MCMC moves, assumes default move weights.

    Args:
        init_tree: TreeNode
            tree to evolve
        n_moves: int
            number of moves to perform
        rng: JAXRandomKey
            random number generator
        move_probs: MoveProbabilities
            move probabilities to use

    Returns:
        trees: list[TreeNode]
            evolved trees in order of evolution
    """

    tree = Tree.tree_from_tree_node(init_tree)

    trees = []
    tree_ev = tree  # set initial tree
    for _ in range(n_moves):
        # split random number generator
        rng, rng_move = random.split(rng)
        tree_ev = mcmc._evolve_tree_mcmc(tree_ev, 1, rng_move, move_probs)
        trees.append(tree_ev.to_TreeNode())

    return trees
