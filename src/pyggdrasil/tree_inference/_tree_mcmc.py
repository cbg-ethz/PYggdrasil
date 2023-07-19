"""Implements Tree operations that rely on _mcmc.py"""

from pyggdrasil import TreeNode

from pyggdrasil.interface import JAXRandomKey

from pyggdrasil.tree_inference._interface import MoveProbabilities
from pyggdrasil.tree_inference._config import MoveProbConfigOptions
import pyggdrasil.tree_inference._mcmc as mcmc
from pyggdrasil.tree_inference._tree import Tree


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
