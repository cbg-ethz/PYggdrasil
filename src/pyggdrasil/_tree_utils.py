"""Tree utilities."""
import anytree

import logging

from typing import Union
from pyggdrasil import TreeNode

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def compare_trees(
    tree1: Union[anytree.Node, TreeNode], tree2: Union[anytree.Node, TreeNode]
) -> bool:
    """Compares two labeled rooted trees.

    Returns:
        True iff the two labeled rooted trees have the same structure

    Note:
        Assumes that labels in each tree is unique
        Pyright may throw a false positive error when passing a TreeNode
    """
    logger.debug(f"Comparing trees {tree1.name} and {tree2.name} \n")
    logger.debug(f"Tree1:\n {tree1} \n Tree2:\n {tree2}")

    # If the names or number of children differs, the trees are different
    if tree1.name != tree2.name or len(tree1.children) != len(tree2.children):
        logger.debug(
            f"Trees differ !\n"
            f"Names or number of children differs, the trees are different \n "
            f"Tree1: {tree1.name} Tree2: {tree2.name}"
        )
        return False

    # Sort children by their name to compare without order assumption
    sorted_children1 = sorted(tree1.children, key=lambda x: x.name)
    sorted_children2 = sorted(tree2.children, key=lambda x: x.name)

    # Now we recursively compare subtrees of matched children
    for child1, child2 in zip(sorted_children1, sorted_children2):
        if not compare_trees(child1, child2):
            logger.debug(
                "Trees differ ! - Recursively compared subtrees of matched children \n "
            )
            logger.debug(f"Child1: {child1.name} Child2: {child2.name}")
            return False

    logger.debug("Trees are the same ! \n ")
    return True
