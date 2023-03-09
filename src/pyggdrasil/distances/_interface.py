"""Interfaces related to distance between trees computations."""
from typing import Any, Protocol
from pyggdrasil.tree import TreeNode

_IntegerTreeRoot = TreeNode[int, Any]


class TreeDistance(Protocol):
    """Interface for different metrics (distance functions) between the trees.

    The hyperparameters of the metric should be set
    at the class initialization stage,
    similarly as with models in SciKit-Learn.
    """

    def calculate_distance(
        self, tree1: _IntegerTreeRoot, tree2: _IntegerTreeRoot
    ) -> float:
        """Calculates distance between ``tree1`` and ``tree2``.

        Args:
            tree1: root of the first tree. The nodes should be labeled with integers.
            tree2: root of the second tree. The nodes should be labeled with integers.
        """
        raise NotImplementedError
