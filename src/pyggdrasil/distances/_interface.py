"""Interfaces related to distance between trees computations."""
from typing import Any, Protocol
from pyggdrasil.tree import TreeNode

from typing_extensions import TypeAlias

_IntegerTreeRoot: TypeAlias = TreeNode[int, Any]


class TreeDistance(Protocol):
    """Interface for distance functions between the trees.

    The hyperparameters of the metric should be set
    at the class initialization stage,
    similarly as with models in SciKit-Learn.

    Note:
        The distances between trees should be treated as tree
        dissimilarity measures, rather than mathematical metrics.
        For example, the triangle inequality does not need to hold.
    """

    def calculate_distance(
        self, /, tree1: _IntegerTreeRoot, tree2: _IntegerTreeRoot
    ) -> float:
        """Calculates distance between ``tree1`` and ``tree2``.

        Args:
            tree1: root of the first tree. The nodes should be labeled with integers.
            tree2: root of the second tree. The nodes should be labeled with integers.

        Returns:
            distance from ``tree1`` to ``tree2``
        """
        raise NotImplementedError

    def is_symmetric(self) -> bool:
        """Returns ``True`` if the distance function is symmetric,
        i.e., :math:`d(t_1, t_2) = d(t_2, t_1)` for all pairs of trees.

        Note:
            If it is not known whether the distance function is symmetric,
            ``False`` should be returned.
        """
        return True

    def triangle_inequality(self) -> bool:
        """Returns ``True`` if the triangle inequality

        .. math::

           d(t_1, t_3) <= d(t_1, t_2) + d(t_2, t_3)

        is known to hold for this distance.

        Note:
            If it is not known whether the triangle inequality
            holds for a metric, ``False`` should be returned.
        """
        return False