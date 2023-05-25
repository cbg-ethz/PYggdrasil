"""Interfaces related to distance between trees computations."""
from typing import Any, Protocol
from pyggdrasil.tree import TreeNode

from typing_extensions import TypeAlias

_IntegerTreeRoot: TypeAlias = TreeNode[int, Any]


class TreeSimilarityMeasure(Protocol):
    """Interface for similarity or distance functions between the trees.

    The hyperparameters should be set
    at the class initialization stage,
    similarly as with models in SciKit-Learn.
    """

    def calculate(self, /, tree1: _IntegerTreeRoot, tree2: _IntegerTreeRoot) -> float:
        """Calculates similarity between ``tree1`` and ``tree2``.

        Args:
            tree1: root of the first tree. The nodes should be labeled with integers.
            tree2: root of the second tree. The nodes should be labeled with integers.

        Returns:
            similarity from ``tree1`` to ``tree2``
        """
        raise NotImplementedError

    def is_symmetric(self) -> bool:
        """Returns ``True`` if the similarity function is symmetric,
        i.e., :math:`s(t_1, t_2) = s(t_2, t_1)` for all pairs of trees.

        Note:
            If it is not known whether the similarity function is symmetric,
            ``False`` should be returned.
        """
        return False


class TreeSimilarity(TreeSimilarityMeasure):
    """Interface for similarity functions between the trees.

    The hyperparameters should be set
    at the class initialization stage,
    similarly as with models in SciKit-Learn.
    """


class TreeDistance(TreeSimilarityMeasure):
    """Interface for distance functions between the trees.

    The hyperparameters of the metric should be set
    at the class initialization stage,
    similarly as with models in SciKit-Learn.

    Note:
        The distances between trees should be treated as tree
        dissimilarity measures, rather than mathematical metrics.
        For example, the triangle inequality does not need to hold.
    """

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
