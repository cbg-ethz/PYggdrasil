"""Thin wrappers around scPhylo distance and similarity metrics."""
import warnings

import scphylo

import anytree

import pyggdrasil._scphylo_utils as utils
import pyggdrasil.distances._interface as interface


class AncestorDescendantSimilarity(interface.TreeSimilarity):
    """Ancestor-descendant accuracy.

    Note: - Considers only ancestor-descendant relationships between mutation,
          i.e. excludes the root node. For an implementation with the root considered
           see AncestorDescendantSimilarityInclRoot instead.

    Raises:
        DivisionByZeroError:
            If first tree is a star tree. Fork of scPhylo's not updated yet.
            Happens as no pairs of ancestor-descendant nodes can be created,
            given root is not considered.
    """

    def calculate(self, /, tree1: anytree.Node, tree2: anytree.Node) -> float:
        """Calculates similarity between ``tree1`` and ``tree2`` using `scphylo.tl.ad`.

        Args:
            tree1: root of the first tree. The nodes should be labeled with integers.
            tree2: root of the second tree. The nodes should be labeled with integers.

        Returns:
            similarity from ``tree1`` to ``tree2``
        """
        warnings.warn(
            "This function might be damaged."
            " See AncestorDescendantSimilarity_lq instead."
        )

        df1 = utils.tree_to_dataframe(tree1)
        df2 = utils.tree_to_dataframe(tree2)

        return scphylo.tl.ad(df1, df2)

    def is_symmetric(self) -> bool:
        """Returns ``True`` if the similarity function is symmetric,
        i.e., :math:`s(t_1, t_2) = s(t_2, t_1)` for all pairs of trees.

        Note:
            If it is not known whether the similarity function is symmetric,
            ``False`` should be returned.
        """
        return True


class MP3Similarity(interface.TreeSimilarity):
    """MP3 similarity."""

    def calculate(self, /, tree1: anytree.Node, tree2: anytree.Node) -> float:
        """Calculates similarity between ``tree1`` and ``tree2`` using `scphulo.tl.mp3`.

        Args:
            tree1: root of the first tree. The nodes should be labeled with integers.
            tree2: root of the second tree. The nodes should be labeled with integers.

        Returns:
            similarity from ``tree1`` to ``tree2``
        """
        df1 = utils.tree_to_dataframe(tree1)
        df2 = utils.tree_to_dataframe(tree2)
        return scphylo.tl.mp3(df1, df2)

    def is_symmetric(self) -> bool:
        """Returns ``True`` if the similarity function is symmetric,
        i.e., :math:`s(t_1, t_2) = s(t_2, t_1)` for all pairs of trees.

        Note:
            If it is not known whether the similarity function is symmetric,
            ``False`` should be returned.
        """
        return True
