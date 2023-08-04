"""Thin wrappers around scPhylo distance and similarity metrics."""

import logging
import scphylo

import anytree

import pyggdrasil._scphylo_utils as utils
import pyggdrasil.distances._interface as interface

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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

        df1 = utils.tree_to_dataframe(tree1)
        df2 = utils.tree_to_dataframe(tree2)

        try:
            return scphylo.tl.ad(df1, df2)
        except ZeroDivisionError:
            # arises if no pairs of ancestor-descendant nodes can be created
            # or are shared between the two trees
            # may arise if tree1 is a star tree or
            # small trees that do not share all nodes i.e. HUNTRESS
            logger.warning("scPhylo's tl.ad raised ZeroDivisionError")
            logger.warning("Probably due to no shared ancestor-descendant pairs.")
            logger.warning("Tree 1:\n" + str(tree1))
            logger.warning("Tree 2:\n" + str(tree2))
            return 0
        except Exception as e:
            logger.warning("scPhylo's tl.ad raised Error")
            logger.warning("Error: " + str(e))
            logger.warning("Tree 1:\n" + str(tree1))
            logger.warning("Tree 2:\n" + str(tree2))
            return 0

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


class DifferentLineageSimilarity(interface.TreeSimilarity):
    """Different-Lineage similarity.

    Similarity out of one."""

    def calculate(self, /, tree1: anytree.Node, tree2: anytree.Node) -> float:
        """Calculates similarity between ``tree1`` and ``tree2`` using `scphulo.tl.dl`.

        Args:
            tree1: root of the first tree. The nodes should be labeled with integers.
            tree2: root of the second tree. The nodes should be labeled with integers.

        Returns:
            similarity from ``tree1`` to ``tree2``
        """
        df1 = utils.tree_to_dataframe(tree1)
        df2 = utils.tree_to_dataframe(tree2)
        return scphylo.tl.dl(df1, df2)

    def is_symmetric(self) -> bool:
        """Returns ``True`` if the similarity function is symmetric,
        i.e., :math:`s(t_1, t_2) = s(t_2, t_1)` for all pairs of trees.

        Note:
            If it is not known whether the similarity function is symmetric,
            ``False`` should be returned.

        Unknown, but probably not symmetric.
        """
        return False


class MLTDSimilarity(interface.TreeSimilarity):
    """Multi-labeled tree dissimilarity measure (MLTD), normalized to [0,1].

    Similarity out of one.

    Raises: Segmentation faults sometimes, unknown why. - scyphylo's issue.
    """

    def calculate(self, /, tree1: anytree.Node, tree2: anytree.Node) -> float:
        """Calculates similarity between ``tree1`` and ``tree2`` using `scphulo.tl.dl`.

        Args:
            tree1: root of the first tree. The nodes should be labeled with integers.
            tree2: root of the second tree. The nodes should be labeled with integers.

        Returns:
            similarity from ``tree1`` to ``tree2``
        """
        df1 = utils.tree_to_dataframe(tree1)
        df2 = utils.tree_to_dataframe(tree2)
        return scphylo.tl.mltd(df1, df2)["normalized_similarity"]

    def is_symmetric(self) -> bool:
        """Returns ``True`` if the similarity function is symmetric,
        i.e., :math:`s(t_1, t_2) = s(t_2, t_1)` for all pairs of trees.

        Note:
            If it is not known whether the similarity function is symmetric,
            ``False`` should be returned.

        Unknown, but probably not symmetric.
        """
        return False
