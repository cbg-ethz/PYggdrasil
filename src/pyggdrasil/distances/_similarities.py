"""Implements Similarity Classes"""

import anytree

import pyggdrasil.distances._interface as interface


class AncestorDescendantSimilarityInclRoot(interface.TreeSimilarity):
    """Ancestor-descendant similarity,
    adopted from @laurabquintas / Laura Quintas

    Counts the root as a mutation, i.e. considers pairs of ancestor-descendant nodes
    between root and nodes - effectivly making comparisons if mutations exist in
    both trees. May lead a higher similarity score than AncestorDescendantSimilarity.
    """

    def calculate(self, /, tree1: anytree.Node, tree2: anytree.Node) -> float:
        """Calculates similarity between ``tree1`` and ``tree2`` using `scphylo.tl.ad`.

        Args:
            tree1: root of the first tree. The nodes should be labeled with integers.
            tree2: root of the second tree. The nodes should be labeled with integers.

        Returns:
            similarity from ``tree1`` to ``tree2``
        """

        def create_pairs(tree: anytree.Node) -> set:
            """Creates a set of all ancestor-descendant pairs in the tree."""
            pairs = set()
            for node in anytree.PreOrderIter(tree):
                if not node.is_leaf:
                    for child in node.descendants:
                        pairs.add((node.name, child.name))
            return pairs

        pairs1 = create_pairs(tree1)
        pairs2 = create_pairs(tree2)

        return len(pairs1.intersection(pairs2)) / len(pairs1.union(pairs2))

    def is_symmetric(self) -> bool:
        """Returns ``True`` if the similarity function is symmetric,
        i.e., :math:`s(t_1, t_2) = s(t_2, t_1)` for all pairs of trees.

        Note:
            If it is not known whether the similarity function is symmetric,
            ``False`` should be returned.
        """
        return True
