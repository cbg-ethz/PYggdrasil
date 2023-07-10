"""Metrics related to analytics between mcmc sample run computations"""

from typing import Callable, Dict

from pyggdrasil import TreeNode, compare_trees
from pyggdrasil.distances import MP3Similarity, AncestorDescendantSimilarity


class Metrics:
    """Metrics for comparing trees."""

    @staticmethod
    def get(metric: str) -> Callable[[TreeNode, TreeNode], float]:
        """Return metric function."""
        return Metrics._METRICS[metric]

    _METRICS: Dict[str, Callable[[TreeNode, TreeNode], float]] = {  # type: ignore
        "AD": AncestorDescendantSimilarity().calculate,
        "MP3": MP3Similarity().calculate,
        "TrueTree": compare_trees,
    }
