"""Metrics related to analytics between mcmc sample run computations"""

from typing import Callable, Dict

from pyggdrasil import TreeNode, compare_trees
import pyggdrasil.distances as dist


class Metrics:
    """Metrics for comparing trees.

    Attributes:
        _METRICS: Dictionary of metrics.
    """

    @staticmethod
    def get(metric: str) -> Callable[[TreeNode, TreeNode], float]:
        """Return metric function.

        Args:
            metric: Name of metric.

        Returns:
            - AD: Ancestor-Descendant Similarity;
              pyggdrasil.distances.AncestorDescendantSimilarity().calculate,
            - MP3: MP3 Similarity;
              pyggdrasil.distances.MP3Similarity().calculate,
            - TrueTree: True Tree Similarity;
              pyggdrasil.compare_trees
            - DL: Different Lineage Similarity;
              pyggdrasil.distances.DifferentLineageSimilarity().calculate,
            - MLTD: MLTD Similarity;
              pyggdrasil.distances.MLTDSimilarity().calculate,

        """
        return Metrics._METRICS[metric]

    _METRICS: Dict[str, Callable[[TreeNode, TreeNode], float]] = {  # type: ignore
        "AD": dist.AncestorDescendantSimilarity().calculate,
        "MP3": dist.MP3Similarity().calculate,
        "TrueTree": compare_trees,
        "DL": dist.DifferentLineageSimilarity().calculate,
        "MLTD": dist.MLTDSimilarity().calculate,
    }
