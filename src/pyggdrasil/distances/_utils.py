"""Temporarily existing module with utility functions.

When the structure of the subpackage starts to appear,
we will move the functions here to the right packages.

Note:
    Treat this submodule as a part of the *private* API.
"""
import numpy as np
from typing import Sequence
from pyggdrasil.distances._interface import TreeDistance, _IntegerTreeRoot


def calculate_distance_matrix(
    trees1: Sequence[_IntegerTreeRoot],
    trees2: Sequence[_IntegerTreeRoot],
    /,
    *,
    distance: TreeDistance,
) -> np.ndarray:
    """Calculates a cross-distance matrix
    ``d[i, j] = distance(trees1[i], trees2[j])``

    Args:
        trees1: sequence of trees in one set, length m
        trees2: sequence of trees in the second set, length n

    Returns:
        distance matrix, shape (m, n)
    """
    m, n = len(trees1), len(trees2)

    result = np.zeros((m, n))

    for i, tree1 in enumerate(trees1):
        for j, tree2 in enumerate(trees2):
            result[i, j] = distance.calculate(tree1, tree2)

    return result
