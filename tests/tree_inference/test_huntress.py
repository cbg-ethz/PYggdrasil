"""Tests of the HUNTRESS algorithm wrapper."""
import pyggdrasil.tree_inference._huntress as inference
from pyggdrasil._tree_utils import compare_trees

import anytree
import numpy as np
import pytest


@pytest.fixture
def model_tree() -> anytree.Node:
    """Simple tree of the form
    4
    ├── 0
    │   ├── 1
    │   └── 2
    └── 3
    """
    root = anytree.Node(4)
    n0 = anytree.Node(0, parent=root)
    anytree.Node(3, parent=root)
    anytree.Node(1, parent=n0)
    anytree.Node(2, parent=n0)
    return root


@pytest.fixture
def model_genotype() -> np.ndarray:
    """Genotype matrix of the cells from `model_tree`."""
    mini_matrix = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    # Generate more cells
    n_copies = 4
    matrix = np.vstack([mini_matrix for _ in range(n_copies)])

    return matrix


def test_huntress(model_tree: anytree.Node, model_genotype: np.ndarray) -> None:
    """Tests whether the reconstructed tree matches the ground truth."""
    assert isinstance(model_genotype, np.ndarray)
    inferred_tree = inference.huntress_tree_inference(
        model_genotype,
        false_positive_rate=1e-4,
        false_negative_rate=1e-4,
    )

    # print statements are ignored for passed tests
    print("Expected tree")
    for pre, _, node in anytree.RenderTree(model_tree):
        print("%s%s" % (pre, node.name))

    print("Reconstructed tree")
    for pre, _, node in anytree.RenderTree(inferred_tree):
        print("%s%s" % (pre, node.name))

    assert compare_trees(model_tree, inferred_tree)
