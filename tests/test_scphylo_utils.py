"""Tests of scPhylo utilities."""
import pyggdrasil._scphylo_utils as utils

import anytree
import numpy as np
import pandas as pd
import pytest

import numpy.testing as nptest


@pytest.fixture
def model_tree() -> anytree.Node:
    """Simple tree of the form
    root
    ├── TP53
    │   ├── NRAS
    │   └── BRCA
    └── KRAS
    """
    root = anytree.Node("root")
    tp53 = anytree.Node("TP53", parent=root)
    anytree.Node("KRAS", parent=root)
    anytree.Node("NRAS", parent=tp53)
    anytree.Node("BRCA", parent=tp53)
    return root


@pytest.fixture
def model_genotype() -> pd.DataFrame:
    """Genotype matrix of the cells from `model_tree`."""
    expected_matrix = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 0],
        ]
    )
    return pd.DataFrame(expected_matrix, columns=["TP53", "KRAS", "NRAS", "BRCA"])


def test_tree_to_dataframe(
    model_tree: anytree.Node, model_genotype: pd.DataFrame
) -> None:
    """Test if `model_tree` is mapped to `model_genotype`."""
    dataframe = utils.tree_to_dataframe(model_tree)
    assert set(dataframe.columns) == set(model_genotype.columns)

    nptest.assert_allclose(
        dataframe[model_genotype.columns].values,
        model_genotype.values,
    )


def test_dataframe_to_tree() -> None:
    """Test if `model_genotype` is mapped to `model_tree`."""
    pass
