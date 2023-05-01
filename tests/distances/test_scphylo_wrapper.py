"""Tests of distance and similarity functions
imported from scphylo"""
import anytree
import numpy as np
import pandas as pd
import pytest
import scphylo

import pyggdrasil.distances as dist


def test_basic() -> None:
    """TEMPORARY: to be adjusted."""
    genotypes = np.asarray(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 0],
        ]
    )

    df = pd.DataFrame(genotypes)

    df.columns = [str(x) for x in df.columns]
    df.index = [str(x) for x in df.index]

    sim = scphylo.tl.mp3(df, df)

    assert sim == pytest.approx(1.0)

    rec = scphylo.tl.huntress(df, 0.001, 0.001)

    assert isinstance(rec, pd.DataFrame)


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


@pytest.mark.parametrize(
    "similarity", [dist.MP3Similarity(), dist.AncestorDescendantSimilarity()]
)
def test_self_similarity(similarity, model_tree) -> None:
    """Tests whether similarity(tree, tree) = 1."""

    assert similarity.calculate(model_tree, model_tree) == pytest.approx(1.0)

    # Now we will create the same tree, but in a different order
    # As this should not matter, we expect the same similarity
    root = anytree.Node("root")
    anytree.Node("KRAS", parent=root)
    tp53 = anytree.Node("TP53", parent=root)
    anytree.Node("BRCA", parent=tp53)
    anytree.Node("NRAS", parent=tp53)

    assert similarity.calculate(model_tree, root) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "similarity", [dist.MP3Similarity(), dist.AncestorDescendantSimilarity()]
)
def test_another_is_different_and_symmetric(similarity, model_tree) -> None:
    """Test whether the similarity of different trees is less than 1.0.

    Additionally, we will check whether the function is really symmetric.
    """
    root = anytree.Node("root")
    tp53 = anytree.Node("TP53", parent=root)
    kras = anytree.Node("KRAS", parent=root)
    anytree.Node("NRAS", parent=tp53)
    anytree.Node("BRCA", parent=kras)

    assert similarity.calculate(model_tree, root) < 1.0

    # Check whether the function is symmetric
    if similarity.is_symmetric():
        assert similarity.calculate(model_tree, root) == pytest.approx(
            similarity.calculate(root, model_tree)
        )
