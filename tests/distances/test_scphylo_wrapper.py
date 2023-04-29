"""Tests of distance and similarity functions
imported from scphylo"""
import numpy as np
import pandas as pd
import pytest
import scphylo


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
