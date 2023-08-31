"""Tests the Gelman-Rubin convergence diagnostics."""


import numpy as np

import pyggdrasil.analyze._rhat as rhat


def test_rhat_basic():
    """Tests the return shape and one of case of R-hat."""
    # given two chains that converge to the same value 8
    chains = np.array(
        [
            np.array(
                [
                    1,
                    1,
                    3,
                    5,
                    6,
                    6,
                    6,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                ]
            ),
            np.array(
                [
                    5,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                ]
            ),
        ]
    )
    # calculate the rhat
    rhats = rhat.rhats(chains)

    # Does it return the correct number of rhats? 4 less than the length of the chains
    chains_len = len(chains[0])
    assert rhats.shape == (chains_len - 3,)

    # Does it approach 1 as the number of draws increases?
    assert rhats[-1] < 1.3  # 1.2 is often used as a cutoff for convergence
