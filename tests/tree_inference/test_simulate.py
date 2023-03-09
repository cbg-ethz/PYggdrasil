"""Tests of the cell data simulations."""
# _simulate.py

import pytest
import jax.random as random
import numpy as np

import pyggdrasil.tree_inference._simulate
import pyggdrasil.tree_inference._simulate as sim


@pytest.mark.parametrize("seed,", [42, 97])
@pytest.mark.parametrize("false_positive_rate,", [1e-5, 0.1])
@pytest.mark.parametrize("false_negative_rate,", [1e-2, 0.3])
@pytest.mark.parametrize("missing_entry_rate,", [0.0, 1e-2, 0.3, 0.5])
@pytest.mark.parametrize("observe_homozygous,", [True, False])
def test_na_freq(
    seed: int,
    false_positive_rate: float,
    false_negative_rate: float,
    missing_entry_rate: float,
    observe_homozygous: bool,
):
    """

    Test for frequency of NAs under all conditions
    Args:
        seed: to generate JAX random key
        false_positive_rate: false positive rate :math:`\\alpha`.
          Should be in the half-open interval [0, 1).
        false_negative_rate: alse negative rate :math:`\\beta`.
          Should be in the half-open interval [0, 1).
        missing_entry_rate: fraction os missing entries.
          Should be in the half-open interval [0, 1).
          If 0, no missing entries are present.
        observe_homozygous: have homozygous mutations or not
    """
    # RNGs for false positives, false negatives, and missing data
    rng = random.PRNGKey(seed)

    # generate random test matrix
    n = 100
    m = 100
    size = (n, m)
    if not observe_homozygous:
        perfect_mat = random.bernoulli(rng, 0.5, size)
    else:
        perfect_mat = random.bernoulli(rng, 0.3, size)
        perfect_mat = perfect_mat.astype(int) + random.bernoulli(rng, 0.2, size)

    perfect_mat = pyggdrasil.tree_inference._simulate.PerfectMutationMatrix(perfect_mat)
    # generate noisy matrices
    noisy_mat = sim.add_noise_to_perfect_matrix(
        rng,
        perfect_mat,
        false_positive_rate,
        false_negative_rate,
        missing_entry_rate,
        observe_homozygous,
    )

    freq_na = np.sum(np.isnan(noisy_mat)) / (n * m)

    # three standard deviations - stdev = (N * p * (1-p))^0.5
    tolerance = 3 * ((n * m) * missing_entry_rate * (1 - missing_entry_rate)) ** 0.5
    tolerance_freq = tolerance / (n * m)

    assert pytest.approx(missing_entry_rate, abs=tolerance_freq) == freq_na
