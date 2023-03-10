"""Tests of the cell data simulations."""
# _simulate.py
import pytest
import jax.random as random
import numpy as np
import jax.numpy as jnp

import pyggdrasil.tree_inference._interface as interface
import pyggdrasil.tree_inference._simulate as sim


def perfect_matrix(
        rng: interface.JAXRandomKey,
        positive_rate: float = 0.3,
        homozygous_rate: float = 0.1,
        shape: tuple = (100, 100),
):
    """
    create perfect matrix
    """
    perfect_mat = np.zeros(shape)

    rng_pos, rng_homo = random.split(rng, 2)

    # Add Positives
    # Generate a random matrix of the same shape as the original
    rand_matrix = random.uniform(key=rng_pos, shape=perfect_mat.shape)
    # Create a mask of elements that satisfy the condition (original value equals y and random value is less than p)
    mask = (perfect_mat == 0) & (rand_matrix < positive_rate)
    # Use the mask to set the corresponding elements of the matrix to x
    perfect_mat = jnp.where(mask, 1, perfect_mat)

    # Generate a random matrix of the same shape as the original
    rand_matrix = random.uniform(key=rng_homo, shape=perfect_mat.shape)
    # Create a mask of elements that satisfy the condition (original value equals y and random value is less than p)
    mask = (perfect_mat == 0) & (rand_matrix < homozygous_rate)
    # Use the mask to set the corresponding elements of the matrix to x
    perfect_mat = jnp.where(mask, 2, perfect_mat)

    return perfect_mat


@pytest.mark.parametrize("seed,", [42])
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
    """ test for expected frequency of NAs in noisy mutation matrix"""
    # RNGs for false positives, false negatives, and missing data
    rng = random.PRNGKey(seed)
    n = 100
    m = 100
    shape = (n, m)
    pos_rate = 0.3
    homozygous_rate = 0.0
    if observe_homozygous:
        homozygous_rate = 0.1

    perfect_mat = perfect_matrix(rng, pos_rate, homozygous_rate, shape)

    # generate noisy matrices
    noisy_mat = sim.add_noise_to_perfect_matrix(
        rng,
        perfect_mat,
        false_positive_rate,
        false_negative_rate,
        missing_entry_rate,
        observe_homozygous,
    )

    freq_na = np.sum(noisy_mat == 3) / (n * m)

    # three standard deviations - stdev = (N * p * (1-p))^0.5
    tolerance = 3 * ((n * m) * missing_entry_rate * (1 - missing_entry_rate)) ** 0.5
    tolerance_freq = tolerance / (n * m)

    assert pytest.approx(missing_entry_rate, abs=tolerance_freq) == freq_na

@pytest.mark.parametrize("seed,", [42])
@pytest.mark.parametrize("false_positive_rate,", [1e-5, 0.1])
@pytest.mark.parametrize("false_negative_rate,", [1e-2, 0.3])
@pytest.mark.parametrize("missing_entry_rate,", [0.0, 1e-2, 0.3, 0.5])
@pytest.mark.parametrize("observe_homozygous,", [True, False])
def test_fp(
    seed: int,
    false_positive_rate: float,
    false_negative_rate: float,
    missing_entry_rate: float,
    observe_homozygous: bool,
):
    """ test for expected frequency of false positives in noisy mutation matrix"""
    # RNGs for false positives, false negatives, and missing data
    rng = random.PRNGKey(seed)
    n = 100
    m = 100
    shape = (n, m)
    pos_rate = 0.3
    homozygous_rate = 0.0
    if observe_homozygous:
        homozygous_rate = 0.1

    perfect_mat = perfect_matrix(rng, pos_rate, homozygous_rate, shape)

    # generate noisy matrices
    noisy_mat = sim.add_noise_to_perfect_matrix(
        rng,
        perfect_mat,
        false_positive_rate,
        false_negative_rate,
        missing_entry_rate,
        observe_homozygous,
    )

    freq_fp = np.sum(noisy_mat == 1) / (n * m)

    # three standard deviations - stdev = (N * p * (1-p))^0.5
    tolerance = 3 * ((pos_rate * n * m) * missing_entry_rate * (1 - missing_entry_rate)) ** 0.5
    tolerance_freq = tolerance / (n * m)

    assert pytest.approx(missing_entry_rate, abs=tolerance_freq) == freq_fp



