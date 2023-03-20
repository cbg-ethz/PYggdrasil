"""Tests of the cell data simulations."""
# _simulate.py
import pytest
import jax.random as random
import numpy as np
import jax.numpy as jnp

# import networkx as nx

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
    negative_rate = 1 - positive_rate - homozygous_rate

    # Set up the probabilities for each y value
    probabilities = jnp.array(
        [
            negative_rate,
            positive_rate,
            homozygous_rate,
        ]
    )

    # Generate the matrix of y values
    perfect_mat = random.choice(
        rng, jnp.arange(len(probabilities)), shape=shape, p=probabilities
    )

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
    """test for expected frequency of NAs in noisy mutation matrix"""

    # RNGs for false positives, false negatives, and missing data
    rng = random.PRNGKey(seed)
    n = 300
    m = 300
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
@pytest.mark.parametrize("false_positive_rate,", [1e-5, 0.1, 0.5])
@pytest.mark.parametrize("false_negative_rate,", [1e-2, 0.3])
@pytest.mark.parametrize("missing_entry_rate,", [1e-2])
@pytest.mark.parametrize("observe_homozygous,", [True, False])
def test_fp(
    seed: int,
    false_positive_rate: float,
    false_negative_rate: float,
    missing_entry_rate: float,
    observe_homozygous: bool,
):
    """test for expected frequency of false positives in noisy mutation matrix"""

    # RNGs for false positives, false negatives, and missing data
    rng = random.PRNGKey(seed)
    n = 300
    m = 300
    shape = (n, m)
    pos_rate = 0.3
    homozygous_rate = 0.0
    if observe_homozygous:
        homozygous_rate = 0.1
    neg_rate = 1 - pos_rate - homozygous_rate

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

    freq_fp = np.sum((noisy_mat == 1) & (perfect_mat == 0)) / (n * m)

    rate = false_positive_rate * neg_rate

    assert pytest.approx(rate, abs=0.03) == freq_fp


@pytest.mark.parametrize("seed,", [42])
@pytest.mark.parametrize("false_positive_rate,", [1e-5, 0.1])
@pytest.mark.parametrize("false_negative_rate,", [1e-2, 0.3, 0.5])
@pytest.mark.parametrize("missing_entry_rate,", [0.0, 1e-2])
@pytest.mark.parametrize("observe_homozygous,", [True, False])
def test_fn(
    seed: int,
    false_positive_rate: float,
    false_negative_rate: float,
    missing_entry_rate: float,
    observe_homozygous: bool,
):
    """test for expected frequency of false negatives in noisy mutation matrix"""

    # RNGs for false positives, false negatives, and missing data
    rng = random.PRNGKey(seed)
    n = 300
    m = 300
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
    freq = np.sum((noisy_mat == 0) & (perfect_mat == 1)) / (n * m)
    # NB if probybilistc tolerance were used
    # remember the changed rates if homozygous or not

    if observe_homozygous:
        rate = false_negative_rate / 2 * pos_rate
    else:
        rate = false_negative_rate * pos_rate

    assert pytest.approx(rate, abs=0.03) == freq


@pytest.mark.parametrize("seed,", [42])
@pytest.mark.parametrize("false_positive_rate,", [1e-5, 0.1])
@pytest.mark.parametrize("false_negative_rate,", [1e-2, 0.3, 0.4])
@pytest.mark.parametrize("missing_entry_rate,", [0.0, 1e-2])
@pytest.mark.parametrize("observe_homozygous,", [True, False])
def test_false_homo_unmutated(
    seed: int,
    false_positive_rate: float,
    false_negative_rate: float,
    missing_entry_rate: float,
    observe_homozygous: bool,
):
    """test for expected frequency of false homozygous
    mutations from non-mutated in noisy mutation matrix"""
    # RNGs for false positives, false negatives, and missing data
    rng = random.PRNGKey(seed)
    n = 300
    m = 300
    shape = (n, m)
    pos_rate = 0.3
    homozygous_rate = 0.0
    if observe_homozygous:
        homozygous_rate = 0.1
    neg_rate = 1 - pos_rate - homozygous_rate

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

    freq = np.sum((noisy_mat == 2) & (perfect_mat == 0)) / (n * m)

    if observe_homozygous:
        rate = (false_positive_rate * false_negative_rate / 2) * neg_rate
    else:
        rate = 0

    assert pytest.approx(rate, abs=0.03) == freq


@pytest.mark.parametrize("seed,", [42])
@pytest.mark.parametrize("false_positive_rate,", [1e-5, 0.1, 0.3])
@pytest.mark.parametrize("false_negative_rate,", [1e-2, 0.3])
@pytest.mark.parametrize("missing_entry_rate,", [0.0, 1e-2])
@pytest.mark.parametrize("observe_homozygous,", [True, False])
def test_false_homo_mutated(
    seed: int,
    false_positive_rate: float,
    false_negative_rate: float,
    missing_entry_rate: float,
    observe_homozygous: bool,
):
    """test for expected frequency of false homozygous
    mutations from mutated in noisy mutation matrix"""
    # RNGs for false positives, false negatives, and missing data
    rng = random.PRNGKey(seed)
    n = 300
    m = 300
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

    freq = np.sum((noisy_mat == 2) & (perfect_mat == 1)) / (n * m)

    if observe_homozygous:
        rate = (false_negative_rate / 2) * pos_rate
    else:
        rate = 0

    assert pytest.approx(rate, abs=0.03) == freq


@pytest.mark.parametrize("seed,", [42])
@pytest.mark.parametrize("n_cells,", [30, 100])
@pytest.mark.parametrize("n_nodes,", [5, 20])
@pytest.mark.parametrize(
    "strategy,",
    [
        sim.CellAttachmentStrategy.UNIFORM_EXCLUDE_ROOT,
        sim.CellAttachmentStrategy.UNIFORM_INCLUDE_ROOT,
    ],
)
def test_sample_cell_attachment_freq(
    seed: int, n_cells: int, n_nodes: int, strategy: sim.CellAttachmentStrategy
):
    """Test expected frequencies of nodes sampled,
    should be accepted in 99.7 % cases - 3 sigma.
    """
    # get counts
    rng = random.PRNGKey(seed)
    sigma = sim.sample_cell_attachment(rng, n_cells, n_nodes, strategy)
    unique, counts = jnp.unique(sigma, return_counts=True)

    # get expected, and accuracy
    if strategy == sim.CellAttachmentStrategy.UNIFORM_INCLUDE_ROOT:
        p = 1 / n_nodes
    elif strategy == sim.CellAttachmentStrategy.UNIFORM_EXCLUDE_ROOT:
        p = 1 / (n_nodes - 1)
    else:
        raise ValueError(f"CellAttachmentStrategy {strategy} is not valid.")

    expected_unique = [n_cells * p] * len(unique)
    unique_stdev = (n_cells * p * (1 - p)) ** 0.5

    for i, (x, y) in enumerate(zip(counts, expected_unique)):
        assert (
            abs(x - y) <= 4 * unique_stdev
        ), f"Sampled to expected count for node {unique[i]} is unlikely: {x} != {y}"


# @pytest.mark.parametrize("seed,", [42,32])
# @pytest.mark.parametrize("n_cells,", [10, 100])
# def test_floyd_warshall(
#         seed: int,
#         n: int
# ):
#     """Tests tests custom floyd warshall algorithm against networkX version.
#     """
#     rng = random.PRNGKey(seed)
#     A = random.choice(rng, 2, shape=(n, n))
#     G = nx.from_numpy_matrix(A)
#     sp_matrix_nx = nx.floyd_warshall_numpy(G)
#     sp_matrix_nx = np.where(sp_matrix_nx == np.inf, -1, sp_matrix_nx)
#     sp_matrix = sim.floyd_warshall(A)
#
#     assert np.array_equal(sp_matrix, sp_matrix_nx)


def test_floyd_warshall_example():
    """Manual test of Floyd Warshall based on example in SCITE paper"""
    A = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [1, 1, 0, 1]])
    SP = np.array([[1, -1, -1, -1], [-1, 1, 1, -1], [-1, -1, 1, -1], [1, 1, 2, 1]])
    sp_matrix = sim.floyd_warshall(A)

    assert np.array_equal(sp_matrix, SP)
