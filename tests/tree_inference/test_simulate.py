"""Tests of the cell data simulations."""
# _simulate.py
import pytest
import jax.random as random
import numpy as np
import jax.numpy as jnp
import networkx as nx

from pyggdrasil.interface import JAXRandomKey
import pyggdrasil.tree_inference._simulate as sim


def perfect_matrix(
    rng: JAXRandomKey,
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
    sigma = sim._sample_cell_attachment(rng, n_cells, n_nodes, strategy)
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


@pytest.mark.parametrize("seed,", [42, 32])
@pytest.mark.parametrize("n_nodes,", [3, 10])
def test_floyd_warshall(seed: int, n_nodes: int):
    """Test custom floyd warshall algorithm against networkX version."""
    rng = random.PRNGKey(seed)
    A = random.choice(rng, 2, shape=(n_nodes, n_nodes))

    # nodes need to be their own parent in the SCITE implementation
    diag_indices = jnp.diag_indices(A.shape[0])
    A = A.at[diag_indices].set(1)
    # get the shortest path matrix in networkX
    G = nx.from_numpy_array(A, create_using=nx.MultiDiGraph())
    sp_matrix_nx = nx.floyd_warshall_numpy(G)
    # adjust to SCITE conventions of infinity
    sp_matrix_nx = np.where(sp_matrix_nx == np.inf, -1, sp_matrix_nx)
    # adjust to SCITE convention of self connected // i.e. only diagonal
    sp_matrix_nx = np.where(sp_matrix_nx == 0, 1, sp_matrix_nx)

    # run this implementation
    sp_matrix = sim.floyd_warshall(A)

    assert np.array_equal(sp_matrix, sp_matrix_nx)


def test_floyd_warshall_SCITE_example():
    """Manual test of Floyd Warshall based on example in SCITE pape"""
    A = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [1, 1, 0, 1]])
    SP = np.array([[1, -1, -1, -1], [-1, 1, 1, -1], [-1, -1, 1, -1], [1, 1, 2, 1]])
    sp_matrix = sim.floyd_warshall(A)

    assert np.array_equal(sp_matrix, SP)


def test_floyd_warshall_case2():
    """Manual test of Floyd Warshall"""
    A = np.array([[1, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1]])
    SP = np.array([[1, 1, 1, 2], [-1, 1, 2, 1], [-1, -1, 1, -1], [-1, -1, 1, 1]])
    sp_matrix = sim.floyd_warshall(A)

    assert np.array_equal(sp_matrix, SP)


def test_shortest_path_to_ancestry_matrix_SCITE_example():
    """Manual test of Floyd Warshall"""
    SP = np.array([[1, -1, -1, -1], [-1, 1, 1, -1], [-1, -1, 1, -1], [1, 1, 2, 1]])
    ancestor_test_mat = sim.shortest_path_to_ancestry_matrix(SP)
    ancestor_true_mat = np.array(
        [[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [1, 1, 1, 1]]
    )

    assert np.array_equal(ancestor_test_mat, ancestor_true_mat)


def test_built_perfect_mutation_matrix():
    """Manual test of building mutation matrix from ancestor matrix
    and cell placement vector sigma by Eqn 11. - SCITE paper example"""
    ancestor_matrix = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [1, 1, 1, 1]])
    sigma = np.array([1, 1, 1, 4, 3, 3, 2])

    mutation_matrix = sim.built_perfect_mutation_matrix(4, ancestor_matrix, sigma)
    mutation_matrix_true = np.array(
        [
            [1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 0],
        ]
    )

    assert np.array_equal(mutation_matrix, mutation_matrix_true)


@pytest.mark.parametrize("seed,", [42])
@pytest.mark.parametrize("n_nodes,", [3, 10])
@pytest.mark.parametrize("n_cells", [10, 20])
@pytest.mark.parametrize(
    "strategy",
    [
        sim.CellAttachmentStrategy.UNIFORM_INCLUDE_ROOT,
        sim.CellAttachmentStrategy.UNIFORM_EXCLUDE_ROOT,
    ],
)
def test_attach_cells_to_tree_for_strategy_check_bool(
    n_nodes: int, seed: int, strategy: sim.CellAttachmentStrategy, n_cells: int
):
    """Test of cell attachment strategy was respected."""

    rng = random.PRNGKey(seed)
    rng_tree, rng_mutation_mat = random.split(rng)
    tree = random.choice(rng_tree, 2, shape=(n_nodes, n_nodes))

    # nodes need to be their own parent in the SCITE implementation
    diag_indices = jnp.diag_indices(tree.shape[0])
    tree = tree.at[diag_indices].set(1)

    mutation_matrix = sim.attach_cells_to_tree(
        rng_mutation_mat, tree, n_cells, strategy
    )

    if sim.CellAttachmentStrategy.UNIFORM_INCLUDE_ROOT == strategy:
        pass
    elif sim.CellAttachmentStrategy.UNIFORM_EXCLUDE_ROOT == strategy:
        pass
    else:
        raise TypeError(
            "CellAttachmentStrategy not known, dimensions of mutation matrix may fail."
        )

    # check is a boolean matrix
    assert np.array_equal(mutation_matrix, mutation_matrix.astype(bool))
    # check for dimensions
    assert mutation_matrix.shape == (n_nodes - 1, n_cells)


@pytest.mark.parametrize("seed,", [32])
@pytest.mark.parametrize("n_nodes,", [4])
@pytest.mark.parametrize("n_cells", [5])
@pytest.mark.parametrize(
    "strategy",
    [
        sim.CellAttachmentStrategy.UNIFORM_INCLUDE_ROOT,
        sim.CellAttachmentStrategy.UNIFORM_EXCLUDE_ROOT,
    ],
)
def test_attach_cells_to_tree_case1(
    n_nodes: int, seed: int, strategy: sim.CellAttachmentStrategy, n_cells: int
):
    """Manual test of attach cells to tree."""
    rng = random.PRNGKey(32)
    tree = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 1]])
    mutation_matrix = sim.attach_cells_to_tree(rng, tree, n_cells, strategy)
    # define truth
    mutation_matrix_true = np.array([[0, 0], [0, 0]])
    if strategy == sim.CellAttachmentStrategy.UNIFORM_INCLUDE_ROOT:
        mutation_matrix_true = np.array(
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 0, 1, 0]]
        )
    elif strategy == sim.CellAttachmentStrategy.UNIFORM_EXCLUDE_ROOT:
        mutation_matrix_true = np.array(
            [[1, 1, 0, 0, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 1]]
        )
    assert np.array_equal(mutation_matrix, mutation_matrix_true)


def test_gen_sim_data():
    """Test that the dimensions of the mock data are correct."""

    params = {
        "seed": 42,
        "n_cells": 100,
        "n_mutations": 8,
        "fpr": 0.01,
        "fnr": 0.02,
        "na_rate": 0.01,
        "observe_homozygous": True,
        "strategy": "UNIFORM_INCLUDE_ROOT",
    }

    rng = random.PRNGKey(params["seed"])

    params_ty = sim.CellSimulationModel(**params)

    data = sim.gen_sim_data(params_ty, rng)

    # check that the dimensions of the data are correct
    assert np.array(data["adjacency_matrix"]).shape == (8 + 1, 8 + 1)
    assert np.array(data["noisy_mutation_mat"]).shape == (
        8,
        100,
    )  # TODO: to be altered if we truncate the matrix
    assert np.array(data["perfect_mutation_mat"]).shape == (
        8,
        100,
    )  # TODO: to be altered if we truncate the matrix
