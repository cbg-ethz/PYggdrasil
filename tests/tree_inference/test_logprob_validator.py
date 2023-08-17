"""Tests for the logprob_validator module."""

import pytest

import jax.numpy as jnp

import pyggdrasil as yg

import jax.random as random

from pyggdrasil.tree_inference import Tree

import pyggdrasil.tree_inference._logprob as logprob
from pyggdrasil.tree_inference._logprob_validator import (
    _expected,
    logprobability_fn_validator,
)


def test_expected():
    """ "Test expected function."""

    # create tree       # 0  1  2  3  4
    adj_mat = jnp.array(
        [
            [0, 0, 0, 0, 0, 0],  # 0
            [0, 0, 0, 0, 0, 0],  # 1
            [0, 0, 0, 0, 0, 0],  # 2
            [1, 0, 0, 0, 0, 0],  # 3
            [0, 1, 1, 0, 0, 0],  # 4
            [0, 0, 0, 1, 1, 0],  # 5
        ]
    )
    labels = jnp.array([0, 1, 2, 3, 4, 5])

    tree = Tree(adj_mat, labels)

    # create cell attachment vector
    # say we have 1 cell and the j=0 cell is attached to node i=2
    cell_attachment = jnp.array([2])

    # cells
    # expected matrix       #  0  1  2  3  4
    expected_mat = jnp.array(
        [
            [0, 0, 0, 0, 0],  # 0  mutations
            [0, 0, 0, 0, 0],  # 1
            [1, 0, 0, 0, 0],  # 2
            [0, 0, 0, 0, 0],  # 3
            [1, 0, 0, 0, 0],  # 4
        ]
    )

    # get expected

    for mutation_i in range(5):  #  forget root
        cell_j = 0
        print("mutation_i", mutation_i)
        fn_value = _expected(tree, mutation_i, cell_attachment).__int__()
        expected = expected_mat[mutation_i, cell_j].__int__()
        print("fn_value", fn_value)
        assert fn_value == expected


def mutation_data_tree_error(n_cells, n_mutations, error_rates, seed) -> tuple:
    """Define tree, error settings, and mutation matrix for testing."""

    # make random key jax
    key = random.PRNGKey(seed)
    # split into 2 keys
    key1, key2 = random.split(key)
    # make random number for tree generation
    seed_tree = random.randint(key1, (), 0, 1000000)

    # make true tree - random
    tree = yg.tree_inference.make_tree(
        n_mutations + 1, yg.tree_inference.TreeType.RANDOM, int(seed_tree)
    )

    # make mutation matrix
    params_model = yg.tree_inference.CellSimulationModel(
        n_cells=n_cells,
        n_mutations=n_mutations,
        fpr=error_rates.value.fpr,
        fnr=error_rates.value.fnr,
        na_rate=0.0,
        observe_homozygous=False,
        strategy=yg.tree_inference.CellAttachmentStrategy.UNIFORM_INCLUDE_ROOT,
    )
    data = yg.tree_inference.gen_sim_data(params_model, key2, tree)

    # define error rates
    error_rate = (error_rates.value.fpr, error_rates.value.fnr)

    return tree, error_rate, jnp.array(data["noisy_mutation_mat"])


@pytest.mark.parametrize("n_cells", [3, 4, 5])
@pytest.mark.parametrize("n_mutations", [2, 3, 4])
@pytest.mark.parametrize(
    "error_rates",
    [
        yg.tree_inference.ErrorCombinations.IDEAL,
        yg.tree_inference.ErrorCombinations.TYPICAL,
        yg.tree_inference.ErrorCombinations.LARGE,
    ],
)
@pytest.mark.parametrize("seed", [23, 890, 2])
def test_logprobability_fn_against_validator(n_cells, n_mutations, error_rates, seed):
    """Test logprobability function against validator."""

    # define tree, error rates, and mutation matrix
    tree, error_rate, data = mutation_data_tree_error(
        n_cells, n_mutations, error_rates, seed
    )

    tree = Tree.tree_from_tree_node(tree)

    # run fast logprob
    logprob_fast = logprob.logprobability_fn(data, tree, error_rate)

    # run slow logprob
    logprob_validator = logprobability_fn_validator(tree, data, error_rate)

    # assert equal
    print(f"\nfast: {logprob_fast}\nvalidator: {logprob_validator}")
    assert jnp.isclose(logprob_fast, logprob_validator, atol=1e-6)
