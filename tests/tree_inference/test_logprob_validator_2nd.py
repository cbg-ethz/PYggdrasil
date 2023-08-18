"""Tests for the logprob_validator module."""

import pytest

import jax.numpy as jnp

import jax.random as random

import pyggdrasil as yg

from pyggdrasil.tree_inference import Tree

import pyggdrasil.tree_inference._logprob as logprob
import pyggdrasil.tree_inference._logprob_validator as logprob_validator

import pyggdrasil.tree_inference._tree as tr

from pyggdrasil.tree_inference._log_prob_validator_2nd import (
    _all_attachments,
    _expected,
    logprbability_fn,
)


def test_attachments():
    """ "Test all attachments function."""

    mutation_labels = jnp.array([0, 1, 2])

    m_cells = 2

    attachments = _all_attachments(m_cells, mutation_labels)

    # attacments_manual = jnp.array([0,0], [0,1], [0,2], [1,0],
    # [1,1], [1,2], [2,0], [2,1], [2,2])

    assert len(attachments) == 9


def test_expected():
    """ "Test expected function."""

    # define tree
    adj_mat = jnp.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]])
    labels = jnp.array([0, 1, 2, 3])
    tree = Tree(adj_mat, labels)
    tr._get_ancestor_matrix(tree.tree_topology)

    # get the ancestor matrix
    ancestor_mat = tr._get_ancestor_matrix(tree.tree_topology)
    # truncate the last row, which is the root
    ancestor_mat = ancestor_mat[:-1, :]

    # define the cell attachment vector
    cell_attachment = jnp.array([0, 1])

    true_expected = jnp.array([[1, 0], [1, 1], [0, 0]])

    # run the function
    for cell_i in range(2):
        for mutation_i in range(3):
            expected = _expected(cell_i, mutation_i, cell_attachment, ancestor_mat)
            assert expected == true_expected[mutation_i, cell_i]


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
def test_orthogonal_log_probs_validator2_fast(n_cells, n_mutations, error_rates, seed):
    """Test orthogonal_log_prob implementations

    Fast: logprob._logprobability_fn: uses einsum and logsumexp
    Slow: validator 2:
    """

    # define tree, error rates, and mutation matrix
    tree, error_rate, data = mutation_data_tree_error(
        n_cells, n_mutations, error_rates, seed
    )

    # convert tree: TreeNode -> Tree
    tree = yg.tree_inference.Tree.tree_from_tree_node(tree)

    # run fast logprob
    logprob_fast = logprob.logprobability_fn(data, tree, error_rate)

    # run validator 2
    logprob_val2 = logprbability_fn(data, tree, error_rate)

    # assert equal
    print(logprob_fast, logprob_val2)
    assert jnp.isclose(logprob_fast, logprob_val2, atol=1e-6)


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
def test_orthogonal_log_probs_validator_validator2(
    n_cells, n_mutations, error_rates, seed
):
    """Test orthogonal_log_prob implementations

    Fast: logprob._logprobability_fn: uses einsum and logsumexp
    Slow: validator 2:
    """

    # define tree, error rates, and mutation matrix
    tree, error_rate, data = mutation_data_tree_error(
        n_cells, n_mutations, error_rates, seed
    )

    # convert tree: TreeNode -> Tree
    tree = yg.tree_inference.Tree.tree_from_tree_node(tree)

    # run validator
    logprob_validator_v = logprob_validator.logprobability_fn_validator(
        tree, data, error_rate
    )

    # run validator 2
    logprob_val2 = logprbability_fn(data, tree, error_rate)

    # assert equal
    print(logprob_validator_v, logprob_val2)
    assert jnp.isclose(logprob_validator_v, logprob_val2, atol=1e-6)
