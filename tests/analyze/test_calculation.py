"""Tree MCMC sample calculation analysis."""
import jax.random as random
import pytest
import jax.numpy as jnp
import xarray as xr

import pyggdrasil.analyze as analyze

from pyggdrasil.interface import MCMCSample, PureMcmcData
from pyggdrasil.tree_inference._tree import Tree
from pyggdrasil.tree_inference._tree_generator import _generate_random_tree_adj_mat


@pytest.fixture
def mcmc_samples() -> list[MCMCSample]:
    """Generate a list of MCMCSample objects.

    of fixed tree size 10, linear increasing log-probabilities.
    and random tree topologies.

    The first and the last tree are the same.
    """
    num_samples = 10

    # Generate and return a list of MCMCSample objects
    mcmc_samples = []

    # Linearly increasing log-probability values
    log_probabilities = jnp.linspace(-121.6, -100.0, num_samples)

    # Generate a list of random trees
    trees = [
        _generate_random_tree_adj_mat(rng=random.PRNGKey(j), n_nodes=10)
        for j in range(num_samples)
    ]

    initial_tree = trees[0]

    # Make the last tree the same as the first - needed for testing
    trees[-1] = initial_tree

    for i in range(num_samples):
        # Create an example MCMCSample
        ds = xr.Dataset(
            {
                "iteration": xr.DataArray([i], dims="sample"),
                "tree": xr.DataArray(trees[i], dims=("from_node_k", "to_node_k")),
                "log-probability": xr.DataArray([log_probabilities[i]], dims="sample"),
            },
            coords={
                "from_node_k": [8, 2, 3, 1, 4, 7, 6, 0, 5, 9],
                "to_node_k": [8, 2, 3, 1, 4, 7, 6, 0, 5, 9],
            },
        )

        mcmc_samples.append(ds)

    return mcmc_samples


@pytest.fixture
def pure_mcmc_data(mcmc_samples) -> PureMcmcData:
    """Generate a PureMcmcData object from mcmc_samples."""

    pure_data = analyze.to_pure_mcmc_data(mcmc_samples)

    # check dimensions of the result
    assert pure_data.iterations.shape == (10,)
    assert pure_data.log_probabilities.shape == (10,)
    assert len(pure_data.trees) == 10

    return pure_data


def test_check_run_for_tree(pure_mcmc_data):
    """Test check_run_for_tree function."""

    topology = jnp.array(
        _generate_random_tree_adj_mat(rng=random.PRNGKey(1000), n_nodes=10)
    )

    labels = jnp.array([8, 2, 3, 1, 4, 7, 6, 0, 5, 9])

    tree_not_found = Tree(topology, labels)

    tree_not_found.to_TreeNode()

    # check this tree is not in the run
    assert (
        analyze.check_run_for_tree(
            desired_tree=tree_not_found, mcmc_samples=pure_mcmc_data
        )
        is False
    )

    # Check that the first tree is equal to the last tree
    result = analyze.check_run_for_tree(
        desired_tree=pure_mcmc_data.trees[0], mcmc_samples=pure_mcmc_data
    )

    assert len(result.trees) == 2  # type: ignore


def test_analyze_mcmc_run_true_tree(pure_mcmc_data: PureMcmcData) -> None:
    """Test analyze_mcmc_run function. - check for true tree"""

    # Check that the first tree is equal to the last tree
    result = analyze.analyze_mcmc_run(
        mcmc_data=pure_mcmc_data,
        metric=analyze.Metrics.get("TrueTree"),
        base_tree=pure_mcmc_data.trees[0],
    )

    expected_result = (
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [True, False, False, False, False, False, False, False, False, True],
    )

    assert result == expected_result
