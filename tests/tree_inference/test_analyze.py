"""Tree MCMC sample analysis."""
import jax.random as random
import pytest
import jax.numpy as jnp
import xarray as xr

import pyggdrasil.tree_inference._analyze as analyze

from pyggdrasil.interface import MCMCSample
from pyggdrasil.tree_inference import generate_random_tree, Tree


@pytest.fixture
def mcmc_samples() -> list[MCMCSample]:
    """Generate a list of MCMCSample objects.

    of fixed tree size 10, linear increasing log-probabilities.
    and random tree topologies.
    """
    num_samples = 10

    # Generate and return a list of MCMCSample objects
    mcmc_samples = []

    # Linearly increasing log-probability values
    log_probabilities = jnp.linspace(-121.6, -100.0, num_samples)

    # Generate a list of random trees
    trees = [
        generate_random_tree(rng=random.PRNGKey(j), n_nodes=10)
        for j in range(num_samples)
    ]

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


def test_is_same_tree():
    """Test is_same_tree function.
    same tree with different label order
    """

    adj_mat1 = jnp.array(
        [
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0],
        ]
    )
    labels1 = jnp.array([3, 1, 2, 4, 5, 6, 0, 7])
    tree1 = Tree(adj_mat1, labels1)

    adj_mat2 = jnp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
        ]
    )
    labels2 = jnp.array([0, 1, 2, 3, 4, 5, 6, 7])
    tree2 = Tree(adj_mat2, labels2)

    assert analyze.is_same_tree(tree1, tree2)


def test_is_not_same_tree():
    """Test is_same_tree function.
    same tree with different label order
    """

    adj_mat1 = jnp.array(
        [
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0],
        ]
    )
    labels1 = jnp.array([3, 1, 2, 4, 5, 6, 0, 7])
    tree1 = Tree(adj_mat1, labels1)

    adj_mat2 = jnp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
        ]
    )
    labels2 = jnp.array([0, 1, 2, 5, 4, 3, 6, 7])
    tree2 = Tree(adj_mat2, labels2)

    assert analyze.is_same_tree(tree1, tree2) is False
