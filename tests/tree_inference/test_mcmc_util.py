"""Tests of the MCMC utility functions."""

# _mcmc_util.py
import pytest
import jax.random as random
import jax.numpy as jnp
import numpy as np


import pyggdrasil.tree_inference._mcmc_util as mcmc_util
import pyggdrasil.tree_inference._mcmc as mcmc
import pyggdrasil.tree_inference as tree_inf


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("n_nodes", [5, 10, 15])
def test_get_descendants(seed: int, n_nodes: int):
    """Test get_descendants against Floyd-Warshall algorithm."""
    # get random numbers keys
    rng = random.PRNGKey(seed)
    rng_tree, rng_nodes = random.split(rng, 2)
    # generate random tree
    adj_mat = tree_inf.generate_random_tree(rng_tree, n_nodes)
    # generate random nodes
    parent = int(random.randint(rng_nodes, shape=(1,), minval=0, maxval=n_nodes)[0])
    # assign labels
    tree = mcmc.Tree(adj_mat, jnp.arange(n_nodes))
    # get descendants with matrix exponentiation
    desc01 = mcmc_util._get_descendants(
        jnp.array(tree.tree_topology), jnp.array(tree.labels), parent
    )
    # get descendants with Floyd-Warshall
    adj_mat_prime = np.array(tree.tree_topology)
    np.fill_diagonal(adj_mat_prime, 1)
    desc02_idx = np.where(tree_inf.get_descendants(adj_mat_prime, parent) == 1)
    desc02 = tree.labels[desc02_idx]
    desc02 = desc02[desc02 != parent]

    # check that descendants are the same
    assert jnp.all(desc01 == desc02)
