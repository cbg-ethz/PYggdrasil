"""Tests of the log-probability calculations."""
# _logprob.py

import jax.numpy as jnp

import pyggdrasil.tree_inference._logprob as logprob
from pyggdrasil.tree_inference._tree import Tree
import pyggdrasil.tree_inference._tree as tr


def test_compute_mutation_likelihood():
    """Test _compute_mutation_likelihood manually

    Computes mutation likelihood given the true mutation matrix as data.
    Sets error rates to alpha to 1 and also beta to 1 each, to check for
    the correct total probability.
    """

    # define tree
    adj_mat = jnp.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]])
    labels = jnp.array([0, 1, 2, 3])
    tree = Tree(adj_mat, labels)
    ancestor_matrix = tr._get_ancestor_matrix(tree.tree_topology)
    ancestor_matrix_truth = jnp.array(
        [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [1, 1, 1, 1]]
    )

    assert jnp.all(ancestor_matrix == ancestor_matrix_truth)

    # cell attachment vector
    # sigma = jnp.array([3, 2, 1 ,0, 1])
    # get mutation matrix / data - get true mutation matrix
    mutation_matrix = jnp.array([[0, 0, 0, 1, 0], [0, 0, 1, 1, 1], [0, 1, 0, 0, 0]])

    # test for is mutation
    # define error rates to check outcome
    alpha = 1.0
    beta = 0.0
    theta = (alpha, beta)
    # thus we expect only for data=1 and expected mutation=1 matrix probability of 1
    # we expect 5 * n (=4) = 4 * j(=5) = 20 entries of unity in the final matrix

    # define mutation likelihoods
    # mutation_likelihoods = logprob._compute_mutation_likelihood(tree, ancestor_matrix)
    mutation_likelihood = logprob._compute_mutation_likelihood(
        mutation_matrix, ancestor_matrix, theta
    )

    # check shape
    n = 4
    m = 5
    assert mutation_likelihood.shape == (n - 1, m, n)

    print(mutation_likelihood[:, :, 0])

    # check total prob.
    total_sum = jnp.einsum("ijk->", mutation_likelihood)
    assert int(total_sum) == 20

    # test for is not mutation
    # define error rates to check outcome
    alpha = 0.0
    beta = 1.0
    theta = (alpha, beta)
    # thus we expect only for data=0 and expected mutation=0 matrix probability of 1
    # we expect 10 * n (=4) = 8 * j(=5) = 40 entries of unity in the final matrix

    # define mutation likelihoods
    # mutation_likelihoods = logprob._compute_mutation_likelihood(tree, ancestor_matrix)
    mutation_likelihood = logprob._compute_mutation_likelihood(
        mutation_matrix, ancestor_matrix, theta
    )

    # check shape
    n = 4
    m = 5
    assert mutation_likelihood.shape == (n - 1, m, n)

    print(mutation_likelihood[:, :, 0])

    # check total prob.
    total_sum = jnp.einsum("ijk->", mutation_likelihood)
    assert int(total_sum) == 40
