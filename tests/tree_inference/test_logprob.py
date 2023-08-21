"""Tests of the log-probability calculations."""
# _logprob.py

import jax.numpy as jnp
import jax.random as random
import pytest

import pyggdrasil.tree_inference._logprob as logprob
import pyggdrasil as yg
from pyggdrasil.tree_inference._tree import Tree
import pyggdrasil.tree_inference._tree as tr


def test_mutation_likelihood_man():
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
    mutation_likelihood = logprob._mutation_likelihood(
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
    mutation_likelihood = logprob._mutation_likelihood(
        mutation_matrix, ancestor_matrix, theta
    )

    # check shape
    n = 4
    m = 5
    assert mutation_likelihood.shape == (n - 1, m, n)

    print(mutation_likelihood[:, :, 0])

    # check total prob.
    total_sum = jnp.einsum("ijk->", mutation_likelihood)
    assert jnp.equal(total_sum, 40)


@pytest.mark.parametrize("alpha,", [0.1, 0.3])
@pytest.mark.parametrize("beta,", [0.2, 0.4])
def test_mutation_likelihood_2D(alpha, beta):
    """Test _mutation_likelihood using a 2D example.
    Checks for expected elements in the matrix."""

    adj_mat = jnp.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]])
    labels = jnp.array([0, 1, 2, 3])
    tree = Tree(adj_mat, labels)

    # cell j=0 attached to node 1
    data_true = jnp.array([[0], [1], [0]])
    # cell j=1 carries mutations different parts of tree - impossible given tree
    data_false = jnp.array([[1], [0], [1]])

    # expected mutation likelihoods - true
    p_true = jnp.array(
        [
            [beta, 1 - alpha, 1 - alpha, 1 - alpha],
            [1 - beta, 1 - beta, alpha, alpha],
            [1 - alpha, 1 - alpha, beta, 1 - alpha],
        ]
    )
    # expected mutation likelihoods - false
    p_false = jnp.array(
        [
            [1 - beta, alpha, alpha, alpha],
            [beta, beta, 1 - alpha, 1 - alpha],
            [alpha, alpha, 1 - beta, alpha],
        ]
    )

    p_mat_true = get_mutation_likelihood(tree, data_true, (alpha, beta))
    p_mat_false = get_mutation_likelihood(tree, data_false, (alpha, beta))

    assert jnp.all(p_mat_true == p_true)
    assert jnp.all(p_mat_false == p_false)
    assert jnp.einsum("ik->", p_mat_true) > jnp.einsum("ik->", p_mat_false)


def get_mutation_likelihood(tree, mutation_mat, theta):
    """Get mutation likelihood for a given tree and data - in 2D"""

    # A(T) - get ancestor matrix
    ancestor_mat = tr._get_ancestor_matrix(tree.tree_topology)
    # get mutation likelihood
    mutation_likelihood = logprob._mutation_likelihood(
        mutation_mat, ancestor_mat, theta
    )

    # kill 3rd dimension
    mutation_likelihood = jnp.einsum("ijk->ik", mutation_likelihood)

    return mutation_likelihood


def test_logprobability_fn():
    """Test logprobability_fn manually."""
    # define tree
    adj_mat = jnp.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]])
    labels = jnp.array([0, 1, 2, 3])
    tree = Tree(adj_mat, labels)

    # define mutation matrix
    mutation_matrix = jnp.array([[0, 0, 0, 1, 0], [0, 0, 1, 1, 1], [0, 1, 0, 0, 0]])

    # define error rates
    alpha = 0.5
    beta = 0.5
    theta = (alpha, beta)
    # should result in a (n)* (n+1) * (m) tensor or all elements 0.5

    # expected
    # manually compute expected logprob
    expected = 5 * jnp.log(4 * jnp.exp(3 * jnp.log(0.5)))

    # define logprob fn
    log_prob = logprob.logprobability_fn(mutation_matrix, tree, theta)
    assert jnp.equal(log_prob, expected)


def test_logprobability_fn_direction():
    """Test logprobability_fn, test if error in mutation matrix gives worse logprob."""
    # define tree
    adj_mat = jnp.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]])
    labels = jnp.array([0, 1, 2, 3])
    tree = Tree(adj_mat, labels)

    # define mutation matrix
    mutation_matrix_true = jnp.array(
        [[0, 0, 0, 1, 0], [0, 0, 1, 1, 1], [0, 1, 0, 0, 0]]
    )
    mutation_matrix_false = jnp.array(
        [[0, 0, 0, 0, 1], [1, 1, 1, 0, 0], [0, 0, 1, 0, 0]]
    )

    # define error rates
    alpha = 0.1
    beta = 0.3
    theta = (alpha, beta)

    # define logprob fn for true mutation matrix
    log_prob_true = logprob.logprobability_fn(mutation_matrix_true, tree, theta)

    # define logprob fn for false mutation matrix
    log_prob_false = logprob.logprobability_fn(mutation_matrix_false, tree, theta)

    assert log_prob_true > log_prob_false


def test_logprobability_fn_exact_m2n3():
    """Test logprobability_fn manually.

    Manual Logprob calculation for a tree with 2 mutations and 3 cells.
    See https://github.com/cbg-ethz/PYggdrasil/pull/149 for calculation.

    """
    # define tree
    adj_mat = jnp.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]])
    labels = jnp.array([0, 1, 2, 3])
    tree = Tree(adj_mat, labels)

    # define mutation matrix
    mutation_matrix = jnp.array([[0, 1], [1, 1], [0, 0]])

    # define error rates
    alpha = 0.1
    beta = 0.3
    theta = (alpha, beta)

    # expected
    # manually compute expected logprob
    # expected = -0.79861
    expected = jnp.log(
        0.3 * 0.7 * 0.9 + 0.9 * 0.7 * 0.9 + 0.9 * 0.1 * 0.3 + 0.9 * 0.1 * 0.9
    ) + jnp.log(0.7 * 0.7 * 0.9 + 0.1 * 0.7 * 0.9 + 0.1 * 0.1 * 0.3 + 0.1 * 0.1 * 0.9)

    # define logprob fn
    log_prob = logprob.logprobability_fn(mutation_matrix, tree, theta)
    assert jnp.isclose(log_prob, expected, atol=1e-10)


def test_mutation_likelihood_fn_exact_m2n3():
    """Test mutation_likelihood_fn manually.

    Manual mutation likelihood calculation for a tree with 2 mutations and 3 cells.
    See https://github.com/cbg-ethz/PYggdrasil/pull/149 for calculation.
    """

    # define tree
    adj_mat = jnp.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]])
    labels = jnp.array([0, 1, 2, 3])
    tree = Tree(adj_mat, labels)

    # get ancestor matrix
    ancestor_mat = tr._get_ancestor_matrix(tree.tree_topology)

    # define mutation matrix
    mutation_mat = jnp.array([[0, 1], [1, 1], [0, 0]])

    # define error rates
    alpha = 0.1
    beta = 0.3
    theta = (alpha, beta)

    # get mutation likelihood
    mutation_likelihood = logprob._mutation_likelihood(
        mutation_mat, ancestor_mat, theta
    )

    # expected
    # manually compute expected mutation likelihood
    expected = jnp.array(
        [
            # n / i = 0
            [
                # m / j = 0
                [0.3, 0.9, 0.9, 0.9],
                # m / j = 1
                [0.7, 0.1, 0.1, 0.1],
            ],
            # n / i = 1
            [
                # m / j = 0
                [0.7, 0.7, 0.1, 0.1],
                # m / j = 1
                [0.7, 0.7, 0.1, 0.1],
            ],
            # n / i = 2
            [
                # m / j = 0
                [0.9, 0.9, 0.3, 0.9],
                # m / j = 1
                [0.9, 0.9, 0.3, 0.9],
            ],
        ]
    )

    assert jnp.isclose(mutation_likelihood, expected, atol=1e-10).all()


def mutation_data_tree_error(
    n_cells,
    n_mutations,
    error_rates: yg.tree_inference.ErrorCombinations,
    seed,
) -> tuple:
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

    return (
        tree,
        error_rate,
        jnp.array(data["noisy_mutation_mat"]),
        jnp.array(data["perfect_mutation_mat"]),
    )


@pytest.mark.parametrize("n_cells", [3, 5, 50, 200])
@pytest.mark.parametrize("n_mutations", [2, 4, 15])
@pytest.mark.parametrize(
    "error_rates",
    [
        yg.tree_inference.ErrorCombinations.IDEAL,
        yg.tree_inference.ErrorCombinations.TYPICAL,
        yg.tree_inference.ErrorCombinations.LARGE,
    ],
)
@pytest.mark.parametrize("seed", [23, 890])
def test_orthogonal_log_probs(n_cells, n_mutations, error_rates, seed):
    """Test orthogonal_log_prob implementations

    Fast: logprob._logprobability_fn: uses einsum and logsumexp
    Slow: logprob._logprobability_fn_verify : uses jnp.sum
    """

    # define tree, error rates, and mutation matrix
    tree, error_rate, data, _ = mutation_data_tree_error(
        n_cells, n_mutations, error_rates, seed
    )

    # convert tree: TreeNode -> Tree
    tree = yg.tree_inference.Tree.tree_from_tree_node(tree)

    # run fast logprob
    logprob_fast = logprob.logprobability_fn(data, tree, error_rate)

    # run slow / no einsum logsumexp logprob
    logprob_slow = logprob._logprobability_fn_verify(data, tree, error_rate)

    # assert equal
    assert jnp.isclose(logprob_fast, logprob_slow, atol=1e-6)


@pytest.mark.parametrize("n_cells", [40])
@pytest.mark.parametrize("n_mutations", [2, 4])
@pytest.mark.parametrize("seed", [23, 2, 5])
def test_logprob_no_noise_many_cells_wrong_tree(
    n_cells: int, n_mutations: int, seed: int
):
    """Test if given no noise and many cell small trees can be identified.
    make a small true tree, and compare its likelihood to that of a mcmc 1 tree."""

    error_rate = yg.tree_inference.ErrorCombinations.IDEAL

    tree_n, error_rate, _, data = mutation_data_tree_error(
        n_cells, n_mutations, error_rate, seed
    )

    tree = Tree.tree_from_tree_node(tree_n)

    rng = random.PRNGKey(seed)

    # until we found a different tree
    tree_mcmc01_nd = yg.tree_inference.evolve_tree_mcmc(tree_n, 1, rng)
    while yg.compare_trees(tree_n, tree_mcmc01_nd):  # type: ignore
        rng, _ = random.split(rng)
        tree_mcmc01_nd = yg.tree_inference.evolve_tree_mcmc(tree_n, 1, rng)

    tree_mcmc01 = Tree.tree_from_tree_node(tree_mcmc01_nd)

    # run logprob on true tree
    logprob_true_tree = logprob.logprobability_fn(data, tree, error_rate)

    # run logprob on mcmc tree
    logprob_mcmc_tree = logprob.logprobability_fn(data, tree_mcmc01, error_rate)

    print(f"\ntrue tree:\n {tree_n}\nmcmc tree:\n {tree_mcmc01_nd}")
    print(f"\nIs the same tree: " f"{yg.compare_trees(tree_n, tree_mcmc01_nd)}")
    print(f"\ntrue tree: {logprob_true_tree}\nmcmc tree: {logprob_mcmc_tree}")
    assert logprob_true_tree >= logprob_mcmc_tree
