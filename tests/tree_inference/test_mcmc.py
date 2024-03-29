"""Tests of the MCMC functions."""
# _mcmc.py
import pytest
import jax.random as random
import jax.numpy as jnp

import pyggdrasil.tree_inference._mcmc as mcmc
import pyggdrasil.tree_inference._tree_generator
import pyggdrasil.tree_inference._tree_generator as tree_gen
import pyggdrasil.tree_inference._tree as tr
import pyggdrasil.tree_inference._mcmc_util as mcmc_util
from pyggdrasil.tree_inference import MoveProbConfigOptions, MoveProbabilities
from pyggdrasil.tree_inference._tree import Tree


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_swap_node_labels_move(seed: int):
    """Test swap_node_labels."""
    # get random numbers keys
    rng = random.PRNGKey(seed)
    rng_tree, rng_nodes = random.split(rng, 2)
    # generate random tree
    n_nodes = 10
    adj_mat = tree_gen._generate_random_tree_adj_mat(rng_tree, n_nodes)
    # generate random nodes - NB: root may not be swapped, hence n_nodes-1
    node1, node2 = random.randint(rng_nodes, shape=(2,), minval=0, maxval=n_nodes - 1)
    # assign labels
    tree01 = mcmc.Tree(jnp.array(adj_mat), jnp.arange(n_nodes))
    # swap labels
    tree01_labels = tree01.labels
    tree02 = mcmc._swap_node_labels_move(tree01, node1, node2)
    tree02_labels = tree02.labels
    # check that labels are swapped
    assert tree01_labels[node1] == tree02_labels[node2]
    assert tree01_labels[node2] == tree02_labels[node1]
    # check that other labels are unchanged
    for i in range(n_nodes):
        if i not in [node1, node2]:
            assert tree01_labels[i] == tree02_labels[i]


def test_prune_and_reattach_move():
    """Test prune_and_reattach_move. - manual test"""
    # Original tree
    tree_adj = jnp.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
        ]
    )
    labels = jnp.array([6, 5, 4, 3, 2, 1])
    tree = Tree(tree_adj, labels)

    # new tree
    new_tree = mcmc._prune_and_reattach_move(tree, pruned_node=2, attach_to=3)

    new_tree_corr = Tree(
        jnp.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ]
        ),
        jnp.array([6, 5, 4, 3, 2, 1]),
    )

    assert jnp.array_equal(new_tree.tree_topology, new_tree_corr.tree_topology)
    assert jnp.array_equal(new_tree.labels, new_tree_corr.labels)


def test_prune_and_reattach_moves():
    """Test mcmc.prune_and_reattach_moves. against
    mcmc_util.prune_and_reattach_move. - manual test

    Note only different lineage moves are to be tested with the mcmc util function."""
    # Original tree
    tree_adj = jnp.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
        ]
    )
    labels = jnp.array([6, 5, 4, 3, 2, 1])
    tree = Tree(tree_adj, labels)

    # new tree
    new_tree_1 = mcmc._prune_and_reattach_move(tree, pruned_node=2, attach_to=3)

    # new tree
    new_tree_2 = mcmc_util._prune_and_reattach_subtree(tree, pruned_node=2, attach_to=3)
    new_tree_2_resort = tr._reorder_tree(new_tree_2, new_tree_1.labels)

    assert jnp.array_equal(new_tree_1.tree_topology, new_tree_2_resort.tree_topology)
    assert jnp.array_equal(new_tree_1.labels, new_tree_2_resort.labels)


@pytest.mark.parametrize("seed", [4, 42])
@pytest.mark.parametrize("n_nodes", [4, 5])
def test_prune_and_reattach_moves_auto(seed: int, n_nodes: int):
    """Test mcmc.prune_and_reattach_moves. against
    mcmc_util.prune_and_reattach_move. - automatic test"""
    rng = random.PRNGKey(seed)
    rng_adj, rng_nodes, rng_move = random.split(rng, 3)
    tree_adj = tree_gen._generate_random_tree_adj_mat(rng_adj, n_nodes)
    tree_adj = jnp.array(tree_adj)
    labels = random.permutation(rng_nodes, jnp.arange(n_nodes))
    tree = Tree(tree_adj, labels)
    print(tree)
    tree.print_topo()

    node_pair_found = False
    pruned_node = -1000
    attach_to = -1000
    while node_pair_found is False:
        rng_move, rng_prune, rng_attach = random.split(rng_move, 3)
        # choose a node to prune - excluding root
        pruned_node = int(random.randint(rng_prune, (), minval=0, maxval=n_nodes - 1))
        # choose a node to attach to - including root
        attach_to = int(random.randint(rng_attach, (), minval=0, maxval=n_nodes))
        # check that attach_to node is not a descendant of pruned_node
        desc = tr.get_descendants(tree_adj, labels, pruned_node, include_parent=True)
        if attach_to not in desc:
            node_pair_found = True

    # new tree
    new_tree_1 = mcmc._prune_and_reattach_move(
        tree, pruned_node=pruned_node, attach_to=attach_to
    )
    print(new_tree_1.tree_topology)
    new_tree_1.print_topo()
    # new tree
    new_tree_2 = mcmc_util._prune_and_reattach_subtree(
        tree, pruned_node=pruned_node, attach_to=attach_to
    )
    new_tree_2_resort = tr._reorder_tree(new_tree_2, new_tree_1.labels)
    new_tree_2_resort.print_topo()

    assert jnp.array_equal(new_tree_1.tree_topology, new_tree_2_resort.tree_topology)
    assert jnp.array_equal(new_tree_1.labels, new_tree_2_resort.labels)


def test_swap_subtrees_move():
    """Test mcmc.swap_subtrees_move  - manual test"""
    # Original tree
    tree_adj = jnp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0],
        ]
    )
    labels = jnp.array([8, 7, 6, 5, 4, 3, 2, 1])
    tree = Tree(tree_adj, labels)

    # new tree
    new_tree = mcmc._swap_subtrees_move_diff_lineage(tree, node1=5, node2=3)

    new_tree_corr = Tree(
        jnp.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 1, 0],
            ]
        ),
        jnp.array([8, 7, 6, 5, 4, 3, 2, 1]),
    )

    assert jnp.array_equal(new_tree.tree_topology, new_tree_corr.tree_topology)
    assert jnp.array_equal(new_tree.labels, new_tree_corr.labels)


def test_swap_subtrees_move_fig16_diff_lineage():
    """Test mcmc.swap_subtrees_move  - manual test
    equal to the simple case in fig 16 of the paper,
    where two nodes are not of the same lineage,
    are not parent-child relationship."""

    # Original tree
    tree_adj = jnp.array(
        [  # 1, 2, 3, 4, 5, 6, 7, 8, R
            [0, 1, 0, 1, 0, 0, 1, 0, 0],  # 1
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
            [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 5
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
            [1, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
            [0, 0, 1, 0, 1, 0, 0, 1, 0],  # R
        ]
    )
    labels = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    tree = Tree(tree_adj, labels)

    # new tree
    new_tree = mcmc._swap_subtrees_move_diff_lineage(tree, node1=5, node2=1)

    new_tree_corr = Tree(
        jnp.array(
            [  # 1, 2, 3, 4, 5, 6, 7, 8, R
                [0, 1, 0, 1, 0, 0, 1, 0, 0],  # 1
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
                [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 5
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
                [0, 0, 0, 0, 1, 0, 0, 0, 0],  # 8
                [1, 0, 1, 0, 0, 0, 0, 1, 0],  # R
            ]
        ),
        labels=jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    )

    assert jnp.array_equal(new_tree.tree_topology, new_tree_corr.tree_topology)
    assert jnp.array_equal(new_tree.labels, new_tree_corr.labels)


@pytest.mark.parametrize("seed", [2, 42, 69])
def test_swap_subtrees_move_fig17_nested_subtrees(seed):
    """Test mcmc.swap_subtrees_move  - manual test
    equal to the simple case in fig 17 of the paper,
    where two nodes are not of the same lineage."""

    # Original tree
    tree_adj = jnp.array(
        [  # 1, 2, 3, 4, 5, 6, 7, 8, R
            [0, 1, 0, 1, 0, 0, 1, 0, 0],  # 1
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
            [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 5
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
            [1, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
            [0, 0, 1, 0, 1, 0, 0, 1, 0],  # R
        ]
    )
    labels = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    tree = Tree(tree_adj, labels)

    # make jax random key
    key = random.PRNGKey(seed)

    # new tree
    new_tree = mcmc._swap_subtrees_move_same_lineage(tree, node1=8, node2=1, key=key)

    # if the i node was attached to node k itself, by uniform sampling
    new_tree_corr1 = Tree(
        jnp.array(
            [  # 1, 2, 3, 4, 5, 6, 7, 8, R
                [0, 1, 0, 1, 0, 0, 1, 1, 0],  # 1
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
                [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 5
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
                [1, 0, 1, 0, 1, 0, 0, 0, 0],  # R
            ]
        ),
        labels=jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    )
    # if the i node was attached node 2
    new_tree_corr2 = Tree(
        jnp.array(
            [  # 1, 2, 3, 4, 5, 6, 7, 8, R
                [0, 1, 0, 1, 0, 0, 1, 0, 0],  # 1
                [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 2
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
                [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 5
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
                [1, 0, 1, 0, 1, 0, 0, 0, 0],  # R
            ]
        ),
        labels=jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    )
    # if the i node was attached node 4
    new_tree_corr3 = Tree(
        jnp.array(
            [  # 1, 2, 3, 4, 5, 6, 7, 8, R
                [0, 1, 0, 1, 0, 0, 1, 0, 0],  # 1
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
                [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 4
                [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 5
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
                [1, 0, 1, 0, 1, 0, 0, 0, 0],  # R
            ]
        ),
        labels=jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    )
    # if the i node was attached node 7
    new_tree_corr4 = Tree(
        jnp.array(
            [  # 1, 2, 3, 4, 5, 6, 7, 8, R
                [0, 1, 0, 1, 0, 0, 1, 0, 0],  # 1
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
                [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 5
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
                [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 7
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
                [1, 0, 1, 0, 1, 0, 0, 0, 0],  # R
            ]
        ),
        labels=jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    )

    # check if any of the four possible trees is the same as the new tree
    assert (
        jnp.array_equal(new_tree.tree_topology, new_tree_corr1.tree_topology)
        or jnp.array_equal(new_tree.tree_topology, new_tree_corr2.tree_topology)
        or jnp.array_equal(new_tree.tree_topology, new_tree_corr3.tree_topology)
        or jnp.array_equal(new_tree.tree_topology, new_tree_corr4.tree_topology)
    )
    # check if the labels are the same, just once for sanity
    assert jnp.array_equal(new_tree.labels, new_tree_corr1.labels)


def test_swap_node_labels_manual():
    """Test mcmc.swap_node_labels_move  - manual test
    Bug was found in deep tree mcmc runs."""

    # Original tree
    tree_adj = jnp.array(
        [
            # 3  0  1  2  4
            [0, 1, 0, 0, 0],  # 3
            [0, 0, 1, 0, 0],  # 0
            [0, 0, 0, 1, 0],  # 1
            [0, 0, 0, 0, 0],  # 2
            [1, 0, 0, 0, 0],  # 4
        ]
    )
    labels = jnp.array([3, 0, 1, 2, 4])
    tree = Tree(tree_adj, labels)

    new_tree = mcmc._swap_node_labels_move(tree, node1=0, node2=3)

    print("Tree after swap")
    print(new_tree)

    # Expected tree
    tree_adj = jnp.array(
        [
            # 0  3  1  2  4
            [0, 1, 0, 0, 0],  # 0
            [0, 0, 1, 0, 0],  # 3
            [0, 0, 0, 1, 0],  # 1
            [0, 0, 0, 0, 0],  # 2
            [1, 0, 0, 0, 0],  # 4
        ]
    )
    labels = jnp.array([0, 3, 1, 2, 4])
    expected_tree = Tree(tree_adj, labels)

    assert jnp.array_equal(new_tree.tree_topology, expected_tree.tree_topology)
    assert jnp.array_equal(new_tree.labels, expected_tree.labels)


def test_prune_and_reattach_deep_tree():
    """Test prune and reattach move again - manual test"""

    # Original tree
    tree_adj = jnp.array(
        [
            # 1  0  2  3  4
            [0, 1, 0, 0, 0],  # 1
            [0, 0, 1, 0, 0],  # 0
            [0, 0, 0, 1, 0],  # 2
            [0, 0, 0, 0, 0],  # 3
            [1, 0, 0, 0, 0],  # 4
        ]
    )
    labels = jnp.array([1, 0, 2, 3, 4])
    tree = Tree(tree_adj, labels)

    new_tree = mcmc._prune_and_reattach_move(tree, pruned_node=0, attach_to=4)

    # Expected tree
    tree_adj = jnp.array(
        [
            # 1  0  2  3  4
            [0, 0, 0, 0, 0],  # 1
            [0, 0, 1, 0, 0],  # 0
            [0, 0, 0, 1, 0],  # 2
            [0, 0, 0, 0, 0],  # 3
            [1, 1, 0, 0, 0],  # 4
        ]
    )
    labels = jnp.array([1, 0, 2, 3, 4])
    expected_tree = Tree(tree_adj, labels)

    assert jnp.array_equal(new_tree.tree_topology, expected_tree.tree_topology)
    assert jnp.array_equal(new_tree.labels, expected_tree.labels)


def make_tree(n: int, seed: int, tree_type: str) -> Tree:
    """Make a tree for testing."""

    rng = random.PRNGKey(seed)

    if tree_type == "r":
        adj_mat = jnp.array(tree_gen._generate_random_tree_adj_mat(rng, n))
        labels = jnp.arange(n)
        return Tree(adj_mat, labels)
    elif tree_type == "s":
        adj_mat = jnp.array(tree_gen._generate_star_tree_adj_mat(n))
        labels = jnp.arange(n)
        return Tree(adj_mat, labels)
    else:
        adj_mat = jnp.array(tree_gen._generate_deep_tree_adj_mat(rng, n))
        labels = jnp.arange(n)
        return Tree(adj_mat, labels)


@pytest.mark.parametrize("tree_type", ["r", "s", "d"])
@pytest.mark.parametrize("n_nodes", [5, 10])
@pytest.mark.parametrize("tree_seed", [34])
@pytest.mark.parametrize("n_moves", [25])
@pytest.mark.parametrize("seed", [97])
def test_mcmc_kernel(tree_type, n_nodes, seed, n_moves, tree_seed):
    """Test mcmc kernel - i.e. test that 200 moves for some tree are all valid"""

    # Make a tree
    tree = make_tree(n_nodes, seed, tree_type)

    # define move probabilities
    move_probs = MoveProbConfigOptions.DEFAULT.value

    # get rng
    rng = random.PRNGKey(tree_seed)

    # make random log prob function
    def log_prob_fn(t: Tree) -> float:
        """Log prob function for testing. - dummy function"""
        return random.uniform(rng, shape=()).__float__()

    # Run the kernel
    for i in range(n_moves):
        rng, rng_now = random.split(rng)
        tree, _ = mcmc._mcmc_kernel(
            rng_now, tree, move_probs, log_prob_fn  # type: ignore
        )
        assert tr.is_valid_tree(tree)


def test_evolve_tree_mcmc():
    """Static Test evolve_tree_mcmc. - assures that imports are correct
    and that the function runs"""

    seed = 42
    rng = random.PRNGKey(seed)

    # generate random tree
    tree = pyggdrasil.tree_inference._tree_generator.generate_random_Tree(rng, 10)

    # define move probabilities
    move_probs = MoveProbabilities()

    # evolve tree
    tree_ev = mcmc._evolve_tree_mcmc(tree, 2, rng, move_probs)

    # check if the tree is still a tree
    assert not tr.is_same_tree(tree, tree_ev)
