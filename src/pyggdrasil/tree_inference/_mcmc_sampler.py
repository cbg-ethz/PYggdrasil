"""Sampler for Markov Chain Monte Carlo inference for mutation trees
    according to the SCITE model.

Note:
    This implementation assumes that the false positive
    and false negative rates are known and provided as input.
"""

import jax
from typing import Tuple
from pathlib import Path
import logging

from jax import Array
from jax import numpy as jnp


from pyggdrasil.interface import JAXRandomKey

import pyggdrasil.tree_inference._mcmc as mcmc
import pyggdrasil.tree_inference._logprob as logprob
import pyggdrasil.tree_inference._mcmc_util as mcmc_util
import pyggdrasil.serialize as serialize

from pyggdrasil.tree_inference._mcmc import MoveProbabilities
from pyggdrasil.tree_inference._tree import Tree
from pyggdrasil.tree_inference._ordered_tree import OrderedTree
from pyggdrasil.tree_inference._interface import (
    MutationMatrix,
    ErrorRates,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def mcmc_sampler(
    rng_key: JAXRandomKey,
    init_tree: Tree,
    error_rates: ErrorRates,
    move_probs: MoveProbabilities,
    data: MutationMatrix,
    num_samples: int,
    out_fp: Path,
    num_burn_in: int = 0,
    thinning: int = 0,
    iteration: int = 0,
) -> None:
    """Sample mutation trees according to the SCITE model.

    Args:
        rng_key: random key for the MCMC sampler
        init_tree: initial tree to start the MCMC sampler from
        error_rates: \theta = (\fpr, \fnr) error rates
        move_probs: probabilities for each move
        data: observed mutation matrix to calculate the log-probability of,
            given current tree
        num_samples: number of samples to return
        out_fp: fullpath to output file (excluding file extension)
        num_burn_in: number of samples to discard before returning samples
        thinning: number of samples to discard between samples
        iteration: sample numer in chain, for restarting

    Returns:
        None
    """

    logger.info("Starting MCMC sampler.")

    # TODO: implement support for NAs and homozygous mutations
    # check data i.e. mutation matrix
    # check if matrix has any entries equal to 3 or 2
    # if so, raise error
    if jnp.any(jnp.logical_or(data == 3, data == 2)):
        raise ValueError(
            "Mutation matrix has entries equal to homozygous"
            " mutations or missing entries. "
            "These entries are currently not allowed."
            "Log-probability calculation does not yet support"
        )

    # assert that the number of mutations and the data matrix size match
    # no of nodes must equal the number of rows in the data matrix plus root truncated
    if not init_tree.labels.shape[0] == data.shape[0] + 1:
        raise AssertionError(
            "Number of mutations and data matrix size do not match.\n"
            f"tree {init_tree.labels.shape[0]} != data {data.shape[0]}"
        )

    # ensure the tree is ordered
    init_tree = OrderedTree.from_tree(init_tree)

    # curry logprobability function
    logprobability_fn = logprob.create_logprob(data, error_rates)

    # get initial state
    init_state = (
        iteration,
        rng_key,
        init_tree.tree_topology,
        init_tree.labels,
        logprobability_fn(init_tree),
    )

    logger.info("Starting MCMC loop.")

    # define loop body
    def body(
        state: Tuple[int, JAXRandomKey, Array, Array, float],
    ) -> Tuple[int, JAXRandomKey, Array, Array, float]:
        """Body of the MCMC loop.

        Args:
            state: tuple containing the current state of the MCMC sampler
                - iteration: current iteration
                - rng_key: random key for the MCMC sampler
                - Array: current tree, adjacency matrix
                - Array: labels of the tree
                - float: log-probability of the current tree

        Returns:
                 updated state: tuple containing the current state of the MCMC sampler
                - iteration: current iteration
                - rng_key: random key for the MCMC sampler
                - Array: current tree, adjacency matrix
                - Array: labels of the tree
                - float: log-probability of the current tree
        """

        # get current state
        iter_sample, rng_key_body, topo, labels, logprobability = state
        iter_sample = iter_sample + 1

        # make Tree
        tree = Tree(tree_topology=topo, labels=labels)

        # split random key to use one
        rng_key_run, rng_key_sample = jax.random.split(rng_key_body)

        # mcmc kernel
        tree, logprobability = mcmc._mcmc_kernel(
            rng_key_sample,
            tree,
            move_probs,
            logprobability_fn,
            logprobability,
        )
        logging.info(
            "Iteration: {:d}, log-probability: {:.4f}".format(
                iter_sample, logprobability
            )
        )

        # burn-in - do not save samples in burning phase
        if iter_sample > num_burn_in:
            # save sample
            if iter_sample % thinning == 0:
                # pack sample
                sample = mcmc_util._pack_sample(iter_sample, tree, logprobability)
                # save sample
                serialize.save_mcmc_sample(sample, out_fp)
                logging.info("Saved sample %d.", iter_sample)

        # return updated state
        topo = tree.tree_topology
        labels = tree.labels
        updated_state = (iter_sample, rng_key_run, topo, labels, logprobability)

        return updated_state

    # conditional function for MCMC kernel
    def cond_fn(state):
        """Condition for the MCMC loop."""
        iter_sample, _, _, _, _ = state
        return jax.lax.lt(iter_sample, num_samples)

    # mcmc loop
    # TODO: use jax.lax.while_loop instead of while_loop,
    # requires further modification of _get_descendants at least
    # jax.lax.while_loop(cond_fn, body, init_state)

    def while_loop(cond_fun, body_fun, init_val):
        """While loop for MCMC sampler."""
        val = init_val
        while cond_fun(val):
            val = body_fun(val)
        return val

    while_loop(cond_fn, body, init_state)

    logger.info("Finished MCMC sampler.")
