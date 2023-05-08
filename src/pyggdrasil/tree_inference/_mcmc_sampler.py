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


import pyggdrasil.tree_inference._mcmc as mcmc
import pyggdrasil.tree_inference._logprob as logprob
import pyggdrasil.tree_inference._mcmc_util as mcmc_util
import pyggdrasil.serialize as serialize

from pyggdrasil.tree_inference._mcmc import MoveProbabilities
from pyggdrasil.tree_inference._tree import Tree
from pyggdrasil.tree_inference._interface import (
    MutationMatrix,
    JAXRandomKey,
    ErrorRates,
)


def mcmc_sampler(
    rng_key: JAXRandomKey,
    init_tree: Tree,
    error_rates: ErrorRates,
    move_probs: MoveProbabilities,
    data: MutationMatrix,
    num_samples: int,
    output_dir: Path,
    num_burn_in: int = 0,
    thinning: int = 0,
    iteration: int = 0,
    **kwargs,
) -> None:
    """Sample mutation trees according to the SCITE model.

    Args:
        rng_key: random key for the MCMC sampler
        init_tree: initial tree to start the MCMC sampler from
        error_rates: \theta = (\alpha, \beta) error rates
        move_probs: probabilities for each move
        data: observed mutation matrix to calculate the log-probability of,
            given current tree
        num_samples: number of samples to return
        output_dir: directory to save samples to
        num_burn_in: number of samples to discard before returning samples
        thinning: number of samples to discard between samples
        iteration: sample numer in chain, for restarting

    Returns:
        None
    """

    logging.info("Starting MCMC sampler.")

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

    logging.info(f"Initial state: {init_state}.")
    logging.info("Starting MCMC loop.")

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
            data,
            error_rates,
            move_probs,
            logprobability_fn,
            logprobability,
        )
        logging.info("Iteration: %d, log-probability: %f", iter_sample, logprobability)

        # burn-in - do not save samples in burning phase
        if iter_sample > num_burn_in:
            # save sample
            if iter_sample % thinning == 0:
                # pack sample
                sample = mcmc_util._pack_sample(iter_sample, tree, logprobability)
                # save sample
                serialize.save_mcmc_sample(
                    sample, output_dir, timestamp=kwargs["timestamp"]
                )
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

    logging.info("Finished MCMC sampler.")
