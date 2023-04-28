"""Sampler for Markov Chain Monte Carlo inference for mutation trees
    according to the SCITE model.

Note:
    This implementation assumes that the false positive
    and false negative rates are known and provided as input.
"""

import jax
from typing import Tuple
from pathlib import Path


import pyggdrasil.tree_inference._mcmc as mcmc
import pyggdrasil.tree_inference._logprob as logprob

from pyggdrasil.tree_inference._mcmc import MoveProbabilities
from pyggdrasil.tree_inference._tree import Tree
from pyggdrasil.tree_inference._interface import (
    MutationMatrix,
    JAXRandomKey,
    ErrorRates,
)
from pyggdrasil.serialize._to_json import save_mcmc_sample


def mcmc_sampler(
    rng_key: JAXRandomKey,
    init_tree: Tree,
    theta: ErrorRates,
    move_probabilities: MoveProbabilities,
    data: MutationMatrix,
    num_samples: int,
    output_dir: Path,
    num_burn_in: int = 0,
    thinning: int = 0,
    iteration: int = 0,
) -> None:
    """Sample mutation trees according to the SCITE model.

    Args:
        rng_key: random key for the MCMC sampler
        init_tree: initial tree to start the MCMC sampler from
        theta: \theta = (\alpha, \beta) error rates
        move_probabilities: probabilities for each move
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

    # curry logprobability function
    logprobability_fn = logprob.create_logprob(data, theta)

    # get initial state
    init_state = (
        iteration,
        rng_key,
        init_tree,
        logprobability_fn(init_tree),
    )

    # define loop body
    def body(
        state: Tuple[int, rng_key, Tree, float],
    ) -> Tuple[int, JAXRandomKey, Tree, float]:
        """Body of the MCMC loop.

        Args:
            state: current state of the MCMC sampler

        Returns:
            state: updated state of the MCMC sampler
        """

        # get current state
        iteration, rng_key_body, tree, logprobability = state
        iteration = +1

        # split random key to use one
        rng_key_run, rng_key_sample = jax.random.split(rng_key_body)

        # mcmc kernel
        tree, logprobability = mcmc._mcmc_kernel(
            rng_key_sample,
            tree,
            data,
            theta,
            move_probabilities,
            logprobability_fn,
            logprobability,
        )

        # burn-in - do not save samples in burning phase
        if iteration > num_burn_in:
            # save sample
            if iteration % thinning == 0:
                # pack sample
                # TODO: consider using xarray for storing the samples
                sample = {
                    "sample_no": iteration,
                    "tree": tree,
                    "log-probability": logprobability,
                    # note need jnp.ndarray.tolist(jnp.asarray(JAXRandomKey))
                    "rng_key_run": rng_key_run,
                }

                # TODO: implement to_json and from_json in
                #  serialize for the MCMC sampler
                save_mcmc_sample(sample, output_dir)

        return iteration, rng_key_run, tree, logprobability

    # conditional function for MCMC kernel
    def cond(state):
        """Condition for the MCMC loop."""
        return state[0] < num_samples

    # mcmc loop
    jax.lax.while_loop(cond, body, init_state)

    raise NotImplementedError("TODO: implement MCMC sampler")


# TODO: implement random tree generation may use the following:
# src/pyggdrasil/tree_inference/_simulate.generate_random_tree()
# Or networkX
