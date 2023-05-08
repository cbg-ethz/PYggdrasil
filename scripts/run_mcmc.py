#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the MCMC sampler for tree inference
Using the given data, error rates and initial tree.

As per definition in the SCITE Jahn et al. 2016.

Allows to start an MCMC chain from a given tree provided as
a dumped tree json file or generate a random tree.

Example Usage:
poetry run python ../scripts/run_mcmc.py

# TODO: add example usage

"""

import argparse
import jax.random as random
import json
import jax.numpy as jnp
import logging

from pathlib import Path
from datetime import datetime

import pyggdrasil.tree_inference as tree_inf

from pyggdrasil.tree_inference import (
    MutationMatrix,
    mcmc_sampler,
    MoveProbabilities,  # type: ignore
)


def create_parser() -> argparse.Namespace:
    """
    Parser for required input user.

    Returns:
        args: argparse.Namespace
    """

    parser = argparse.ArgumentParser(description="Run MCMC sampler for tree inference.")
    parser.add_argument(
        "--seed",
        required=False,
        help="Seed for random chain ( and tree generation if no tree is provided).",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--init_tree_fp",
        required=False,
        help="Fullpath of initial tree to start the MCMC sampler from.",
        default=None,
    )

    parser.add_argument("--config_fp", required=True, help="Config file path", type=str)

    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory to save the MCMC samples.",
        type=str,
    )

    parser.add_argument(
        "--data_fp", required=True, help="File path containing the data.", type=str
    )

    args = parser.parse_args()

    return args


def get_mutation_matrix(data_fp: str) -> MutationMatrix:
    """Load the mutation matrix from file.

    Args:
        data_fp: str
            Filepath of the data.

    Returns:
        mut_mat: MutationMatrix
            observed mutation matrix
    """

    # load data from file to json object
    with open(data_fp, "r") as f:
        data = json.load(f)

    # convert json object to mutation matrix
    mut_mat = data["noisy_mutation_mat"]  # TODO: adjust to flexible title
    # convert to array
    mut_mat = jnp.array(mut_mat)

    return mut_mat


def run_chain(params: argparse.Namespace, config: dict) -> None:
    """Run the MCMC sampler for tree inference.

    Args:
        params: argparse.Namespace
            input parameters from parser for MCMC sampler
        config: dict
            config dictionary

    Returns:
        None
    """

    # Set random seed
    rng = random.PRNGKey(params.seed)

    # load observed mutation matrix data from file
    mut_mat = jnp.array(get_mutation_matrix(params.data_fp))

    # check if init tree is provided
    if params.init_tree_fp is not None:
        # infer dimensions of tree from data
        n_mutations, m_cells = mut_mat.shape
        # split rng key
        rng_tree, rng = random.split(rng, 2)
        #  generate random trees (uniform sampling) as adjacency matrix
        #  / add +1 for root
        tree = tree_inf.generate_random_tree(rng_tree, n_nodes=n_mutations + 1)
        tree = jnp.array(tree)
        # make Tree
        labels = jnp.arange(n_mutations + 1)
        init_tree = tree_inf.Tree(tree, labels)

    else:
        # parse tree from file given as input
        # TODO: implement load tree from file - mcmc sample
        raise NotImplementedError("load init tree from file not implemented yet.")

    # Make Move Probabilities
    prune_and_reattach = config["move_probs"]["prune_and_reattach"]
    swap_node_labels = config["move_probs"]["swap_node_labels"]
    swap_subtrees = config["move_probs"]["swap_subtrees"]
    move_probs = MoveProbabilities(prune_and_reattach, swap_node_labels, swap_subtrees)

    # run mcmc sampler
    mcmc_sampler(
        rng_key=rng,
        data=mut_mat,
        error_rates=(config["fpr"], config["fnr"]),
        move_probs=move_probs,
        num_samples=config["num_samples"],
        num_burn_in=config["burn_in"],
        output_dir=Path(params.out_dir),
        thinning=params.thinning,
        init_tree=init_tree,
    )

    print("Done!")


def get_config(config_fp: str) -> dict:
    """Load the config file.

    Args:
        config_fp: str
            Filepath of the config file.
    Returns:
        config: dict
            config file as dict
    """
    # load config from file to json object
    with open(config_fp, "r") as f:
        config = json.load(f)

    return config


def main() -> None:
    """
    Main function.
    """
    # Parse command line arguments
    params = create_parser()

    # TODO: check params

    # load config file
    config = get_config(params.config_fp)

    # get date and time for output file
    time = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = Path(params.out_dir)
    fullpath = out_dir / f"mcmc_run_{time}.log"

    # Set up logging
    logging.basicConfig(filename=fullpath, level=logging.INFO)
    logging.info("Starting Session")

    # Run the simulation and save to disk
    run_chain(params, config)

    logging.info("Finished Session")


#########################################################################################
# MAIN
########################################################################################
if __name__ == "__main__":
    main()
