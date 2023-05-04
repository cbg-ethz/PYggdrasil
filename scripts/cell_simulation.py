#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate cell mutation matrices,
by generating random trees and sampling cell attachments
with noise if error rates are non-zero.

As per definition in the SCITE Jahn et al. 2016.

Example Usage:
poetry run python ../scripts/cell_simulation.py
--seed 42
--out_dir ../data --n_trees 3 --n_cells 100 --n_mutations 8
 --strategy UNIFORM_INCLUDE_ROOT --alpha 0.01 --beta 0.02
 --na_rate 0.01 --observe_homozygous True --verbose
"""

import argparse
import logging

import jax.random as random
import numpy as np
from jax.random import PRNGKeyArray
import json
import os

import pyggdrasil.serialize as serialize
import pyggdrasil.tree_inference as tree_inf


class NpEncoder(json.JSONEncoder):
    """Encoder for numpy types."""

    def default(self, obj):
        """Default encoder."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            # 👇️ alternatively use str()
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def t_or_f(arg):
    """Converts string input to boolean.

    Args:
        arg: str
    Returns:
        bool

    """
    ua = str(arg).upper()
    if "TRUE".startswith(ua):
        return True
    elif "FALSE".startswith(ua):
        return False
    else:
        logging.warning("boolean argument not valid")


def create_parser() -> argparse.Namespace:
    """
    Parser for required input user.

    Returns:
        args: argparse.Namespace
    """

    parser = argparse.ArgumentParser(
        description="Generate test cell mutation matrices."
    )
    parser.add_argument(
        "--seed",
        required=False,
        help="Seed for random generation",
        type=int,
        default=42,
    )
    parser.add_argument("--out_dir", required=True, help="Path to output directory")
    parser.add_argument("--n_trees", required=True, help="Number of trees", type=int)
    parser.add_argument("--n_cells", required=True, help="Number of cells", type=int)
    parser.add_argument(
        "--n_mutations", required=True, help="Number of mutations", type=int
    )

    parser.add_argument(
        "--strategy",
        required=False,
        choices=["UNIFORM_INCLUDE_ROOT", "UNIFORM_EXCLUDE_ROOT"],
        help="Cell attachment strategy",
        default="UNIFORM_INCLUDE_ROOT",
    )

    parser.add_argument(
        "--alpha", required=True, help="False negative rate", type=float
    )
    parser.add_argument("--beta", required=True, help="False positive rate", type=float)

    parser.add_argument(
        "--na_rate", required=True, help="Missing entry rate", type=float
    )

    parser.add_argument(
        "--observe_homozygous",
        required=False,
        default=False,
        help="Observing homozygous mutations",
        choices=[False, True],
        type=t_or_f,
    )

    parser.add_argument(
        "--verbose",
        help="Print trees and full save path, By default False.",
        action="store_true",
    )

    args = parser.parse_args()
    return args


def compose_save_name(params: argparse.Namespace, *, tree_no: int) -> str:
    """Composes save name for the results."""
    save_name = (
        f"seed_{params.seed}_"
        f"n_trees_{params.n_trees}_"
        f"n_cells_{params.n_cells}_"
        f"n_mutations_{params.n_mutations}_"
        f"alpha_{params.alpha}_"
        f"beta_{params.beta}_"
        f"na_rate_{params.na_rate}_"
        f"observe_homozygous_{params.observe_homozygous}_"
        f"strategy_{params.strategy}"
    )
    if tree_no is not None:
        save_name += f"_tree_{tree_no}"
    return save_name


def gen_sim_data(
    params: argparse.Namespace,
    rng: PRNGKeyArray,
    *,
    tree_no: int,
) -> None:
    """
    Generates cell mutation matrix for one tree and writes to file.

    Args:
        params: input parameters from parser
            input parameters from parser for simulation
        rng: JAX random number generator
        tree_no: int - optional
            tree number if a series is generated
    Returns:
        None
    """
    ############################################################################
    # Parameters
    ############################################################################
    out_dir = params.out_dir
    n_cells = params.n_cells
    n_mutations = params.n_mutations
    alpha = params.alpha
    beta = params.beta
    na_rate = params.na_rate
    observe_homozygous = params.observe_homozygous
    strategy = params.strategy
    verbose = params.verbose

    ############################################################################
    # Random Seeds
    ############################################################################
    rng_tree, rng_cell_attachment, rng_noise = random.split(rng, 3)

    ##############################################################################
    # Generate Trees
    ##############################################################################
    #  generate random trees (uniform sampling) as adjacency matrix / add +1 for root
    tree = tree_inf.generate_random_tree(rng_tree, n_nodes=n_mutations + 1)

    ##############################################################################
    # Attach Cells To Tree
    ###############################################################################
    # convert adjacency matrix to self-connected tree - in tree_inference format
    np.fill_diagonal(tree, 1)
    # define strategy
    strategy = tree_inf.CellAttachmentStrategy[strategy]
    # attach cells to tree - generate perfect mutation matrix
    perfect_mutation_mat = tree_inf.attach_cells_to_tree(
        rng_cell_attachment, tree, n_cells, strategy
    )

    ###############################################################################
    # Add Noise
    ################################################################################
    # add noise to perfect mutation matrix
    noisy_mutation_mat = None
    if (beta > 0) or (alpha > 0) or (na_rate > 0):
        noisy_mutation_mat = tree_inf.add_noise_to_perfect_matrix(
            rng_noise, perfect_mutation_mat, alpha, beta, na_rate, observe_homozygous
        )

    ################################################################################
    # Save Simulation Results
    ################################################################################
    # make save name and path from parameters
    filename = compose_save_name(params, tree_no=tree_no) + ".json"
    fullpath = os.path.join(out_dir, filename)
    # make output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # format tree for saving
    root = tree_inf.adjacency_to_root_dfs(tree)
    root_serialized = serialize.serialize_tree_to_dict(
        root, serialize_data=lambda x: None
    )

    # print tree if prompted verbose
    if verbose:
        print(root)

    # Save the data to a JSON file
    # Create a dictionary to hold matrices
    if noisy_mutation_mat is not None:
        data = {
            "adjacency_matrix": tree.tolist(),
            "perfect_mutation_mat": perfect_mutation_mat.tolist(),
            "noisy_mutation_mat": noisy_mutation_mat.tolist(),
            "tree": tree.tolist(),
            "root": root_serialized,
        }
    else:
        data = {
            "adjacency_matrix": tree.tolist(),
            "perfect_mutation_mat": perfect_mutation_mat.tolist(),
            "tree": tree.tolist(),
            "root": root_serialized,
        }

    # Save the data to a JSON file
    with open(fullpath, "w") as f:
        json.dump(data, f, cls=NpEncoder)

    # Print the path to the file if verbose
    if verbose:
        print(f"Saved simulation results to {fullpath}\n")


def run_sim(params: argparse.Namespace) -> None:
    """Generate {n_trees} of simulated data and save to disk.

    Args:
        params: argparse.Namespace
            input parameters from parser for simulation

    Returns:
        None
    """

    # Create a random number generator
    rng = random.PRNGKey(params.seed)

    keys = random.split(rng, params.n_trees)
    for i, key in enumerate(keys, 1):
        print(f"Generating simulation {i}/{len(keys)}")
        gen_sim_data(params, key, tree_no=i)

    # Print success message
    print(f"{ params.n_trees } trees generated successfully!")
    print("Done!")


def main() -> None:
    """
    Main function.
    """
    # Parse command line arguments
    params = create_parser()
    # Run the simulation and save to disk
    run_sim(params)


#########################################################################################
# MAIN
########################################################################################
if __name__ == "__main__":
    main()
