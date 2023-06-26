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
--out_dir ../data --n_cells 100 --init_tree ../data/T_d_10_123.json
 --strategy UNIFORM_INCLUDE_ROOT --fpr 0.01 --fnr 0.02
 --na_rate 0.01 --observe_homozygous True --verbose
"""

import argparse
import logging

import jax.random as random
import json
import os

from pathlib import Path

import pyggdrasil.tree_inference as tree_inf
import pyggdrasil.serialize as serialize

from pyggdrasil.tree_inference import (
    CellSimulationModel,
    CellSimulationId,
    TreeId,
    CellAttachmentStrategy,
)
from pyggdrasil.serialize import JnpEncoder


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
        help="Seed for cell attachment and noise generation",
        type=int,
        default=42,
    )
    parser.add_argument("--out_dir", required=True, help="Path to output directory")
    parser.add_argument("--n_cells", required=True, help="Number of cells", type=int)

    parser.add_argument(
        "--strategy",
        required=False,
        choices=["UIR", "UXR"],
        help="Cell attachment strategy,"
        " UIR: UNIFORM_INCLUDE_ROOT,"
        " UXR: UNIFORM_EXCLUDE_ROOT,"
        " default: UIR",
        default="UNIFORM_INCLUDE_ROOT",
    )

    parser.add_argument("--fpr", required=True, help="False positive rate", type=float)
    parser.add_argument("--fnr", required=True, help="False negative rate", type=float)

    parser.add_argument(
        "--na_rate", required=True, help="Missing entry rate", type=float
    )

    parser.add_argument(
        "--true_tree_fp",
        required=True,
        help="Tree to use as truth to sample from - "
        "requires filename with tree_inference.TreeId format",
        type=str,
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


def run_sim(params: argparse.Namespace) -> None:
    """Generate {n_trees} of simulated data and save to disk.

    Args:
        params: argparse.Namespace
            input parameters from parser for simulation

    Returns:
        None
    """

    ###############################
    # Get Tree information from TreeId
    ###############################
    # make tree id from tree path
    true_tree_fp = Path(params.true_tree_fp)
    # Get the filename without the file extension and directories
    tt_filename = true_tree_fp.name
    tt_filename_without_extension = tt_filename.split(".")[0]

    tree_id = TreeId.from_str(tt_filename_without_extension)

    # Load the tree from the file
    tree = serialize.read_tree_node(true_tree_fp)

    # parse strategy
    if params.strategy == "UIR":
        params.strategy = CellAttachmentStrategy.UNIFORM_INCLUDE_ROOT
    elif params.strategy == "UXR":
        params.strategy = CellAttachmentStrategy.UNIFORM_EXCLUDE_ROOT
    else:
        raise ValueError(f"Invalid strategy: {params.strategy}")

    ###############################
    # Get Cell Simulation Id and set up CellSimulationModel
    ###############################
    # make cell simulation id
    cell_sim_id = CellSimulationId(
        seed=params.seed,
        tree_id=tree_id,
        n_cells=params.n_cells,
        strategy=params.strategy,
        fpr=params.fpr,
        fnr=params.fnr,
        na_rate=params.na_rate,
        observe_homozygous=params.observe_homozygous,
    )

    # set up the CellSimulationModel from the params and Tree Information from TreeId
    params_dict = vars(params)
    params_dict["n_mutations"] = tree_id.n_nodes - 1
    params_model = CellSimulationModel(**params_dict)

    ###############################
    # Generate Data
    ###############################
    # Create a random number generator
    rng = random.PRNGKey(params.seed)
    # Generate Data
    data = tree_inf.gen_sim_data(params_model, rng, tree)

    ###############################
    # Save Data to Disk
    ###############################
    # make filename
    filename = f"{cell_sim_id}.json"
    fullpath = os.path.join(params.out_dir, filename)
    # make output directory if it doesn't exist
    os.makedirs(params.out_dir, exist_ok=True)

    # Save the data to a JSON file
    with open(fullpath, "w") as f:
        json.dump(data, f, cls=JnpEncoder)

    # Print the path to the file if verbose
    if params.verbose:
        print(f"Saved simulation results to {fullpath}\n")


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
