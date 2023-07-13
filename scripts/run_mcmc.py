#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the MCMC sampler for tree inference
Using the given mutation data, error rates and initial tree.

As per definition in the SCITE Jahn et al. 2016.

Allows to start an MCMC chain from a given tree provided as
a dumped tree json file as a TreeNode.

Example Usage:
poetry run python ../scripts/run_mcmc.py

Example Usage with arguments - start from provided TreeNode:
    poetry run python scripts/run_mcmc.py
    --seed 42
    --config <<JSON tree_inference.McmcConfig>>
    --out_dir data/mark00/mcmc/
    --data_fp data/mark00/mutations/XXXXX.json
    --init_tree_fp data/mark00/trees/XXXXX.json
    --init_TreeNode

JSON tree_inference.McmcConfig
{
    "move_probs": {
        "prune_and_reattach": 0.1,
        "swap_node_labels": 0.65,
        "swap_subtrees": 0.25
    },
    "fnr": 0.1,
    "fpr": 0.2,
    "num_samples": 1000,
    "burn_in": 20,
    "thinning": 100
}

"""

import argparse
import jax.random as random
import json
import logging
import pytz

from pathlib import Path
from datetime import datetime, timedelta

import pyggdrasil.tree_inference as tree_inf
import pyggdrasil.serialize as serialize

from pyggdrasil.tree_inference import (
    mcmc_sampler,
    MoveProbabilities,
    McmcConfig,
    TreeId,
    McmcRunId,
    CellSimulationId,
    MutationDataId,
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

    parser.add_argument(
        "--config_fp", required=False, help="Config file path", type=str
    )

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


def run_chain(
    params: argparse.Namespace, config: McmcConfig, mc_run_id: McmcRunId
) -> None:
    """Run the MCMC sampler for tree inference.

    Args:
        params: argparse.Namespace
            input parameters from parser for MCMC sampler
        config: dict
            config dictionary
        mc_run_id: McmcRunId
            id of the MCMC run, used for saving the samples

    Returns:
        None
    """

    # Set random seed
    rng = random.PRNGKey(params.seed)

    # load data of mutation matrix
    with open(params.data_fp, "r") as f:
        cell_simulation_data = json.load(f)
    cell_simulation_data = tree_inf.get_simulation_data(cell_simulation_data)
    # get the mutation matrix
    mut_mat = cell_simulation_data["noisy_mutation_mat"]

    # parse tree from file given as input
    # make path and check if Path exists
    p = Path(params.init_tree_fp)
    if not p.exists():
        raise FileNotFoundError(f"File {params.init_tree_fp} does not exist.")

    init_tree_node = serialize.read_tree_node(params.init_tree_fp)
    # convert TreeNode to Tree
    init_tree = tree_inf.tree_from_tree_node(init_tree_node)
    logging.info("Loaded tree (TreeNode) from file.")

    # assert that the number of mutations and the data matrix size match
    # no of nodes must equal the number of rows in the data matrix plus root truncated
    if not init_tree.labels.shape[0] == mut_mat.shape[0] + 1:
        raise AssertionError(
            "Number of mutations and data matrix size do not match.\n"
            f"tree {init_tree.labels.shape[0]} != data {mut_mat.shape[0]}"
        )
        # TODO (Gordon): if certain about this add check also in mcmc_sampler

    # Make Move Probabilities
    prune_and_reattach = config.move_probs.prune_and_reattach
    swap_node_labels = config.move_probs.swap_node_labels
    swap_subtrees = config.move_probs.swap_subtrees
    move_probs = MoveProbabilities(prune_and_reattach, swap_node_labels, swap_subtrees)

    # run mcmc sampler
    mcmc_sampler(
        rng_key=rng,
        data=mut_mat,
        error_rates=(config.fpr, config.fnr),
        move_probs=move_probs,
        num_samples=config.n_samples,
        num_burn_in=config.burn_in,
        out_fp=Path(params.out_dir) / f"{mc_run_id}.json",
        thinning=config.thinning,
        init_tree=init_tree,
    )


def get_config(config_fp: str) -> McmcConfig:
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

    config_td = McmcConfig(**config)

    return config_td


def get_str_timedelta(td: timedelta) -> str:
    """Convert timedelta to string."""
    # get days
    days = td.days
    # get hours
    hours = td.seconds // 3600
    # get minutes
    minutes = (td.seconds // 60) % 60
    # get seconds
    seconds = td.seconds % 60
    # get milliseconds
    milliseconds = td.microseconds // 1000
    # format string
    # only add time item if it is >0
    str_td = ""
    if days > 0:
        str_td += f"{days}d "
    if hours > 0:
        str_td += f"{hours}h "
    if minutes > 0:
        str_td += f"{minutes}m "
    if seconds > 0:
        str_td += f"{seconds}s "
    if milliseconds > 0:
        str_td += f"{milliseconds}ms "
    # remove trailing whitespace
    str_td = str_td.strip()

    return str_td


def mcmc_run_id_from_params(params: argparse.Namespace) -> McmcRunId:
    """Make MCMC run id from command line parameters."""
    # make MCMC run id
    # get tree id from filepath excluding json extension
    tree_path = Path(params.init_tree_fp)
    tree_id = tree_path.stem
    init_tree_id = TreeId.from_str(tree_id)
    # ge the config id
    # read json from file
    with open(params.config_fp, "r") as f:
        json_config = json.load(f)
    config_id = McmcConfig(**json_config)
    # get data id
    # get data id from filepath excluding json extension
    data_path = Path(params.data_fp)
    data_id = data_path.stem
    try:
        data_id = CellSimulationId.from_str(data_id)
    except AssertionError:
        data_id = MutationDataId(data_id)
    # make MC run id
    mc_run_id = McmcRunId(params.seed, data_id, init_tree_id, config_id)

    return mc_run_id


def main() -> None:
    """
    Main function.
    """
    # Parse command line arguments
    params = create_parser()

    # load config file
    config = get_config(params.config_fp)

    # make MC run id
    mc_run_id = mcmc_run_id_from_params(params)

    out_dir = Path(params.out_dir)
    # fullpath = out_dir / f"mcmc_run_{timestamp}.log"
    fullpath = out_dir / f"{mc_run_id}.log"

    # if out_dir does not exist, create it
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # if logfile does not exist, create it
    if not Path(fullpath).exists():
        Path(fullpath).touch()
    # Set up logging
    logging.basicConfig(filename=fullpath, level=logging.INFO)
    # set logging level for jax
    logging.getLogger("jax._src.dispatch").setLevel(logging.ERROR)
    logging.getLogger("jax._src.interpreters.pxla").setLevel(logging.ERROR)
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

    logging.info("Starting Session")

    logging.info(f"Configuration:/ {config}")

    logging.info(f"Using config file: {params.config_fp}")
    logging.info(f"Using data file: {params.data_fp}")
    logging.info(f"Using output directory: {params.out_dir}")
    logging.info(f"Using data file: {params.data_fp}")

    # get date and time for output file
    local_timezone = pytz.timezone(pytz.country_timezones["CH"][0])
    datetime_start = datetime.now(local_timezone)
    timestamp_start = datetime_start.strftime("%Y %m %d - %H:%M:%S %Z%z")
    logging.info(f"Started run at datetime: {timestamp_start}")

    # Run the simulation and save to disk
    run_chain(params, config, mc_run_id)

    # get date and time for output file
    datetime_end = datetime.now(local_timezone)
    timestamp_end = datetime_end.strftime("%Y %m %d - %H:%M:%S %Z%z")
    logging.info(f"Finished run at datetime: {timestamp_end}")
    # get runtime of each sample
    runtime = datetime_end - datetime_start
    str_dt = get_str_timedelta(runtime)
    logging.info("Runtime: " + str_dt)
    # runtime per sample
    runtime_per_sample = runtime / (config.n_samples)
    runtime_per_sample_str = get_str_timedelta(runtime_per_sample)
    logging.info("Runtime per sample: " + runtime_per_sample_str)
    logging.info("Finished Session")


#########################################################################################
# MAIN
########################################################################################
if __name__ == "__main__":
    main()
