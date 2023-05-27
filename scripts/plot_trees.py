#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots all trees from an MCMC run.

Example Usage:
poetry run python ../scripts/plot_trees.py

Example Usage with arguments:
    poetry run python scripts/run_mcmc.py
    --config_fp data/config/mcmc_config_mark00.json
    --out_dir data/mcmc/mark01
    --data_fp data/mock/seed_32_n_..._tree_3.json

    if you want to provide a tree:
    from a mcmc run:
    --init_tree_fp data/mcmc/mark01/samples_XXXXXXXX_XXXXXX.json
    --iteration 1000
    or to read in a TreeNode
    --init_tree_fp data/trees/tree_3.json
    --init_TreeNode


"""

import argparse
import logging


from pathlib import Path


import pyggdrasil.serialize as serialize
import pyggdrasil.tree_inference as tree_inf
import pyggdrasil.visualize as visualize


def create_parser() -> argparse.Namespace:
    """
    Parser for required input user.

    Returns:
        args: argparse.Namespace
    """

    parser = argparse.ArgumentParser(description="Plot trees from MCMC sample.")

    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory to save the tree plots.",
        type=str,
    )

    parser.add_argument(
        "--data_fp",
        required=True,
        help="File path containing the mcmc samples.",
        type=str,
    )

    args = parser.parse_args()

    return args


def main() -> None:
    """
    Main function for plotting trees from MCMC samples.

    Returns:
        None
    """

    args = create_parser()

    # get the full path
    out_dir = Path(args.out_dir)
    # if out_dir does not exist, create it
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    fullpath_log = out_dir / "plot_trees.log"
    # if logfile does not exist, create it
    if not Path(fullpath_log).exists():
        Path(fullpath_log).touch()
    logging.basicConfig(filename=fullpath_log, level=logging.DEBUG)
    # set logging level for jax
    logging.getLogger("jax._src.dispatch").setLevel(logging.ERROR)
    logging.getLogger("jax._src.interpreters.pxla").setLevel(logging.ERROR)
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

    logging.info("Starting Session")

    # get the full path
    fullpath_d = Path(args.data_fp)
    # plot the trees
    mcmc_samples = serialize.read_mcmc_samples(fullpath=fullpath_d)
    print(mcmc_samples)
    pure_data = tree_inf.to_pure_mcmc_data(mcmc_samples)

    # get iterations
    iterations = pure_data.iterations
    print(iterations)

    # print options
    print_options = dict()
    print_options["title"] = True
    print_options["data_tree"] = dict()
    print_options["data_tree"]["log-likelihood"] = True
    print_options["data_tree"]["Data type"] = False
    print_options["data_tree"]["Run"] = False

    # for each iteration, plot the tree
    for i in iterations:
        # get the sample
        sample = pure_data.get_sample(i)
        print(sample)
        # get the tree
        tree = sample[1]
        tree.print_topo()
        tree.data = dict()
        tree.data["log-likelihood"] = sample[2]
        # get the log probability
        log_prob = sample[2]
        # make savename from iteration number

        save_name = "iteration_" + str(i) + "_log_prob_" + str(log_prob) + ".svg"

        # make full path with pathlib
        save_dir = Path(args.out_dir).joinpath(save_name)

        # plot tree
        visualize.plot(tree, save_name, save_dir, print_options)

    logging.info("End Session")


if __name__ == "__main__":
    main()