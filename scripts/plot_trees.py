#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots all trees from an MCMC run.

Example Usage:
poetry run python ../scripts/plot_trees.py

Example Usage with arguments:
    poetry run python scripts/plot_trees.py
    --out_dir data/plots/tree/mark00
    --data_fp data/mcmc/mark00/samples.json

"""

import argparse
import logging


from pathlib import Path
from tqdm import tqdm

import pyggdrasil.serialize as serialize
import pyggdrasil.analyze as analyze
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

    parser.add_argument(
        "--progress_bar_off",
        required=False,
        help="Turn off the progress bar",
        action="store_true",
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
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    fullpath_log = out_dir / "plot_trees.log"
    # if logfile does not exist, create it
    fullpath_log.unlink(missing_ok=True)
    fullpath_log.touch()

    logging.basicConfig(filename=fullpath_log, level=logging.INFO)
    # set logging level for jax
    logging.getLogger("jax._src.dispatch").setLevel(logging.ERROR)
    logging.getLogger("jax._src.interpreters.pxla").setLevel(logging.ERROR)
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
    logging.getLogger("matplotlib.texmanager").setLevel(logging.ERROR)

    logging.info("Starting Session")

    # get the full path
    fullpath_d = Path(args.data_fp)
    # plot the trees
    mcmc_samples = serialize.read_mcmc_samples(fullpath=fullpath_d)
    pure_data = analyze.to_pure_mcmc_data(mcmc_samples)

    # get iterations
    iterations = pure_data.iterations
    # convert iterations to list of integers
    iterations = list(map(int, iterations))

    # for each iteration, plot the tree
    for i in tqdm(iterations, disable=args.progress_bar_off):
        # get the sample
        sample = pure_data.get_sample(i)

        save_dir = Path(args.out_dir)

        visualize.plot_tree_mcmc_sample(sample, save_dir)
        # log
        logging.info("Plotted tree for iteration %s", i)
    # log end
    logging.info("End Session")


if __name__ == "__main__":
    main()
