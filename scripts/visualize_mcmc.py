#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize MCMC samples from a single run,
given a directory of samples and a true tree.

Allows for individual and panel plots.

Example Usage:
poetry run python  XXXXXXXXX

Example Usage with arguments:
   XXXXXXXXXXXXXXXXXXXXXXX

"""

import argparse
from pathlib import Path

import pyggdrasil.serialize as serialize
import pyggdrasil.tree_inference as ti
import pyggdrasil.visualize as viz
import pyggdrasil.distances as dist


def create_parser() -> argparse.Namespace:
    """
    Parser for required input user.

    Returns:
        args: argparse.Namespace
    """

    parser = argparse.ArgumentParser(description="Visualize MCMC samples.")

    parser.add_argument(
        "--true_tree_fp",
        required=False,
        help="Fullpath of true tree if known, else None.",
        default=None,
    )

    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory to save the plots.",
        type=str,
    )

    parser.add_argument(
        "--mcmc_samples_fp",
        required=True,
        help="File path containing the mcmc samples.",
        type=str,
    )

    args = parser.parse_args()

    return args


def main() -> None:
    """
    Main function for the script.

    Returns:
        None
    """

    # get arguments
    args = create_parser()

    # load data
    # load mcmc samples
    fullpath_d = Path(args.mcmc_samples_fp)
    mcmc_samples = serialize.read_mcmc_samples(fullpath=fullpath_d)
    pure_data = ti.to_pure_mcmc_data(mcmc_samples)
    # load true tree
    if args.true_tree_fp is not None:
        fullpath_tt = Path(args.true_tree_fp)
        true_tree = serialize.read_tree_node(fullpath=fullpath_tt)

        # define distance function
        dist_func = dist.MP3Similarity()

        out_dir = Path(args.out_dir)
        viz.make_mcmc_run_panel(
            pure_data,
            similarity_measure=dist_func,
            true_tree=true_tree,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()
