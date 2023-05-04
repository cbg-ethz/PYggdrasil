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


# TODO: consider adding proper logging to show progress
# https://www.codingem.com/log-file-in-python/


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
        "--init_tree",
        required=False,
        help="Initial tree to start the MCMC sampler from.",
        default=None,
    )

    parser.add_argument(
        "--alpha", required=True, help="False negative rate", type=float
    )
    parser.add_argument("--beta", required=True, help="False positive rate", type=float)

    # TODO switch to fpr and fnr

    parser.add_argument(
        "--move_prob",
        required=False,
        help="Probability of a move step.",
        type=float,
        default=0.1,
    )
    args = parser.parse_args()

    return args


def run_chain(params: argparse.Namespace) -> None:
    """Run the MCMC sampler for tree inference.

    Args:
        params: argparse.Namespace
            input parameters from parser for MCMC sampler

    Returns:
        None
    """

    print("Done!")


def main() -> None:
    """
    Main function.
    """
    # Parse command line arguments
    params = create_parser()
    # Run the simulation and save to disk
    run_chain(params)


#########################################################################################
# MAIN
########################################################################################
if __name__ == "__main__":
    main()
