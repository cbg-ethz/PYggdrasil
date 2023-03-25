#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate cell mutation matrices,
by generating random trees and sampling cell attachments
with noise if error rates are non-zero.

As per definition in the SCITE Jahn et al. 2016.
"""

import argparse


def run_sim(params):
    """
    Generates cell mutation matrices.

    Args:
        params: dict

    Returns:
        results: dict
            tree, perfect and noisy mutation matrix
    """

    ########################################################################################
    # Generate Trees
    ########################################################################################
    # used network X to generate random trees
    # https://networkx.org/documentation/stable/reference/generated/networkx.generators.trees.random_tree.html

    ########################################################################################
    # Attach Cells To Tree
    ########################################################################################

    ########################################################################################
    # Add Noise
    ########################################################################################

    #########################################################################################
    # Save Simulation Results
    #########################################################################################

    # Save Tree
    # convert to serialized

    # Save Mutation Matrix
    # Perfect
    # Noisy

    raise NotImplementedError


def create_parser() -> dict:
    """
    Parser for required input user.

    Returns:
        args: dict
    """

    parser = argparse.ArgumentParser(
        description="Generate test cell mutation matrices."
    )
    parser.add_argument("--outdir", required=True, help="Path to output directory")
    parser.add_argument("--n_trees", required=True, help="Number of trees", type=int)
    parser.add_argument("--n_cells", required=True, help="Number of cells", type=int)
    parser.add_argument(
        "--n_mutations", required=True, help="Number of mutations", type=int
    )
    parser.add_argument(
        "--alpha", required=True, help="False Negative rate", type=float
    )
    parser.add_argument("--beta", required=True, help="False positive rate", type=float)

    args = parser.parse_args()
    params = vars(args)

    return params


def main() -> None:
    """
    Main function.
    """
    create_parser()

    # with open(file_name, "w") as file_handler:


#########################################################################################
# MAIN
########################################################################################
if __name__ == "__main__":
    main()

    # result = run_sim(args)

    # print("simulation successfully finished for " + result["save_name"])
