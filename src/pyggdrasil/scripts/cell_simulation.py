#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate cell mutation matrices,
by generating random trees and sampling cell attachments
with noise if error rates are non-zero.

As per definition in the SCITE Jahn et al. 2016.
"""

import os
import sys
import argparse


def run_sim(args):
    """
    Generates cell mutation matrices.

    Args:
        args:

    Returns:
        results: dict
            tree, perfect and noisy mutation matrix
    """
    int(args.n_trees)
    int(args.n_cells)
    int(args.n_mutations)

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


#########################################################################################
# MAIN
########################################################################################
if __name__ == "__main__":
    # ========================================================================
    # Set up the parsing of command-line arguments
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
    outdir = args.outdir.strip("/")

    os.makedirs(args.outdir, exist_ok=True)

    # Create the output file
    file_name = "{}/output.txt".format(args.outdir)
    try:
        f_out = open(file_name, "w")
    except IOError:
        print("Output file {} cannot be created".format(file_name))
        sys.exit(1)

    # result = run_sim(args)

    # print("simulation successfully finished for " + result["save_name"])
