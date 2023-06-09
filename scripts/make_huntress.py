#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make a TreeNode tree given a mutation matrix
to generate a huntress tree.

Example Usage:
poetry run python ../scripts/make_huntress.py

Example Usage with arguments:
    poetry run python scripts/plot_trees.py
    --data_fp data/mutation_matrices/mark00
    --out_dir data/plots/tree/mark00

"""

import argparse
import json

from pathlib import Path

import pyggdrasil.serialize as serialize


from pyggdrasil.tree_inference import (
    huntress_tree_inference,
    CellSimulationId,
    MutationDataId,
)


#############################################
# Placeholder Fns implemented in PR #42

from typing import TypedDict


class CellSimulationData(TypedDict):
    pass


def get_simulation_data(data: dict):
    """Load the mutation matrix from json object of
    the simulation data output of cell_simulation.py
    Args:
        data: dict
            data dictionary containing - serialised data
    Returns:
        TypedDict of: CellSimulationData
            adjacency_matrix: interface.TreeAdjacencyMatrix
                Adjacency matrix of the tree.
            perfect_mutation_mat: PerfectMutationMatrix
                Perfect mutation matrix.
            noisy_mutation_mat: interface.MutationMatrix
                Noisy mutation matrix. May be none if cell simulation was errorless.
            root: TreeNode
                Root of the tree.
    """
    pass


#############################################


def create_parser() -> argparse.Namespace:
    """
    Parser for required input user.

    Returns:
        args: argparse.Namespace
    """

    parser = argparse.ArgumentParser(
        description="Make (random, deep, star) trees and save their TreeNode."
    )

    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory to save the TreeNode trees.",
        type=str,
    )

    parser.add_argument(
        "--data_fp",
        required=True,
        help="data_fp to load the mutation matrix",
        type=str,
    )

    parser.add_argument(
        "--n_nodes",
        required=True,
        help="number of nodes in the tree, i.e. no of nodes - 1 = mutations",
    )

    parser.add_argument(
        "--fpr",
        required=True,
        help="False positive rate of data",
    )

    parser.add_argument(
        "--fnr",
        required=True,
        help="False negative rate of data",
    )

    args = parser.parse_args()

    return args


def main() -> None:
    """
    Main function for plotting trees from MCMC samples.

    Returns:
        None
    """
    # get the arguments
    args = create_parser()

    # get data filepath
    data_fp = Path(args.data_fp)
    # load data of mutation matrix from json
    with open(data_fp, "r") as f:
        cell_simulation_data = json.load(f)
    # TODO: Once PR #42 is merged, replace this with the fn from tree_inference
    # get the simulation data
    cell_simulation_data = get_simulation_data(cell_simulation_data)
    # get the mutation matrix
    mut_mat = cell_simulation_data["noisy_mutation_mat"]

    # run huntress tree inference
    tree_tn = huntress_tree_inference(mut_mat, args.fpr, args.fnr)

    # Save the tree - make path
    out_dir = Path(args.out_dir)
    # if out_dir does not exist, create it
    out_dir.mkdir(parents=True, exist_ok=True)
    # make TreeId
    # get the number of rows in mutation matrix
    mut_mat.shape[0]

    # cell simulation id / MutationDataId
    # get cell simulation id from filename
    filename = data_fp.name
    mut_data_id = ""
    try:
        # get the cell simulation id from the filename
        mut_data_id = CellSimulationId.from_str(filename)

    except AssertionError:
        # TODO: consider adding logger
        print("Could not get CellSimulationId from filename, switching to MutationID")
        # make mutation id instead
        mut_data_id = MutationDataId(filename)

    except Exception as E:
        raise E

    # make full output path
    out_fp = out_dir / f"{mut_data_id}.json"

    # save huntress data to file
    # TODO: resolve the issue of TreeNode is not Node by Anytree ??
    serialize.save_tree_node(tree_tn, out_fp)


if __name__ == "__main__":
    main()
