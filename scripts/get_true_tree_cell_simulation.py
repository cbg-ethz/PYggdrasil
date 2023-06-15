#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reads in the output of cell_simulation.py, i.e.
a directory of cell mutation matrices, and adjacency matrices,
output a TreeNode tree of the true tree for the simulated data.

Example Usage with arguments:
   poetry run python ../scripts/get_true_tree_cell_simulation.py
    --cell_simulation_data_fp ../data/cell_simulation_data.json
    --out_fp ../data/true_tree.json

"""

import argparse
import json


from pyggdrasil.serialize import save_tree_node
from pyggdrasil.tree_inference import get_simulation_data


def create_parser() -> argparse.Namespace:
    """
    Parser for required input user.

    Returns:
        args: argparse.Namespace
    """

    parser = argparse.ArgumentParser(
        description="Extract and convert cell_simulation.py "
        "output to TreeNode true tree to read in."
    )

    parser.add_argument(
        "--cell_simulation_data_fp",
        required=False,
        help="Fullpath of simulated data:",
        default=None,
    )

    parser.add_argument(
        "--out_fp",
        required=True,
        help="Output directory to true tree",
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

    # read in the cell simulation data
    cell_simulation_data_fp = args.cell_simulation_data_fp
    # define output directory
    out_fp = args.out_fp

    # read in the cell simulation data
    with open(cell_simulation_data_fp, "r") as f:
        cell_simulation_data = json.load(f)

    cell_simulation_data = get_simulation_data(cell_simulation_data)

    # get the true tree
    true_tree = cell_simulation_data["root"]

    # save the true tree
    save_tree_node(true_tree, out_fp)


if __name__ == "__main__":
    main()
