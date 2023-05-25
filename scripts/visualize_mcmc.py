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
        help="Output directory to save the MCMC samples.",
        type=str,
    )

    parser.add_argument(
        "--data_fp", required=True, help="File path containing the data.", type=str
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
    create_parser()


if __name__ == "__main__":
    main()
