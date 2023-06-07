#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make a TreeNode tree of (random, deep, star) topology.
Given number of nodes and if applicable seed.

Example Usage:
poetry run python ../scripts/make_tree.py

Example Usage with arguments:
    poetry run python scripts/plot_trees.py
    --out_dir data/plots/tree/mark00
    --seed 42
    --n_nodes 10
    --tree_type r
"""

import argparse
import jax

from pathlib import Path

import pyggdrasil.serialize as serialize


from pyggdrasil.tree_inference import TreeId, TreeType

#####################################################
# Placeholder Functions - implemented in PR # 62
# in pyggdrasil.tree_inference.tree_generator

from pyggdrasil.tree_inference import JAXRandomKey


def generate_deep_TreeNode(rng: JAXRandomKey, n_nodes: int):
    pass


def generate_star_TreeNode(n_nodes: int):
    pass


def generate_random_TreeNode(rng: JAXRandomKey, n_nodes: int):
    pass


#####################################################


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
        "--seed",
        required=False,
        help="seed used for JaxRNGKey",
        type=int,
    )

    parser.add_argument(
        "--n_nodes",
        required=True,
        help="number of nodes in the tree, i.e. no of nodes - 1 = mutations",
    )

    parser.add_argument(
        "--tree_type",
        required=True,
        help="tree type: random, deep, star, first letter only",
        type=str,
        options=[
            "r",
            "d",
            "s",
        ],
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

    # tree type
    tree_type = TreeType(args.tree_type)
    # make tree id
    tree_id = TreeId(seed=args.seed, n_nodes=args.n_nodes, tree_type=tree_type)

    # depending on the tree type, generate the tree
    if tree_type == TreeType.STAR:
        tree = generate_star_TreeNode(n_nodes=args.n_nodes)
    else:
        # make the random key
        rng = jax.random.PRNGKey(args.seed)
        if tree_type == TreeType.RANDOM:
            tree = generate_random_TreeNode(rng=rng, n_nodes=args.n_nodes)
        elif tree_type == TreeType.DEEP:
            tree = generate_deep_TreeNode(rng=rng, n_nodes=args.n_nodes)
        else:
            raise ValueError(f"Unknown tree type: {tree_type}")

    # make savename
    savename = f"{tree_id}.json"
    # save the tree
    assert tree is not None
    serialize.save_tree_node(tree, out_dir / savename)


if __name__ == "__main__":
    main()
