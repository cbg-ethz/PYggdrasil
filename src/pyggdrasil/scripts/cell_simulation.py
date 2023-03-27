#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate cell mutation matrices,
by generating random trees and sampling cell attachments
with noise if error rates are non-zero.

As per definition in the SCITE Jahn et al. 2016.
"""

import argparse
import jax.random as random
import networkx as nx
import numpy as np
from jax.random import PRNGKeyArray
import json
import os


# TODO: Ask Pawel if this is the way / modify __init__.py
import pyggdrasil.tree_inference._simulate as sim


def create_parser() -> dict:
    """
    Parser for required input user.

    Returns:
        args: dict
    """

    parser = argparse.ArgumentParser(
        description="Generate test cell mutation matrices."
    )
    parser.add_argument(
        "--seed",
        required=False,
        help="Seed for random generation",
        type=int,
        default=42,
    )
    parser.add_argument("--outdir", required=True, help="Path to output directory")
    parser.add_argument("--n_trees", required=True, help="Number of trees", type=int)
    parser.add_argument("--n_cells", required=True, help="Number of cells", type=int)
    parser.add_argument(
        "--n_mutations", required=True, help="Number of mutations", type=int
    )

    parser.add_argument(
        "--strategy",
        required=False,
        choices=["UNIFORM_INCLUDE_ROOT", "UNIFORM_EXCLUDE_ROOT"],
        help="Cell Attachment Strategy",
        default="UNIFORM_INCLUDE_ROOT",
    )

    # TODO: add 2nd parser upon confirmation that muation matirx is noisy

    parser.add_argument(
        "--alpha", required=True, help="False Negative rate", type=float
    )
    parser.add_argument("--beta", required=True, help="False positive rate", type=float)

    parser.add_argument(
        "--na_rate", required=True, help="Missing entry rate", type=float
    )

    parser.add_argument(
        "--observe_homozygous",
        required=True,
        help="Observing homozygous mutations",
        type=bool,
    )

    args = parser.parse_args()
    params = vars(args)

    return params


# TODO: check with Pawel for requirements of tree topology
def generate_random_tree(rng: PRNGKeyArray, n_nodes: int) -> np.ndarray:
    """
    Generates a random tree with n nodes, where the root is the first node.
    Args:
        rng: JAX random number generator
        n_nodes: int number of nodes in the tree

    Returns:
        adj_matrix: np.ndarray
            Note: nodes are here not self-connected
    """
    # NOTE: opted to not to used networkx random tree generation
    # as it used numpy randomness which is not compatible with JAX

    # Initialize adjacency matrix with zeros
    adj_matrix = np.zeros((n_nodes, n_nodes))
    # Generate random edges for the tree
    for i in range(1, n_nodes):
        # Select a random parent node from previously added nodes
        parent = random.choice(rng, i)
        # Add an edge from the parent to the current node
        adj_matrix[parent, i] = 1
    # Return the adjacency matrix
    return adj_matrix


def reverse_node_order(adj_matrix: np.ndarray) -> np.ndarray:
    """
    Reverses the order of the nodes in the tree.
    Args:
        adj_matrix: np.ndarray

    Returns:
        adj_matrix: np.ndarray
    """
    # Reverse the order of the nodes
    adj_matrix = adj_matrix[::-1, ::-1]
    # Return the adjacency matrix
    return adj_matrix


def print_tree(adj_matrix: np.ndarray, root: int = 0):  # type: ignore
    """
    Prints a tree to the console.

    Args:
        adj_matrix: np.ndarray

    Returns:
        None
    """
    graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    print(nx.forest_str(graph, sources=[root]))


def compose_save_name(params: dict) -> str:
    """Composes save name for the results."""
    save_name = (
        f"seed_{params['seed']}_"
        f"n_trees_{params['n_trees']}_"
        f"n_cells_{params['n_cells']}_"
        f"n_mutations_{params['n_mutations']}_"
        f"alpha_{params['alpha']}_"
        f"beta_{params['beta']}_"
        f"na_rate_{params['na_rate']}_"
        f"observe_homozygous_{params['observe_homozygous']}_"
        f"strategy_{params['strategy']}"
    )
    return save_name


# from typing import List

# def build_tree_from_adjacency_matrix(
# adj_matrix: List[List[int]], root_name: str) -> TreeNode:
#     # Create a dictionary to hold nodes
#     nodes_dict = {}
#
#     # Create root node
#     root_node = TreeNode(root_name)
#     nodes_dict[root_name] = root_node
#
#     # Create nodes for each row in the adjacency matrix
#     for i in range(len(adj_matrix)):
#         node_name = i
#         node = TreeNode(node_name)
#         nodes_dict[node_name] = node
#
#     # Create edges between nodes
#     for i in range(len(adj_matrix)):
#         for j in range(len(adj_matrix[i])):
#             if adj_matrix[i][j] == 1:
#                 parent_name = i
#                 child_name = j
#                 parent = nodes_dict[parent_name]
#                 child = nodes_dict[child_name]
#                 child.parent = parent
#
#     return root_node


def run_sim(params):
    """
    Generates cell mutation matrices.

    Args:
        params: dict

    Returns:
        results: dict
            tree, perfect and noisy mutation matrix
    """
    ############################################################################
    # Parameters
    ############################################################################
    seed = params["seed"]
    outdir = params["outdir"]
    params["n_trees"]
    n_cells = params["n_cells"]
    n_mutations = params["n_mutations"]
    alpha = params["alpha"]
    beta = params["beta"]
    na_rate = params["na_rate"]
    observe_homozygous = params["observe_homozygous"]
    strategy = params["strategy"]

    ############################################################################
    # Random Seeds
    ############################################################################
    rng = random.PRNGKey(seed)
    rng_tree, rng_cell_attachment, rng_noise = random.split(rng, 3)

    ##############################################################################
    # Generate Trees
    ##############################################################################
    # used network X to generate random trees and convert to adjacency matrix
    tree = generate_random_tree(rng_tree, n_nodes=n_mutations)
    # reverse node order
    tree = reverse_node_order(tree)
    print(tree)
    # if n_trees <=5:
    print_tree(tree, root=n_mutations - 1)

    ##############################################################################
    # Attach Cells To Tree
    ###############################################################################
    # convert adjacency matrix to self-connected tree - in tree_inference format
    np.fill_diagonal(tree, 1)
    # define strategy
    strategy = sim.CellAttachmentStrategy[strategy]
    # attach cells to tree
    perfect_mutation_mat = sim.attach_cells_to_tree(
        rng_cell_attachment, tree, n_cells, strategy
    )

    ###############################################################################
    # Add Noise
    ################################################################################
    noisy_mutation_mat = None
    if (beta > 0) or (alpha > 0) or (na_rate > 0):
        noisy_mutation_mat = sim.add_noise_to_perfect_matrix(
            rng_noise, perfect_mutation_mat, alpha, beta, na_rate, observe_homozygous
        )

    ################################################################################
    # Save Simulation Results
    ################################################################################
    # make save name from parameters

    filename = compose_save_name(params) + ".json"
    fullpath = os.path.join(outdir, filename)
    os.makedirs(outdir, exist_ok=True)

    # Save Tree

    # root_node = build_tree_from_adjacency_matrix(tree, root_node_name)

    # print(root_node)

    # Save Mutation Matrix
    # Create a dictionary to hold matrices
    if noisy_mutation_mat is not None:
        data = {
            "adjaency_matrix": tree.tolist(),
            "perfect_mutation_mat": perfect_mutation_mat.tolist(),
            "noisy_mutation_mat": noisy_mutation_mat.tolist(),
            "tree": tree.tolist(),
        }
    else:
        data = {
            "adjaency_matrix": tree.tolist(),
            "perfect_mutation_mat": perfect_mutation_mat,
        }

    # Save the data to a JSON file
    with open(fullpath, "w") as f:
        json.dump(data, f)

    raise NotImplementedError


def main() -> None:
    """
    Main function.
    """
    params = create_parser()
    run_sim(params)

    # with open(file_name, "w") as file_handler:


#########################################################################################
# MAIN
########################################################################################
if __name__ == "__main__":
    main()
