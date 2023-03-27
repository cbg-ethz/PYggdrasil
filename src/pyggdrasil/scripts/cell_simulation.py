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

    print(perfect_mutation_mat)

    ###############################################################################
    # Add Noise
    ################################################################################
    if (beta > 0) or (alpha > 0) or (na_rate > 0):
        noisy_mutation_mat = sim.add_noise_to_perfect_matrix(
            rng_noise, perfect_mutation_mat, alpha, beta, na_rate, observe_homozygous
        )
        print(noisy_mutation_mat)

    ################################################################################
    # Save Simulation Results
    ################################################################################

    # Save Tree
    # convert to serialized

    # Save Mutation Matrix
    # Perfect
    # Noisy

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
