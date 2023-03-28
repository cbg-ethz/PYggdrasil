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

from pyggdrasil.tree import TreeNode
from pyggdrasil.serialize._to_json import serialize_tree_to_dict
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
        help="Cell attachment strategy",
        default="UNIFORM_INCLUDE_ROOT",
    )

    parser.add_argument(
        "--alpha", required=True, help="False negative rate", type=float
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

    parser.add_argument(
        "--verbose",
        required=False,
        default=False,
        help="Print trees and full save path",
        type=bool,
    )

    args = parser.parse_args()
    params = vars(args)

    return params


def generate_random_tree(rng: PRNGKeyArray, n_nodes: int) -> np.ndarray:
    """
    Generates a random tree with n nodes, where the root is the first node.
    Args:
        rng: JAX random number generator
        n_nodes: int number of nodes in the tree

    Returns:
        adj_matrix: np.ndarray
            adjacency matrix
            Note 1: nodes are here not self-connected
            Note 2: the root is the first node
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
    Reverses the order of the nodes in the tree adjacency matrix.
    Args:
        adj_matrix: np.ndarray
            adjacency matrix

    Returns:
        adj_matrix: np.ndarray
            adjacency matrix
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
            adjacency matrix
            with no self-loops (i.e. diagonal is all zeros)

    Returns:
        None
    """
    graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    print(nx.forest_str(graph, sources=[root]))


def compose_save_name(params: dict, *, tree_no: int) -> str:
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
    if tree_no is not None:
        save_name += f"_tree_{tree_no}"
    return save_name


def adjacency_to_root_dfs(adj_matrix: np.ndarray) -> TreeNode:
    """Convert adjacency matrix to tree in tree.TreeNode
        traverses a tree using depth first search.

    Args:
        adj_matrix: np.ndarray
            with no self-loops (i.e. diagonal is all zeros)
            and the root as the highest index node
    Returns:
        root: TreeNode containing the entire tree
    """

    # Determine the root node (node with highest index)
    root_idx = len(adj_matrix) - 1

    # Create a stack to keep track of nodes to visit
    stack = [root_idx]

    # Create a set to keep track of visited nodes
    visited = set()

    # Create a list to keep track of nodes
    child_parent = {}
    child_parent[root_idx] = None

    # Create a list to keep track of TreeNodes
    list_TreeNode = [] * len(adj_matrix)

    # Traverse the tree using DFS
    while stack:
        # Get the next node to visit
        node = stack.pop()

        # Skip if already visited
        if node in visited:
            # print(f"Already Visited node {node}")
            continue

        # Visit the node
        # print(f"Visiting node {node}")

        # Recall parent
        parent = child_parent[node]
        # print(f"Parent of node {node} is {child_parent[node]}")

        if node == root_idx:
            root = TreeNode(name=node, data=None, parent=None)
            list_TreeNode[node] = root
        else:
            child = TreeNode(name=node, data=None, parent=list_TreeNode[parent])
            list_TreeNode[node] = child

        # Add to visited set
        visited.add(node)

        # Add children to the stack
        # (in reverse order to preserve order in adjacency matrix)
        for child in reversed(range(len(adj_matrix))):
            if adj_matrix[node][child] == 1 and child not in visited:
                stack.append(child)
                # print(f"Adding node {child} to stack")
                # Commit Parent to Memory
                child_parent[child] = node

    root = list_TreeNode[root_idx]

    return root


def dummy_serialize(root: TreeNode):
    """Dummy function to serialize the data a TreeNode."""
    pass


def gen_sim_data(
    params: dict,
    rng: PRNGKeyArray,
    *,
    tree_no: int,
) -> None:
    """
    Generates cell mutation matrix for one tree and writes to file.

    Args:
        params: dict
            input parameters from parser for simulation
        rng: JAX random number generator
        tree_no: int - optional
            tree number if a series is generated
    Returns:
        None
    """
    ############################################################################
    # Parameters
    ############################################################################
    outdir = params["outdir"]
    n_cells = params["n_cells"]
    n_mutations = params["n_mutations"]
    alpha = params["alpha"]
    beta = params["beta"]
    na_rate = params["na_rate"]
    observe_homozygous = params["observe_homozygous"]
    strategy = params["strategy"]
    verbose = params["verbose"]

    ############################################################################
    # Random Seeds
    ############################################################################
    rng_tree, rng_cell_attachment, rng_noise = random.split(rng, 3)

    ##############################################################################
    # Generate Trees
    ##############################################################################
    #  generate random trees (uniform sampling) as adjacency matrix
    tree_inv = generate_random_tree(rng_tree, n_nodes=n_mutations)
    # reverse node order
    tree = reverse_node_order(tree_inv)
    # print tree if prompted
    if verbose:
        print_tree(tree, root=n_mutations - 1)

    ##############################################################################
    # Attach Cells To Tree
    ###############################################################################
    # convert adjacency matrix to self-connected tree - in tree_inference format
    np.fill_diagonal(tree, 1)
    # define strategy
    strategy = sim.CellAttachmentStrategy[strategy]
    # attach cells to tree - generate perfect mutation matrix
    perfect_mutation_mat = sim.attach_cells_to_tree(
        rng_cell_attachment, tree, n_cells, strategy
    )

    ###############################################################################
    # Add Noise
    ################################################################################
    # add noise to perfect mutation matrix
    noisy_mutation_mat = None
    if (beta > 0) or (alpha > 0) or (na_rate > 0):
        noisy_mutation_mat = sim.add_noise_to_perfect_matrix(
            rng_noise, perfect_mutation_mat, alpha, beta, na_rate, observe_homozygous
        )

    ################################################################################
    # Save Simulation Results
    ################################################################################
    # make save name and path from parameters
    filename = compose_save_name(params, tree_no=tree_no) + ".json"
    fullpath = os.path.join(outdir, filename)
    # make output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    # format tree for saving
    root = adjacency_to_root_dfs(tree)
    root_serialized = serialize_tree_to_dict(root, serialize_data=dummy_serialize)

    # Save the data to a JSON file
    # Create a dictionary to hold matrices
    if noisy_mutation_mat is not None:
        data = {
            "adjacency_matrix": tree.tolist(),
            "perfect_mutation_mat": perfect_mutation_mat.tolist(),
            "noisy_mutation_mat": noisy_mutation_mat.tolist(),
            "tree": tree.tolist(),
            "root": root_serialized,
        }
    else:
        data = {
            "adjacency_matrix": tree.tolist(),
            "perfect_mutation_mat": perfect_mutation_mat,
            "tree": tree.tolist(),
            "root": root_serialized,
        }

    # Save the data to a JSON file
    with open(fullpath, "w") as f:
        json.dump(data, f)

    # Print the path to the file if verbose
    if verbose:
        print(f"Saved simulation results to {fullpath}\n")


def run_sim(params) -> None:
    """Generate {n_trees} of simulated data and save to disk.

    Args:
        params: dict
            input parameters from parser for simulation

    Returns:
        None
    """

    # Create a random number generator
    rng = random.PRNGKey(params["seed"])

    # Generate n_simulations of simulated data
    for i in range(params["n_trees"]):
        print(f"Generating simulation {i+1}/{params['n_trees']}")
        gen_sim_data(params, rng, tree_no=i + 1)
        # Generate new random number generator
        rng, _ = random.split(rng)

    # Print success message
    print(f"{ params['n_trees'] } trees generated successfully!")
    print("Done!")


def main() -> None:
    """
    Main function.
    """
    # Parse command line arguments
    params = create_parser()
    # Run the simulation and save to disk
    run_sim(params)


#########################################################################################
# MAIN
########################################################################################
if __name__ == "__main__":
    main()
