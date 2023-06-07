"""Simulations of mutation matrices from mutation trees."""
from enum import Enum

import numpy as np
import jax.numpy as jnp
import logging
from typing import Union

from jax import random, Array
from pydantic import BaseModel, validator

import pyggdrasil.serialize as serialize

from pyggdrasil.tree_inference import (
    _generate_random_tree_adj_mat,
    JAXRandomKey,
    MutationMatrix,
    TreeAdjacencyMatrix,
    AncestorMatrix,
    CellAttachmentVector,
)

from pyggdrasil.tree import TreeNode


# Mutation matrix without noise
# Represents mutations in sampled cells (n_sites, n_cells)
# with n_cell columns, and n_site rows
# (i.e. each row is a site, no root), and its cells attached
PerfectMutationMatrix = Union[np.ndarray, Array]

# adjacency matrix of tree
adjacency_matrix = Union[np.ndarray, Array]


def _add_false_positives(
    rng: JAXRandomKey,
    matrix: PerfectMutationMatrix,
    noisy_mat: MutationMatrix,
    false_positive_rate: float,
) -> MutationMatrix:
    """adds false positives to  mutation matrix

    Args:
        rng: JAX random key
        matrix: perfect mutation matrix
        noisy_mat: matrix to modify, accumulated changes
        false_positive_rate: false positive rate :math:`\\fpr`

    Returns:
        Mutation matrix of size and entries as noisy_mat given
         with false positives at rate given
    """

    # P(D_{ij} = 1 |E_{ij}=0)=fpr
    # Generate a random matrix of the same shape as the original
    rand_matrix = random.uniform(key=rng, shape=matrix.shape)
    # Create a mask of elements that satisfy the condition
    # (original value equals y and random value is less than p)
    mask = (matrix == 0) & (rand_matrix < false_positive_rate)
    # Use the mask to set the corresponding elements of the matrix to x
    noisy_mat = jnp.where(mask, 1, noisy_mat)

    return noisy_mat


def _add_false_negatives(
    rng: JAXRandomKey,
    matrix: PerfectMutationMatrix,
    noisy_mat: MutationMatrix,
    false_negative_rate: float,
    observe_homozygous: bool,
) -> MutationMatrix:
    """adds false negatives to mutation matrix

    Args:
        rng: JAX random key
        matrix: perfect mutation matrix
        noisy_mat: matrix to modify, accumulated changes
        false_negative_rate: false positive rate :math:`\\fpr`

    Returns:
        Mutation matrix of size and entries as noisy_mat given
        with false negatives at rate given
    """

    # P(D_{ij}=0|E_{ij}=1) = fnr if non-homozygous
    # P(D_{ij}=0|E_{ij}=1) = fnr / 2 if homozygous
    rand_matrix = random.uniform(key=rng, shape=matrix.shape)
    mask = matrix == 1
    mask_homozygous = observe_homozygous & (rand_matrix < false_negative_rate / 2)
    mask_non_homozygous = (not observe_homozygous) & (rand_matrix < false_negative_rate)
    mask = mask & np.logical_or(mask_homozygous, mask_non_homozygous)
    noisy_mat = jnp.where(mask, 0, noisy_mat)

    return noisy_mat


def _add_homozygous_errors(
    rng_neg: JAXRandomKey,
    rng_pos: JAXRandomKey,
    matrix: PerfectMutationMatrix,
    noisy_mat: MutationMatrix,
    false_negative_rate: float,
    false_positive_rate: float,
    observe_homozygous: bool,
) -> MutationMatrix:
    """Adds both homozygous errors to mutation matrix, if observe_homozygous.

    Args:
        rng_neg: Jax random key for given E=0
        rng_pos: Jax random key for given E=1
        matrix: perfect mutation matrix
        noisy_mat: matrix to modify, accumulated changes
        false_negative_rate: false negative rate :math:`\\fnr`
        false_positive_rate: false positive rate :math:`\\fpr`
        observe_homozygous: is homozygous or not

    Returns:
        Mutation matrix of size and entries as noisy_mat given
        with false homozygous calls at rates given.
    """

    # Add Homozygous False Un-mutated
    # # P(D_{ij} = 2 | E_{ij} = 0) = fpr*fnr / 2
    rand_matrix = random.uniform(key=rng_neg, shape=matrix.shape)
    mask = (
        (matrix == 0)
        & observe_homozygous
        & (rand_matrix < (false_negative_rate * false_positive_rate / 2))
    )
    noisy_mat = jnp.where(mask, 2, noisy_mat)

    # Add Homozygous False Mutated
    # P(D_{ij} = 2| E_{ij} = 1) = fnr / 2
    rand_matrix = random.uniform(key=rng_pos, shape=matrix.shape)
    mask = (
        (matrix == 1) & observe_homozygous & (rand_matrix < (false_negative_rate / 2))
    )
    noisy_mat = jnp.where(mask, 2, noisy_mat)

    return noisy_mat


def _add_missing_entries(
    rng: JAXRandomKey,
    matrix: PerfectMutationMatrix,
    noisy_mat: MutationMatrix,
    missing_entry_rate: float,
) -> MutationMatrix:
    """Adds missing entries

    Args:
        rng: Jax random key
        matrix: perfect mutation matrix
        noisy_mat: matrix to modify, accumulated changes

    Returns:
        Mutation matrix of size and entries as noisy_mat given
        with missing entries e=3 at rate given.
    """

    # Add missing data
    # P(D_{ij} = 3) = missing_entry_rate
    rand_matrix = random.uniform(key=rng, shape=matrix.shape)
    mask = rand_matrix < missing_entry_rate
    noisy_mat = jnp.where(mask, 3, noisy_mat)

    return noisy_mat


def add_noise_to_perfect_matrix(
    rng: JAXRandomKey,
    matrix: PerfectMutationMatrix,
    false_positive_rate: float = 1e-5,
    false_negative_rate: float = 1e-2,
    missing_entry_rate: float = 1e-2,
    observe_homozygous: bool = False,
) -> MutationMatrix:
    """

    Args:
        rng: JAX random key
        matrix: binary matrix with 1 at sites where a mutation is present.
          Shape (n_cells, n_sites).
        false_positive_rate: false positive rate :math:`\\fpr`.
          Should be in the half-open interval [0, 1).
        false_negative_rate: false negative rate :math:`\\fnr`.
          Should be in the half-open interval [0, 1).
        missing_entry_rate: fraction os missing entries.
          Should be in the half-open interval [0, 1).
          If 0, no missing entries are present.
        observe_homozygous: if true, some homozygous mutations will be observed
          due to noise. See Eq. (8) on p. 5 of the original SCITE paper.

    Returns:
        array with shape (n_cells, n_sites)
          with the observed mutations and ``int`` data type.
          Entries will be:
            - 0 (no mutation)
            - 1 (mutation present)
            - ``HOMOZYGOUS_MUTATION`` if ``observe_homozygous`` is true
            - ``MISSING ENTRY`` if ``missing_entry_rate`` is non-zero
    """
    # RNGs for false positives, false negatives, and missing data
    rng_false_pos, rng_false_neg, rng_miss, rng_homo_pos, rng_homo_neg = random.split(
        rng, 5
    )
    # make matrix to edit and keep unchanged
    noisy_mat = matrix.copy()

    # Add False Positives - P(D_{ij} = 1 |E_{ij}=0)=fpr
    noisy_mat = _add_false_positives(
        rng_false_pos, matrix, noisy_mat, false_positive_rate
    )

    # Add False Negatives
    # P(D_{ij}=0|E_{ij}=1) = fnr if non-homozygous
    # P(D_{ij}=0|E_{ij}=1) = fnr / 2 if homozygous
    noisy_mat = _add_false_negatives(
        rng_false_neg, matrix, noisy_mat, false_negative_rate, observe_homozygous
    )

    # Add Homozygous Errors if applicable
    noisy_mat = _add_homozygous_errors(
        rng_homo_neg,
        rng_homo_pos,
        matrix,
        noisy_mat,
        false_negative_rate,
        false_positive_rate,
        observe_homozygous,
    )

    # Add missing entries
    noisy_mat = _add_missing_entries(rng_miss, matrix, noisy_mat, missing_entry_rate)

    return noisy_mat


class CellAttachmentStrategy(Enum):
    """Enum representing valid strategies for attaching
    cells to the mutation tree.

    Allowed values:
      - UNIFORM_INCLUDE_ROOT: each node in the tree has equal probability
          of being attached a cell
      - UNIFORM_EXCLUDE_ROOT: each non-root node in the tree has equal probability
          of being attached a cell
    """

    UNIFORM_INCLUDE_ROOT = "UNIFORM_INCLUDE_ROOT"
    UNIFORM_EXCLUDE_ROOT = "UNIFORM_EXCLUDE_ROOT"


def attach_cells_to_tree(
    rng: JAXRandomKey,
    tree: TreeAdjacencyMatrix,
    n_cells: int,
    strategy: CellAttachmentStrategy,
) -> PerfectMutationMatrix:
    """Attaches cells to the mutation tree.

    Args:
        rng: JAX random key
        tree: matrix representing mutation tree
            with the highest index representing the root
        n_cells: number of cells to sample
        strategy: cell attachment strategy.
          See ``CellAttachmentStrategy`` for more information.

    Returns:
        binary matrix of shape ``(n_mutations, n_cells)``,
          where ``n_sites`` is determined from the ``tree``
          NOTE: Last row will be all ones is the root node.
          NOTE: not truncated the last row as shown in the SCITE paper
    """
    if n_cells < 1:
        raise ValueError(f"Number of sampled cells {n_cells} cannot be less than 1.")

    # get no of nodes
    n_nodes = tree.shape[0]

    # sample cell attachment vector
    sigma = _sample_cell_attachment(rng, n_cells, n_nodes, strategy)

    # get ancestor matrix from adjacency matrix
    # get shortest path matrix
    sp_matrix = floyd_warshall(tree)
    # converts the shortest path to ancestor matrix
    ancestor_matrix = shortest_path_to_ancestry_matrix(sp_matrix)

    # get mutation matrix
    mutation_matrix = built_perfect_mutation_matrix(n_nodes, ancestor_matrix, sigma)

    assert mutation_matrix.shape == (n_nodes - 1, n_cells)

    return mutation_matrix


def _sample_cell_attachment(
    rng: JAXRandomKey,
    n_cells: int,
    n_nodes: int,
    strategy: CellAttachmentStrategy,
) -> CellAttachmentVector:
    """Samples the node attachment for each cell given a uniform prior,
        with the value n_nodes corresponding to the root

    Args:
        rng: JAX random key
        n_cells: number of cells
        n_nodes: number of nodes including root,
            nodes counted from 1, root = n_nodes
        strategy: ell attachment strategy.
          See ``CellAttachmentStrategy`` for more information.

    Returns:
        \\sigma - sample/cell attachment vector
            where elements are sampled node indices
            (index+1) of \\sigma corresponds to cell/sample number
    """

    # define probabilities to sample nodes - respective of cell attachment strategy
    if strategy == CellAttachmentStrategy.UNIFORM_INCLUDE_ROOT:
        nodes = jnp.arange(1, n_nodes + 1)
    elif strategy == CellAttachmentStrategy.UNIFORM_EXCLUDE_ROOT:
        nodes = jnp.arange(1, n_nodes)
    else:
        raise ValueError(f"CellAttachmentStrategy {strategy} is not valid.")

    # sample vector - uniform sampling is implicit
    sigma = random.choice(rng, nodes, shape=[n_cells])

    return sigma


def floyd_warshall(tree: TreeAdjacencyMatrix) -> np.ndarray:
    """Implement the Floyd-Warshall on an adjacency matrix A.

        Complexity: O(n^3)
    Args:
    tree : `np.array` of shape (n, n)
        Adjacency matrix of an input graph. If tree[i, j] is `1`, an edge
        connects nodes `i` and `j`.
        Nodes are required to be their own parent, i.e. Adjacency matrix must have
        unity on diagonal.

    Returns
    An `np.array` of shape (n, n), corresponding to the shortest-path
    matrix obtained from tree, -1 represents no path i.e. infinite path length.
    """
    # check dimensions
    if tree.shape[0] != tree.shape[1]:
        raise ValueError(
            f"The input adjacency matrix is not a square matrix. Shape :{tree.shape}"
        )

    if not (np.array_equal(np.diagonal(tree), np.ones(tree.shape[0]))):
        raise ValueError(
            "Nodes are their own parent, Adjacency matrix needs 1 on the diagonal."
        )

    tree = np.array(tree)
    # define a quasi infinity
    inf = 10**7
    # set zero entries to quasi infinity
    tree[~np.eye(tree.shape[0], dtype=bool) & np.where(tree == 0, True, False)] = inf
    # get shape of A - assume n x n
    n = np.shape(tree)[0]
    # make copy of A
    dist = list(map(lambda p: list(map(lambda j_count: j_count, p)), tree))
    # Adding vertices individually
    for r in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][r] + dist[r][j])
    # replace quasi infinity with -1
    dist = np.array(dist)
    sp_mat = np.where(dist >= inf, -1, dist)
    return sp_mat


def shortest_path_to_ancestry_matrix(sp_matrix: np.ndarray) -> AncestorMatrix:
    """Convert the shortest path matrix to an ancestry matrix.

    Args:
        sp_matrix: shortest path matrix,
            with no path indicated by -1

    Returns:
        Ancestry matrix, (n_mutations+1, n_mutations +1)
                    every zero/positive shortest path is ancestry.

    """
    ancestor_mat = np.where(sp_matrix >= 1, 1, 0)

    return ancestor_mat


def get_descendants_fw(
    adj_matrix: adjacency_matrix,
    node: int,
) -> np.ndarray:
    """Get the descendants of a node.
       Assumes indices as node labels.

    Args:
        adj_matrix: (n_sites, n_sites) adjacency matrix of tree,
                    with root
        node: node index

    Returns:
        Descendant vector of node.
    """
    sp_matrix = floyd_warshall(adj_matrix)
    ancestor_matrix = shortest_path_to_ancestry_matrix(sp_matrix)
    descendants = np.array(ancestor_matrix[node, :])

    return descendants


def built_perfect_mutation_matrix(
    n_nodes: int,
    ancestor_matrix: AncestorMatrix,
    sigma: CellAttachmentVector,
) -> PerfectMutationMatrix:
    """Built perfect mutation matrix from adjacency matrix and cell attachment vector.

    Args:
        n_nodes: number of nodes including root,
        ancestor_matrix: ancestor matrix of mutation tree.
        sigma: sampled cell attachment vector
            of length n_cells and values denoting the sampled cells
            counting from 1 to n_nodes (where n_nodes represents the root
            included in sampling)

    Returns:
        Perfect mutation matrix based on Eqn. 11) in on
        p. 14 of the original SCITE paper.
    """
    nodes = np.arange(n_nodes)

    # Eqn. 11.
    # NB: sigma -1  only adjust to python indexing
    mutation_matrix = ancestor_matrix[nodes[:, None], sigma - 1]

    # truncate the last row - the root
    mutation_matrix = mutation_matrix[:-1, :]

    return mutation_matrix


def adjacency_to_root_dfs(
    adj_matrix: adjacency_matrix,
    labels: np.ndarray = None,  # type: ignore
    root_label: int = None,  # type: ignore
) -> TreeNode:
    """Convert adjacency matrix to tree in tree.TreeNode
        traverses a tree using depth first search.

    Args:
        adj_matrix: np.ndarray
            with no self-loops (i.e. diagonal is all zeros)
            and the root as the highest index node
        labels: np.ndarray
            labels of the nodes, if different from indices
            will be used as node names
        root_label: int
            root node root_label, if not provided, will be the highest index node
    Returns:
        root: TreeNode containing the entire tree
    """
    # Sanity checks
    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square")

    if labels is not None:
        if adj_matrix.shape[0] != len(labels):
            raise ValueError("Number of labels must match number of nodes")

    if adj_matrix.shape[0] <= 1:
        raise ValueError("Adjacency matrix must contain at least two nodes")

    # Check if labels are provided - if not, use indices
    if labels is None:
        labels = np.arange(len(adj_matrix))

    # Determine the root node (node with the highest index)
    if root_label is None:
        root_idx = len(adj_matrix) - 1
    else:
        root_idx = np.where(labels == root_label)[0][0]

    # Create a stack to keep track of nodes to visit
    stack = [root_idx]

    # Create a set to keep track of visited nodes
    visited = set()

    # Create a list to keep track of nodes
    child_parent = {}

    # Create a list to keep track of TreeNodes
    list_tree_node = np.empty(len(adj_matrix), dtype=TreeNode)

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

        # print(f"Parent of node {node} is {child_parent[node]}")

        if node == root_idx:
            root = TreeNode(name=labels[node], data=None, parent=None)
            list_tree_node[node] = root
        else:
            # Recall parent
            parent = child_parent[node]
            child = TreeNode(
                name=labels[node],
                data=None,
                parent=list_tree_node[parent],  # type: ignore
            )
            list_tree_node[node] = child

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

    root = list_tree_node[root_idx]

    return root


class CellSimulationModel(BaseModel):
    """Model for Cell Simulation Parameters.

    Note: used in the simulation of cells and mutations gen_sim_data()
    """

    n_cells: int
    n_mutations: int
    fpr: float = 1.24e-06
    fnr: float = 0.097
    na_rate: float = 1.4e-02
    observe_homozygous: bool = False
    strategy: CellAttachmentStrategy = CellAttachmentStrategy.UNIFORM_INCLUDE_ROOT

    @validator("n_cells")
    def realistic_cell_number(cls, v):
        """Validate that the number of cells is realistic."""
        if v > 10000:
            logging.warning(f"Simulating {v} cells")
            raise ValueError("Cell number must be less than 10000")
        return v

    @validator("n_mutations")
    def realistic_mutation_number(cls, v):
        """Validate that the number of mutations is realistic."""
        if v > 100:
            logging.warning(f"Simulating {v} mutations")
            raise ValueError("Mutation number must be less than 100")
        return v

    @validator("fpr")
    def realistic_fpr(cls, v):
        """Validate that the false positive rate is realistic."""
        if not (0 <= v < 1):
            logging.warning(f"FPR {v}")
            raise ValueError("FPR must be in the interval (0,1)")
        return v

    @validator("fnr")
    def realistic_fnr(cls, v):
        """Validate that the false negative rate is realistic."""
        if not (0 <= v < 1):
            logging.warning(f"FNR {v} cells")
            raise ValueError("FNR must be less in the interval (0,1)")
        return v

    @validator("na_rate")
    def realistic_na_rate(cls, v):
        """Validate that the NA rate is realistic."""
        if not (0 <= v < 1):
            logging.warning(f"NA Rate {v}")
            raise ValueError("NA Rate must be in the interval (0,1)")
        return v


def gen_sim_data(
    params: CellSimulationModel,
    rng: JAXRandomKey,
) -> dict:
    """
    Generates cell mutation matrix for one tree.

    Args:
        params: TypedDict from parser of cell_simulation.py
            input parameters from parser for simulation
        rng: JAX random number generator
    Returns:
        data: dict
            data dictionary containing - serialised data for the tree:
                adjacency_matrix - adjacency matrix of the tree
                perfect_mutation_mat - perfect mutation matrix
                noisy_mutation_mat - noisy mutation matrix
                                    (only if fpr > 0 | fnr > 0 | na_rate > 0)
                root - root of the tree (TreeNode)

    """

    # Parameters
    n_cells = params.n_cells
    n_mutations = params.n_mutations
    fnr = params.fnr
    fpr = params.fpr
    na_rate = params.na_rate
    observe_homozygous = params.observe_homozygous
    strategy = params.strategy

    # Random Seeds
    rng_tree, rng_cell_attachment, rng_noise = random.split(rng, 3)

    # Generate Trees
    #  generate random trees (uniform sampling) as adjacency matrix / add +1 for root
    tree = _generate_random_tree_adj_mat(rng_tree, n_nodes=n_mutations + 1)

    # Attach Cells To Tree
    # convert adjacency matrix to self-connected tree - in tree_inference format
    np.fill_diagonal(tree, 1)
    # attach cells to tree - generate perfect mutation matrix
    perfect_mutation_mat = attach_cells_to_tree(
        rng_cell_attachment, tree, n_cells, strategy
    )

    # Add Noise to Perfect Mutation Matrix
    noisy_mutation_mat = None
    if (fnr > 0) or (fpr > 0) or (na_rate > 0):
        noisy_mutation_mat = add_noise_to_perfect_matrix(
            rng_noise, perfect_mutation_mat, fpr, fnr, na_rate, observe_homozygous
        )

    # Package Data
    # format tree for saving
    root = adjacency_to_root_dfs(tree)
    root_serialized = serialize.serialize_tree_to_dict(
        root, serialize_data=lambda x: None
    )

    # Save the data to a JSON file
    # Create a dictionary to hold matrices
    if noisy_mutation_mat is not None:
        data = {
            "adjacency_matrix": tree.tolist(),
            "perfect_mutation_mat": perfect_mutation_mat.tolist(),
            "noisy_mutation_mat": noisy_mutation_mat.tolist(),
            "root": root_serialized,
        }
    else:
        data = {
            "adjacency_matrix": tree.tolist(),
            "perfect_mutation_mat": perfect_mutation_mat.tolist(),
            "root": root_serialized,
        }

    return data
