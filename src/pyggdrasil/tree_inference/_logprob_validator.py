"""Implementation of the log-probability functions.

The log-probability functions are used to calculate the log-probability of a tree.

Implements a dumb version of the log-probability function, which is used for testing.
"""


import pyggdrasil.tree_inference._tree as tr

from pyggdrasil.tree_inference import Tree


def _expected(
    tree: Tree,
    mutation_i,
    cell_j,
    cell_attachment,
) -> int:
    """Calculates the expected likelihood of a tree given error rates and data.

    Args:
        tree: tree to calculate the expected likelihood of
        cell_attachment: cell attachment vector, for a single cell
        mutation_i : mutation index

    Returns:
        expected likelihood of the tree
    """

    # get the ancestor matrix
    ancestor_mat = tr._get_ancestor_matrix(tree.tree_topology)
    # truncate the last row, which is the root
    ancestor_mat = ancestor_mat[:-1, :]

    # get parent of mutation_i
    # get column of ancestor matrix for a given cell_attachment
    ancestor_col = ancestor_mat[:, cell_attachment]
    # get indices of non-zero elements
    parents_i = []
    for i, val in enumerate(ancestor_col):
        if val != 0:
            parents_i.append(i)

    if mutation_i in parents_i:
        # mutation_i is a parent of cell_j
        # so return 1
        return 1
    else:
        # mutation_i is not a parent of cell_j
        # so return 0
        return 0
