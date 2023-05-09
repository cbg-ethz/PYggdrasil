"""scPhylo utilities."""
from typing import Union

import anytree
import numpy as np
import pandas as pd


def tree_to_dataframe(
    root: anytree.Node,
    include_root: bool = False,
    name_to_str_mapping=str,
) -> pd.DataFrame:
    """Maps the tree with root and mutations to
    the conflict-free data frame required by scPhylo.

    Args:
        root: root of the tree
        include_root: whether the root should be treated as a mutation
          and included in the generated genotypes
        name_to_str_mapping: scPhylo's data frame needs to have columns which are `str`.
          This mapping of signature `mutation -> str` will be used to convert mutation
          names to strings

    Returns:
        data frame with columns being the mutations (after `name_to_str_mapping`)

    Note:
        The infinite sites assumption must hold for this tree.
        The labels need to be necessarily unique.
    """
    all_names = set()
    cells = []

    def step(
        node: anytree.Node,
        mutations_present: set[str],
    ) -> None:
        """Build genotype matrices by recursively traversing the tree."""
        label = name_to_str_mapping(node.name)

        mutations_new = mutations_present | {label}

        all_names.add(label)
        cells.append(mutations_new)

        for child in node.children:
            step(
                node=child,
                mutations_present=mutations_new,
            )

    step(root, mutations_present=set())

    if not include_root:
        root_name = name_to_str_mapping(root.name)
        all_names.remove(root_name)
        for s in cells:
            s.remove(root_name)

    columns = list(all_names)
    mutations = np.zeros((len(cells), len(columns)), dtype=int)

    for i, cell in enumerate(cells):
        for mutation in cell:
            mutations[i, columns.index(mutation)] = 1

    return pd.DataFrame(
        mutations, columns=columns, index=[f"Cell-{i}" for i, _ in enumerate(cells, 1)]
    )


def _ancestor_descendant(anc: tuple[int, ...], desc: tuple[int, ...]) -> bool:
    """Checks if `anc` is the ancestor of `desc` under the infinite sites assumption.

    Args:
        anc: binary tuple of the form (0, 1, 0, ...) representing mutations
          which occurred in potential ancestor cell
        desc: binary tuple of length of `anc` representing mutations which occurred
          in the potential descendant cell

    Returns:
        True iff `anc` is the ancestor of `desc`, False otherwise
    """
    for a, d in zip(anc, desc):
        # If at any locus we have 1 in "ancestor"
        # and 0 in "descendant", something is wrong
        if a > d:
            return False
    # At every locus this works, so we use ISA to conclude that
    # `anc` is really the ancestor of `desc`
    return True


def dataframe_to_tree(
    df: pd.DataFrame,
    root_name: Union[str, int] = "root",
    mutation_name_mapping=lambda x: x,
) -> anytree.Node:
    """Converts a mutation data frame (under
    the infinite sites and non-reversible mutations assumptions,
    and the assumption that in the cell lineage tree there is always
    a cell with exactly one mutation different from its parent.

    Args:
        df: data frame representing the cell genotypes.
          Columns represent the mutation names.
        root_name: name of the root in the mutation tree
        mutation_name_mapping: can be used to convert the mutation
          names (in the data frame columns) to custom mutation names
          stored in the tree nodes

    Returns:
        the root of the tree
    """
    # We want to have a list with genotypes such that
    # (a) each present genotype occurs exactly once (hence np.unique)
    # (b) each genotype is a tuple (0, 1, ..., 0)
    # (c) we definitely have the wildtype/root, i.e., tuple (0, 0, 0, ..., 0)
    # (d) the list is sorted by the number of mutations
    sorted_cells = sorted(
        np.unique(np.vstack([df.values, np.zeros_like(df.values)]), axis=0),
        key=lambda a: a.sum(),
    )
    sorted_cells = [tuple(a.tolist()) for a in sorted_cells]

    # Root node
    root = anytree.Node(root_name)

    # Existing nodes and their genotypes, sorted out by the number of mutations
    # At the beginning we only have the root with profile (0, 0, ..., 0)
    existing_nodes = [(root, sorted_cells[0])]

    def genotype_found(genotype: tuple[int, ...]) -> bool:
        """Checks if the genotype already exists."""
        for _, candidate in existing_nodes:
            if candidate == genotype:
                return True
        return False

    def predecessor(genotype: tuple[int, ...]):
        """Finds the closest ancestor.
        It uses the fact that the list of existing nodes
        is sorted by the number of mutations."""
        for node, prof in reversed(existing_nodes):
            if _ancestor_descendant(prof, genotype):
                return node, prof
        raise ValueError("Ancestor not found.")

    for cell in sorted_cells:
        if genotype_found(cell):
            continue

        # Find the predecessor (nearest ancestor)
        anc_node, anc_prof = predecessor(cell)
        # Find the index of the mutation which doesn't appear in the
        # predecessor, but appears in the current genotype
        i = None
        for i, (anc, desc) in enumerate(zip(anc_prof, cell)):
            if anc != desc:
                break
        if i is None or anc_prof[i] == cell[i]:
            raise ValueError("Mutation not found.")

        # Take the name of the mutation
        name = mutation_name_mapping(df.columns[i])

        # Define a new node with the mutation name
        new_node = anytree.Node(
            name,
            parent=anc_node,
        )
        existing_nodes.append((new_node, cell))

    # Return the root of the tree
    return root
