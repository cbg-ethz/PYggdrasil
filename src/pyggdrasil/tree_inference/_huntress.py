"""Tree inference with the HUNTRESS algorithm."""

import anytree
import numpy as np
import pandas as pd
import scphylo
import pyggdrasil._scphylo_utils as utils


def huntress_tree_inference(
    mutations: np.ndarray,
    false_positive_rate: float,
    false_negative_rate: float,
    n_threads: int = 1,
) -> anytree.Node:
    """Runs the HUNTRESS algorithm.

    Args:
        mutations: binary array with entries 0 or 1,
          depending on whether the mutation is present or not.
          Shape (n_cells, n_sites)
        false_positive_rate: false positive rate, in [0, 1)
        false_negative_rate: false negative rate, in [0, 1)
        n_threads: number of threads to be used, default 1

    Returns:
        inferred tree. The root node (wildtype) has name `n_mutations`
          and the other nodes are named with integer labels using
          the mutation index (starting at 0)

    Example:
        For a matrix of shape (n_cells, 4) an example tree can be
        4
        ├── 0
        │   ├── 1
        │   └── 2
        └── 3
    """
    assert 0 <= false_positive_rate < 1
    assert 0 <= false_negative_rate < 1

    n_cells, n_mutations = mutations.shape

    mutations_dataframe = pd.DataFrame(
        mutations, columns=[str(i) for i in range(n_mutations)]
    )

    clean_matrix = scphylo.tl.huntress(
        mutations_dataframe,
        alpha=false_positive_rate,
        beta=false_negative_rate,
        n_threads=n_threads,
    )
    return utils.dataframe_to_tree(
        clean_matrix,
        root_name=n_mutations,
        mutation_name_mapping=int,
    )
