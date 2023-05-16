"""Analyze trees from MCMC runs."""

import jax.numpy as jnp
from typing import Union, Callable
import xarray as xr

from pyggdrasil.tree_inference import Tree
from pyggdrasil.tree_inference._tree import _reorder_tree
from pyggdrasil.interface import MCMCSample, PureMcmcData, AugmentedMcmcData

McmcData = Union[PureMcmcData, AugmentedMcmcData]


def to_pure_mcmc_data(mcmc_samples: list[MCMCSample]) -> PureMcmcData:
    """Converts McmcRunData to PureMcmcData.

    Takes a list of MCMCSamples
    converts it into a xarray easy to plot.

    Args:
        mcmc_samples : McmcRunData
    Returns:
        PureMcmcData
    """

    # Assuming you have a list of xarray datasets called `mcmc_samples`
    combined_dataset = xr.concat(mcmc_samples, dim="sample")

    # Assigning the iteration number as a coordinate
    iterations = [ds.iteration.item() for ds in mcmc_samples]
    combined_dataset["iteration"] = ("sample", iterations)

    # Set the iteration coordinate as the dimension for easier indexing
    combined_dataset = combined_dataset.set_index(sample="iteration")

    # Create an instance of PureMcmcData
    pure_data = PureMcmcData(
        iteration=combined_dataset["iteration"],
        tree=combined_dataset["tree"],
        log_probability=combined_dataset["log-probability"],
    )

    return pure_data


def add_distance_to_mcmc_data(
    mcmc_samples: PureMcmcData, dist_fn: Callable[[Tree], float]
) -> AugmentedMcmcData:
    """Add distance to MCMC data.

    Args:
        mcmc_samples : McmcRunData
        dist_fn : Callable[[Tree], float]
    Returns:
        AugmentedMcmcData
    """

    # Apply the calculate_distance function to each sample in the dataset
    distance = xr.apply_ufunc(
        dist_fn,  # Custom function to calculate distance
        mcmc_samples.tree,  # Input variable(s)
        input_core_dims=[
            ["from_node_k", "to_node_k"]
        ],  # Specify the dimensions of the input variable(s)
        output_core_dims=[["sample"]],  # Specify the dimensions of the output variable
        vectorize=True,  # Vectorize the function for better performance
        dask="parallelized",  # Enable parallelization if using dask arrays
        output_dtypes=[float],  # Specify the dtype of the output variable
    )

    # Add the distance variable to the combined dataset
    # Create an instance of AugmentedMcmcData
    augmented_samples = AugmentedMcmcData(
        iteration=mcmc_samples.iteration,
        tree=mcmc_samples.tree,
        log_probability=mcmc_samples.log_probability,
        distance=distance,
    )

    return augmented_samples


def is_same_tree(tree1: Tree, tree2: Tree) -> bool:
    """Check if two trees are the same.

    Args:
        tree1 : Tree
        tree2 : Tree
    Returns:
        bool
    """

    result = bool(
        jnp.all(tree1.tree_topology == tree2.tree_topology)
        and jnp.all(tree1.labels == tree2.labels)
    )

    # if the trees are not the same, check if their labels are just ordered differently
    if result is False:
        tree2_reordered = _reorder_tree(
            tree2, from_labels=tree2.labels, to_labels=tree1.labels
        )
        result = jnp.all(
            tree1.tree_topology == tree2_reordered.tree_topology
        ) and jnp.all(tree1.labels == tree2_reordered.labels)

    return bool(result)


def check_run_for_tree(
    desired_tree: Tree, mcmc_samples: McmcData
) -> Union[tuple[int, Tree, float], bool]:
    """Check if a tree is in an MCMC run.

    Args:
        desired_tree : Tree
        mcmc_samples : McmcRunData
    Returns:
        bool or tuple[int, Tree, float]
    """

    # Check if the desired tree is in the MCMC run
    for i, tree in enumerate(mcmc_samples.tree):
        if is_same_tree(desired_tree, tree):
            return i, tree, mcmc_samples.log_probability[i].item()

    return False
