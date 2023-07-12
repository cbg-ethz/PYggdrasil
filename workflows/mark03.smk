"""Experiment mark03

 Investigate convergence of SCITE MCMC chains,
 given different initial points with tree
 distances"""

# imports
import matplotlib.pyplot as plt

from pathlib import Path

import pyggdrasil as yg

import jax
import jax.numpy as jnp


from pyggdrasil.tree_inference import CellSimulationId, TreeType, TreeId, McmcConfig

# Generate an initial trees - for each category 10 trees
# - Deep Tree
# - Random Tree
# - Star Tree (only one possible)
# - Random Tree -> Huntress (hence generate 10 random trees)
# - (True Random Tree -> 5 MCMC moves -> Initial Tree)
# 5 trees.
#
# Start MCMC runs from: deep, shallow, random, huntress

# Plot the mcmc runs against log_prob / distance to iteration number coloured by their initial tree type
# count the average number of iteration until true tree is reached.

#####################
# Environment variables
#DATADIR = "../data"
DATADIR = "/cluster/work/bewi/members/gkoehn/data"

#####################
experiment="mark03"

# Metrics: Distances / Similarity Measure to use
metrics = ["MP3", "AD"]  # also AD <-- configure distances here

#####################
# Error Parameters
# used for both cell simulation and MCMC inference

# Errors <--- set the error rates here
errors = {
        member.name: member.value.dict()
        for member in yg.tree_inference.ErrorCombinations
}

rate_na = 0.0 # <-- configure NA rate here

#####################
#####################
# Cell Simulation Parameters

n_mutations = [5, 10, 30, 50] # <-- configure number of mutations here
n_cells = [200, 1000, 5000] # <-- configure number of cells here

# Homozygous mutations
observe_homozygous = False # <-- configure whether to observe homozygous mutations here

# cell attachment strategy
cell_attachment_strategy = yg.tree_inference.CellAttachmentStrategy.UNIFORM_EXCLUDE_ROOT # <-- configure cell attachment strategy here

# cell simulation seed
CS_seed = 42 # <-- configure cell simulation seed here

#####################
# True Tree Parameters
tree_types = ["r"] # <-- configure tree type here ["r","s","d"]
tree_seeds = [42, 34] # <-- configure tree seed here

#####################
#####################
# MCMC Parameters

# define 4 initial points, different chains
# given each error rate, true tree, no of cells and mutations
# make random trees and mcmc seeds
desired_counts = {
    'd': 10,  # Deep Trees
    'r': 10,  # Random Trees
    's': 1,   # Star Tree
    'h': 10,  # Huntress Trees
    'mcmc': 5 # MCMC Move Trees
}

# MCMC config
n_samples = 2000 # <-- configure number of samples here

#####################
#####################


def make_initial_points_mark03(desired_counts : dict):
    """Make initial mcmc points for mark03 experiment.

    Args:
        desired_counts: dict
            A dictionary of the form
            {
                'd': 10,  # Deep Trees
                'r': 10,  # Random Trees
                's': 1,   # Star Tree
                'h': 10,  # Huntress Trees
                'mcmc': 5 # MCMC Move Trees
            }

    Returns:
        list of tuples (mcmc_seed, init_tree_type, init_tree_seed)
    """

    key = jax.random.PRNGKey(0)  # Set the initial PRNG key
    new_trees = []
    seed_pool = set()
    for init_tree_type, count in desired_counts.items():
        for _ in range(count):
            key, subkey = jax.random.split(key)  # Split the PRNG key
            mcmc_seed = jax.random.randint(subkey,(),1,100)  # Generate a random MCMC seed
            key, subkey = jax.random.split(key)  # Split the PRNG key
            init_tree_seed = jax.random.randint(subkey,(),1,100)  # Generate a random seed for init_tree
            while init_tree_seed.item() in seed_pool:  # Ensure the seed is unique
                key, subkey = jax.random.split(key)  # Split the PRNG key
                init_tree_seed = jax.random.randint(subkey,(),1,100)
            new_trees.append((mcmc_seed.item(), init_tree_type, init_tree_seed.item()))
            seed_pool.add(init_tree_seed.item())
    return new_trees


def make_all_mark03():
    """Make all final output file names."""

    raise NotImplementedError("TODO: implement this")


rule mark03:
    """Make mark03 plots."""
    input:
        make_all_mark03()
