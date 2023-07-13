"""Experiment mark03

 Investigate convergence of SCITE MCMC chains,
 given different initial points with tree
 distances"""

# imports
import matplotlib.pyplot as plt
import matplotlib

from pathlib import Path

import pyggdrasil as yg

import jax
import jax.numpy as jnp


from pyggdrasil.tree_inference import CellSimulationId, TreeType, TreeId, McmcConfig

#####################
# Environment variables
DATADIR = "../data"
#DATADIR = "/cluster/work/bewi/members/gkoehn/data"

#####################
experiment="mark03"

# Metrics: Distances / Similarity Measure to use
metrics = ["MP3"] #, "AD", "log_prob"]  # <-- configure distances here

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

n_mutations = [5] #, 10, 30, 50] # <-- configure number of mutations here
n_cells = [200] #, 1000, 5000] # <-- configure number of cells here

# Homozygous mutations
observe_homozygous = False # <-- configure whether to observe homozygous mutations here

# cell attachment strategy
cell_attachment_strategy = yg.tree_inference.CellAttachmentStrategy.UNIFORM_EXCLUDE_ROOT # <-- configure cell attachment strategy here

# cell simulation seed
CS_seed = 42 # <-- configure cell simulation seed here

#####################
# True Tree Parameters
tree_types = ["r"] # <-- configure tree type here ["r","s","d"]
tree_seeds = [42] # <-- configure tree seed here

#####################
#####################
# MCMC Parameters

# define 4 initial points, different chains
# given each error rate, true tree, no of cells and mutations
# make random trees and mcmc seeds
desired_counts = {
    'd': 3, #10,  # Deep Trees
    'r': 3, #10,  # Random Trees
    's': 1,   # Star Tree
    'h': 1,  # Huntress Tree, derived from the cell simulation
#    'mcmc': 5 # MCMC Move Trees
}

# MCMC config
n_samples = 100 #2000 # <-- configure number of samples here

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
            indicating the number of initial points to generate for each type of tree.

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

# Generate the initial points of mcmc chains
initial_points = make_initial_points_mark03(desired_counts)


def make_all_mark03():
    """Make all final output file names."""

    #f"{DATADIR}/{experiment}/plots/{McmcConfig}/{CellSimulationId}/"

    # "AD.svg" and "MP3.svg" or log_prob.svg

    filepaths = []
    filepath = f'{DATADIR}/{experiment}/plots/'
    # add +1 to n_mutation to account for the root mutation
    n_nodes = [n_mutation + 1 for n_mutation in n_mutations]

    # make true tree ids for cell simulation
    tree_id_ls = []
    for tree_type in tree_types:
        for tree_seed in tree_seeds:
            for n_node in n_nodes:
                if tree_type == "s":
                    tree_id_ls.append(TreeId(tree_type=TreeType(tree_type),n_nodes=n_node))
                else:
                    tree_id_ls.append(TreeId(tree_type=TreeType(tree_type),n_nodes=n_node,seed=tree_seed))


    # make cell simulation ids
    #cell_simulation_id_ls = []
    for true_tree_id in tree_id_ls:
        for n_cell in n_cells:
                for error_name, error in errors.items():
                    #cell_simulation_id_ls.append(
                    cs = CellSimulationId(
                            seed=CS_seed,
                            tree_id=true_tree_id,
                            n_cells=n_cell,
                            fpr=error["fpr"],
                            fnr=error["fnr"],
                            na_rate=rate_na,
                            observe_homozygous = observe_homozygous,
                            strategy=cell_attachment_strategy
                        )
                    #)

                    mc = McmcConfig(
                        n_samples=n_samples,
                        fpr=error["fpr"],
                        fnr=error["fnr"]
                    ).id()

                    for each_metric in metrics:
                        filepaths.append(
                            filepath + mc + "/" + str(cs) + "/" + str(true_tree_id) + "/" + each_metric + "_iter.svg"
                        )

    return filepaths


rule mark03:
    """Make mark03 plots."""
    input:
        make_all_mark03()


def make_combined_metric_iteration_in():
    """Make input for combined_metric_iteration rule."""
    input = []
    tree_type = []

    for mcmc_seed, init_tree_type, init_tree_seed in initial_points:
        # catch the case where init_tree_type is star tree
        if init_tree_type == 's':
            input.append('{DATADIR}/mark03/analysis/MCMC_' + str(mcmc_seed) + '-{mutation_data_id}-iT_'
                             + str(init_tree_type)+ '_{n_nodes,\d+}' +
                             '-{mcmc_config_id}/T_{base_tree_type}_{n_nodes,\d+}_{base_tree_seed,\d+}/{metric}.json')
        elif init_tree_type == 'h':
            input.append('{DATADIR}/mark03/analysis/MCMC_' + str(mcmc_seed) + '-{mutation_data_id}-' +
                            'iHUN-{mutation_data_id}' +
                             '-{mcmc_config_id}/T_{base_tree_type}_{n_nodes,\d+}_{base_tree_seed,\d+}/{metric}.json')
        else:
            input.append('{DATADIR}/mark03/analysis/MCMC_' + str(mcmc_seed) + '-{mutation_data_id}-iT_'
                             + str(init_tree_type)+ '_{n_nodes,\d+}_' + str(init_tree_seed) +
                             '-{mcmc_config_id}/T_{base_tree_type}_{n_nodes,\d+}_{base_tree_seed,\d+}/{metric}.json')
        tree_type.append(init_tree_type)

    return input, tree_type

rule combined_metric_iteration_plot:
    """Make combined metric iteration plot.
    
    For each metric, make a plot with all the chains, where
    each initial tree type is a different color.
    """
    input:
        # calls analyze_metric rule
        all_chain_metrics = make_combined_metric_iteration_in()[0]
    wildcard_constraints:
    # metric wildcard cannot be log_prob
        metric = r'(?!(log_prob))\w+'
    output:
        combined_metric_iter = '{DATADIR}/{experiment}/plots/{mcmc_config_id}/{mutation_data_id}/'
                                    'T_{base_tree_type}_{n_nodes,\d+}_{base_tree_seed,\d+}/{metric}_iter.svg',
    run:
        # load the data
        # for each metric/chain, load the json file
        distances_chains = []
        # get the initial tree type
        initial_tree_type = make_combined_metric_iteration_in()[1]

        for each_chain_metric in input.all_chain_metrics:
            # load the distances
            _ , distances = yg.serialize.read_metric_result(Path(each_chain_metric))
            # append to the list
            distances_chains.append(distances)

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Define the list of colors to repeat
        colors = {'h': 'red',
                  's': 'green',
                  'd' : 'blue',
                  'r' : 'orange',
                  'mcmc' : 'purple'
                  }
        labels = {'h': 'Huntress',
                  's': 'Star',
                  'd' : 'Deep',
                  'r' : 'Random',
                  'mcmc' : 'MCMC5'
                  }

        # Define opacity and line style
        alpha = 0.6
        line_style = 'solid'

        # Plot each entry of distance chain as a line with a color unique to the
        # initial tree type onto one axis

        # Plot each entry of distance chain as a line with a color unique to the
        # initial tree type onto one axis
        for i, distances in enumerate(distances_chains):
            color = colors[initial_tree_type[i]]
            ax.plot(distances,color=color,label=f'{labels[initial_tree_type[i]]}',
                alpha=alpha,linestyle=line_style)

        # Set labels and title
        ax.set_xlabel(f"Distance/Similarity: {wildcards.metric}")
        ax.set_ylabel('iteration')

        # Add a legend
        ax.legend()

        # save the histogram
        fig.savefig(Path(output.combined_metric_iter))


def make_combined_log_prob_iteration_in():
    """Make input for combined_metric_iteration rule."""
    input = []

    for mcmc_seed, init_tree_type, init_tree_seed in initial_points:
        # catch the case where init_tree_type is star tree
        if init_tree_type == 's':
            input.append('{DATADIR}/mark03/analysis/MCMC_' + str(mcmc_seed) + '-{mutation_data_id}-iT_'
                             + str(init_tree_type)+ '_{n_nodes,\d+}' +
                             '-{mcmc_config_id}/log_prob.json')
        elif init_tree_type == 'h':
            input.append('{DATADIR}/mark03/analysis/MCMC_' + str(mcmc_seed) + '-{mutation_data_id}-i' +
                            'HUN-{mutation_data_id}' +
                             '-{mcmc_config_id}/T_{base_tree_type}_{n_nodes,\d+}_{base_tree_seed,\d+}/log_prob.json')
        else:
            input.append('{DATADIR}/mark03/analysis/MCMC_' + str(mcmc_seed) + '-{mutation_data_id}-iT_'
                             + str(init_tree_type)+ '_{n_nodes,\d+}_' + str(init_tree_seed) +
                             '-{mcmc_config_id}/T_{base_tree_type}_{n_nodes,\d+}_{base_tree_seed,\d+}/log_prob.json')
    return input


rule combined_logProb_iteration_plot:
    """Make combined logProb iteration plot."""
    input:
        # calls analyze_metric rule
        all_chain_logProb = make_combined_log_prob_iteration_in()

    output:
        combined_logP_iter = '{DATADIR}/{experiment}/plots/{mcmc_config_id}/{mutation_data_id}/T_{base_tree_type}_{n_nodes,\d+}_{base_tree_seed,\d+}/log_prob_iter.svg',
    run:
        # load the data
        # for each metric/chain, load the json file
        logP_chains = []
        # get the initial tree type
        initial_tree_type = make_combined_metric_iteration_in()[1]

        for each_chain_metric in input.all_chain_logProb:
            # load the distances
            _, logP = yg.serialize.read_metric_result(Path(each_chain_metric))
            # append to the list
            logP_chains.append(logP)

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Define the list of colors to repeat
        colors = {'h': 'red',
                  's': 'green',
                  'd': 'blue',
                  'r': 'orange',
                  'mcmc': 'purple'
                  }
        labels = {'h': 'Huntress',
                  's': 'Star',
                  'd': 'Deep',
                  'r': 'Random',
                  'mcmc': 'MCMC5'
                  }

        # Define opacity and line style
        alpha = 0.6
        line_style = 'solid'

        # Plot each entry of distance chain as a line with a color unique to the
        # initial tree type onto one axis

        # Plot each entry of distance chain as a line with a color unique to the
        # initial tree type onto one axis
        for i, logP in enumerate(logP_chains):
            color = colors[initial_tree_type[i]]
            ax.plot(logP,color=color,label=f'{labels[initial_tree_type[i]]}',
                alpha=alpha,linestyle=line_style)

        # Set labels and title
        ax.set_xlabel(f"Log Probability:" +  r"$\log(P(D|T,\theta))$")
        ax.set_ylabel('iteration')

        # Add a legend
        ax.legend()

        # save the histogram
        fig.savefig(Path(output.combined_logP_iter))