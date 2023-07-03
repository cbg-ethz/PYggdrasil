"""Experiment mark02

 Investigate convergence of SCITE MCMC chains,
 given different initial points with tree
 distances"""


# imports
import jax.numpy as jnp
import pyggdrasil as yg

from pyggdrasil.tree_inference import CellSimulationId, TreeType, TreeId, McmcConfig

#####################
# Environment variables
WORKDIR = "../data"
#WORKDIR = "/cluster/home/gkoehn/data"

#####################
experiment="mark02"

# Metrics: Distances / Similarity Measure to use
metrics = ["MP3"]  # also AD <-- configure distances here

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

n_mutations = [5] # <-- configure number of mutations here
n_cells = [200] # <-- configure number of cells here

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
initial_points = [ # (mcmc_seed, init_tree_type, init_tree_seed)
    (42, 'r', 45),
    (12, 'r', 34),
    (34, 'r', 42),
    (79, 'r', 89),
]

# MCMC config
n_samples = 5000 # <-- configure number of samples here

#####################
#####################


def make_all_mark02():
    """Make all final output file names."""

    #f"{WORKDIR}/{experiment}/plots/{McmcConfig}/{CellSimulationId}/"

    # "AD_hist.svg" and "MP3_hist.svg"

    filepaths = []
    filepath = f'{WORKDIR}/{experiment}/plots/'
    # add +1 to n_mutation to account for the root mutation
    n_nodes = [n_mutation + 1 for n_mutation in n_mutations]


    # # make mcmc configs
    # mcmc_config_id_ls = []
    # for error_name, error in errors.items():
    #     mcmc_config_id_ls.append(
    #         McmcConfig(
    #             n_samples=n_samples,
    #             fpr=error["fpr"],
    #             fnr=error["fnr"]
    #         ).id()
    #     )


    # make true tree ids for cell simulation
    tree_id_ls = []
    for tree_type in tree_types:
        for tree_seed in tree_seeds:
            for n_node in n_nodes:
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
                            filepath + mc + "/" + str(cs) + "/" + str(true_tree_id) + "/" + each_metric + ".svg"
                        )

    return filepaths


rule mark02:
    """Make multiple chain histograms"""
    input:
        make_all_mark02()


rule combined_chain_histogram:
    """Make combined chain histogram"""
    input:
        # calls analyze_metric rule
        all_chain_metrics = ['{WORKDIR}/{experiment}/analysis/MCMC_' + str(mcmc_seed) + '-{mutation_data_id}-iT_'
                             + str(init_tree_type)+ '_{n_nodes,\d+}_' + str(init_tree_seed) +
                             '-{mcmc_config_id}/T_{base_tree_type}_{n_nodes,\d+}_{base_tree_seed,\d+}/{metric}.json'
                             for mcmc_seed, init_tree_type, init_tree_seed in initial_points]

    output:
        combined_chain_histogram = '{WORKDIR}/{experiment}/plots/{mcmc_config_id}/{mutation_data_id}/'
                                   'T_{base_tree_type}_{n_nodes,\d+}_{base_tree_seed,\d+}/{metric}.svg',



