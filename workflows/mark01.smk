"""Experiment mark00 - small test run of MCMC"""

from pyggdrasil.analyze import Metrics
from pyggdrasil.tree_inference import McmcConfigOptions, TreeId, TreeType, CellSimulationId, CellAttachmentStrategy


###############################################
## Experiment mark00
experiment = "mark01"
# Mutation Data / Cell Simulation
CS_seed = 42
n_cells = 1000
CS_fpr = 0.01
CS_fnr = 0.02
CS_na = 0.0
observe_homozygous = False
cell_attachment_strategy = CellAttachmentStrategy.UNIFORM_EXCLUDE_ROOT
# True Tree
true_tree_type = TreeType("r")
true_tree_seed = 5
true_tree_n_nodes = 5
# Initial Tree
initial_tree_type = TreeType("d")
initial_tree_seed = 4545
initial_tree_n_nodes = 5
# MCMC Parameters
mcmc_seed = 42
mcmc_config = McmcConfigOptions.DEFAULT.value
mcmc_config_id = mcmc_config(n_samples=2500).id()
###############################################


###############################################
# Support variables
init_tree_id = TreeId(initial_tree_type, initial_tree_n_nodes, initial_tree_seed)
true_tree_id = TreeId(true_tree_type, true_tree_n_nodes, true_tree_seed)
cell_simulation_id = CellSimulationId(seed=CS_seed, tree_id=true_tree_id, n_cells=n_cells, fpr=CS_fpr, fnr=CS_fnr,na_rate=CS_na, observe_homozygous = observe_homozygous, strategy=cell_attachment_strategy)
###############################################


###############################################
# Output files
top_tree_info = f"../data/{experiment}/plots/MCMC_{mcmc_seed}-{cell_simulation_id}-i{init_tree_id}-{mcmc_config_id}/top_tree_info.json"
topTree = f"../data/{experiment}/plots/MCMC_{mcmc_seed}-{cell_simulation_id}-i{init_tree_id}-{mcmc_config_id}/top_tree_1.svg"
#log_prob =
initial_tree = f'../data/{experiment}/plots/MCMC_{mcmc_seed}-{cell_simulation_id}-i{init_tree_id}-{mcmc_config_id}/init_tree.svg'


AD_iteration = f'../data/{experiment}/plots/MCMC_{mcmc_seed}-{cell_simulation_id}-i{init_tree_id}-{mcmc_config_id}/{true_tree_id}/AD.svg'
mp3_iteration = f'../data/{experiment}/plots/MCMC_{mcmc_seed}-{cell_simulation_id}-i{init_tree_id}-{mcmc_config_id}/{true_tree_id}/MP3.svg'
log_prob = f'../data/{experiment}/plots/MCMC_{mcmc_seed}-{cell_simulation_id}-i{init_tree_id}-{mcmc_config_id}/log_prob.svg'

true_tree_found = f'../data/{experiment}/analysis/MCMC_{mcmc_seed}-{cell_simulation_id}-i{init_tree_id}-{mcmc_config_id}/true_trees_found.txt',

rule mark01:
    input:
        top_tree_info = top_tree_info,
        topTree = topTree,
        initial_tree = initial_tree,
        ancestor_descendant = AD_iteration,
        mp3_iteration = mp3_iteration,
        log_prob_iteration = log_prob,
        true_tree_found = true_tree_found
