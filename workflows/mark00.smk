"""Experiment mark00 - small test run of MCMC"""

from pyggdrasil.tree_inference import McmcConfig, TreeId, TreeType, CellSimulationId, CellAttachmentStrategy

################################################################################
# Define Environment
WORKDIR = "../data"

###############################################
## Experiment mark00
experiment = "mark00"
# Mutation Data / Cell Simulation
CS_seed = 42
n_cells = 1000
CS_fpr = 0.4
CS_fnr = 0.4
CS_na = 0.0
observe_homozygous = False
cell_attachment_strategy = CellAttachmentStrategy.UNIFORM_EXCLUDE_ROOT
# mutations
mutations = 7
nodes= mutations + 1
# True Tree
true_tree_type = TreeType("r")
true_tree_seed = 5
true_tree_n_nodes = nodes
# Initial Tree
initial_tree_type = TreeType("d")
initial_tree_seed = 4545
initial_tree_n_nodes = nodes
# MCMC Parameters
mcmc_seed = 42
mcmc_config_id = McmcConfig(n_samples=100, fpr=0.4, fnr=0.4).id()
###############################################


###############################################
# Support variables
init_tree_id = TreeId(initial_tree_type, initial_tree_n_nodes, initial_tree_seed)
true_tree_id = TreeId(true_tree_type, true_tree_n_nodes, true_tree_seed)
cell_simulation_id = CellSimulationId(seed=CS_seed, tree_id=true_tree_id, n_cells=n_cells, fpr=CS_fpr, fnr=CS_fnr,na_rate=CS_na, observe_homozygous = observe_homozygous, strategy=cell_attachment_strategy)
###############################################


###############################################
# Output files
# most likely tree
top_tree_info = f"{WORKDIR}/{experiment}/plots/MCMC_{mcmc_seed}-{cell_simulation_id}-i{init_tree_id}-{mcmc_config_id}/top_tree_info.json"
topTree = f"{WORKDIR}/{experiment}/plots/MCMC_{mcmc_seed}-{cell_simulation_id}-i{init_tree_id}-{mcmc_config_id}/top_tree_1.svg"
#log_prob =
initial_tree = f'{WORKDIR}/{experiment}/plots/MCMC_{mcmc_seed}-{cell_simulation_id}-i{init_tree_id}-{mcmc_config_id}/init_tree.svg'
# metrics
# choose from ['AD', 'MP3', 'log_prob'] i.e. all defined in pyggdrasil.analyze.Metrics
ancestor_descendant_iteration = f'{WORKDIR}/{experiment}/plots/MCMC_{mcmc_seed}-{cell_simulation_id}-i{init_tree_id}-{mcmc_config_id}/{true_tree_id}/AD.svg'
mp3_iteration = f'{WORKDIR}/{experiment}/plots/MCMC_{mcmc_seed}-{cell_simulation_id}-i{init_tree_id}-{mcmc_config_id}/{true_tree_id}/MP3.svg'
log_prob = f'{WORKDIR}/{experiment}/plots/MCMC_{mcmc_seed}-{cell_simulation_id}-i{init_tree_id}-{mcmc_config_id}/log_prob.svg'
# true trees
true_tree_plot = f'{WORKDIR}/{experiment}/plots/MCMC_{mcmc_seed}-{cell_simulation_id}-i{init_tree_id}-{mcmc_config_id}/true_tree.svg'
true_tree_found = f'{WORKDIR}/{experiment}/analysis/MCMC_{mcmc_seed}-{cell_simulation_id}-i{init_tree_id}-{mcmc_config_id}/true_trees_found.txt',
###############################################


###############################################
rule mark00:
    input:
        # Plots and json of top most likely trees
        top_tree_info = top_tree_info,
        topTree_plots = topTree,
        # plot of initial tree
        initial_tree_plot = initial_tree,
        # metrics vs iteration
        ancestor_descendant_plot = ancestor_descendant_iteration,  # known to fail for some trees (conjecture: small / dissimilar trees)
        mp3_iteration_plot = mp3_iteration,
        log_prob_iteration_plot = log_prob,
        # list of iterations in which true trees where found
        true_tree_found_info = true_tree_found,
        # plot of true tree
        plot_true_tree = true_tree_plot
