"""Experiment mark02

 Investigate convergence of SCITE MCMC chains,
 given different initial points with tree
 distances"""


# imports
import matplotlib.pyplot as plt

from pathlib import Path

import pyggdrasil as yg

from pyggdrasil.tree_inference import CellSimulationId, TreeType, TreeId, McmcConfig

#####################
# Environment variables
#DATADIR = "../data"
DATADIR = "/cluster/work/bewi/members/gkoehn/data"

#####################
experiment="mark02"

# Metrics: Distances / Similarity Measure to use
metrics = ["MP3", "AD", "DL"]  # also AD <-- configure distances here

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
tree_seeds = [42]#, 34] # <-- configure tree seed here

#####################
#####################
# MCMC Parameters

# define 4 initial points, different chains
# given each error rate, true tree, no of cells and mutations
initial_points = [ # (mcmc_seed, init_tree_type, init_tree_seed)
    (42, 'r', 45),
    (12, 'r', 20),
    (34, 'r', 31),
    (79, 'r', 89),
]

# MCMC config
n_samples = 10000 # <-- configure number of samples here

# Burnin
n_burnin = 5000 # <-- configure burnin here

#####################
#####################


def make_all_mark02():
    """Make all final output file names."""

    #f"{DATADIR}/{experiment}/plots/{McmcConfig}/{CellSimulationId}/"

    # "AD_hist.svg" and "MP3_hist.svg"

    filepaths = []
    filepath = f'{DATADIR}/{experiment}/plots/'
    # add +1 to n_mutation to account for the root mutation
    n_nodes = [n_mutation + 1 for n_mutation in n_mutations]

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
    """Make multiple chain histograms
    
    Output:
        Histograms for each metric, for each MCMC config, for each cell simulation, for each true tree, combining
        multiple chains of different starting points.
        
        The histograms are saved in the following directory structure:
        
        '{DATADIR}/mark02/plots/{mcmc_config_id}/{mutation_data_id}/'
                                   'T_{base_tree_type}_{n_nodes,}_{base_tree_seed}/{metric}.svg',
                                   
        where:
            - mcmc_config_id: the id of the MCMC config
                - as MC_{mcmc_fpr}_{mcmc_fnr}_{mcmc_n_samples}_{burn_in}_{thinning}-{move_prob_config_id}
                where
                    - mcmc_fpr: the false positive rate of the MCMC config
                    - mcmc_fnr: the false negative rate of the MCMC config
                    - mcmc_n_samples: the number of samples of the MCMC config
                    - burn_in: the number of burnin samples of the MCMC config
                    - thinning: the thinning of the MCMC config
                    - move_prob_config_id: the id of the move probability config
                        - as MPC_{prune_and_reattach}_{swap_node_labels}_{swap_subtrees}
                        where
                            - prune_and_reattach: the probability of the prune and reattach move
                            - swap_node_labels: the probability of the swap node labels move
                            - swap_subtrees: the probability of the swap subtrees move
            - mutation_data_id: the id of the cell simulation
                - as CS_{CS_seed}-{true_tree_id}-{n_cells}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}
                where
                    - CS_seed: the seed of the cell simulation
                    - true_tree_id: the id of the true tree
                        - as T_{base_tree_type}_{n_nodes}_{base_tree_seed}
                        where
                            - base_tree_type: the type of the true tree
                            - n_nodes: the number of nodes in the true tree
                            - base_tree_seed: the seed of the true tree
                    - n_cells: the number of cells in the cell simulation
                    - CS_fpr: the false positive rate of the cell simulation
                    - CS_fnr: the false negative rate of the cell simulation
                    - CS_na: the NA rate of the cell simulation
                    - observe_homozygous: whether to observe homozygous mutations in the cell simulation
                    - cell_attachment_strategy: the cell attachment strategy of the cell simulation
            - base_tree_type: the type of the true tree
            - n_nodes: the number of nodes in the true tree
            - base_tree_seed: the seed of the true tree
            - metric: the metric used to calculate the distance / log probability
    
    """
    input:
        make_all_mark02()


rule combined_chain_histogram:
    """Make combined chain histogram for a given metric
    
    Takes the output of analyze_metric rule as input, i.e. the distances for a given metric
    and combines them into a single histogram, with different colors for each chain.
    Up to 6 different chains are colored uniquely.
    
    Requires `n_burnin` to be set. The first `n_burnin` samples are discarded.
    """
    input:
        # calls analyze_metric rule
        all_chain_metrics = ['{DATADIR}/mark02/analysis/MCMC_' + str(mcmc_seed) + '-{mutation_data_id}-iT_'
                             + str(init_tree_type)+ '_{n_nodes,\d+}_' + str(init_tree_seed) +
                             '-{mcmc_config_id}/T_{base_tree_type}_{n_nodes,\d+}_{base_tree_seed,\d+}/{metric}.json'
                             for mcmc_seed, init_tree_type, init_tree_seed in initial_points]

    output:
        combined_chain_histogram = '{DATADIR}/mark02/plots/{mcmc_config_id}/{mutation_data_id}/'
                                   'T_{base_tree_type}_{n_nodes,\d+}_{base_tree_seed,\d+}/{metric}.svg',

    run:
        # load the data
        # for each metric/chain, load the json file
        distances_chains = []

        for each_chain_metric in input.all_chain_metrics:
            # load the distances
            _ , distances = yg.serialize.read_metric_result(Path(each_chain_metric))
            # discard the n_burnin samples from the beginning
            distances = distances[n_burnin:]
            # append to the list
            distances_chains.append(distances)

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Define the list of colors to repeat
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']

        # Generate labels for sub-histograms
        labels = [f'MCMC {i + 1}' for i in range(len(distances_chains))]

        # Iterate over each sublist of distances
        for i, sublist in enumerate(distances_chains):
            # Calculate the index of the color in the predefined list
            color_index = i % len(colors)

            # ensure that the sublist is a list of floats
            sublist = [float(x) for x in sublist]

            # Create histogram for the sublist with the color and label
            ax.hist(sublist,bins='auto', range = (0,1),alpha=0.5,color=colors[color_index],label=labels[i])

        # Set labels and title
        ax.set_xlabel(f"Similarity: {wildcards.metric}")
        ax.set_ylabel('Frequency')

        # Add a legend
        ax.legend()

        # save the histogram
        fig.savefig(Path(output.combined_chain_histogram))



rule mark02_long:
    """To validate the results of mark02, run it for a long time
      100000 i.e. 10x longer than the default
      
      Conditions: 
        - mutations: 50
        - cells: 200, 1000
        - distance: MP3
        - noise: typical      
      """
    input:
        "../data/mark02/plots/MC_1e-06_0.1_100000_0_1-MPC_0.1_0.65_0.25/CS_42-T_r_6_42-200_1e-06_0.1_0.0_f_UXR/T_r_6_42/MP3.svg",
        "../data/mark02/plots/MC_1e-06_0.1_100000_0_1-MPC_0.1_0.65_0.25/CS_42-T_r_6_42-1000_1e-06_0.1_0.0_f_UXR/T_r_6_42/MP3.svg",
        "../data/mark02/plots/MC_1e-06_0.1_100000_0_1-MPC_0.1_0.65_0.25/CS_42-T_r_31_42-1000_1e-06_0.1_0.0_f_UXR/T_r_31_42/MP3.svg",

