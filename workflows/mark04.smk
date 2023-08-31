"""Experiment mark04

 assessing the HUNTRESS trees with distance
 metrics under the SCITE generative model"""

# imports
import jax.numpy as jnp
import pyggdrasil as yg

from pyggdrasil.tree_inference import CellSimulationId, TreeType, TreeId

################################################################################
# Environment variables
DATADIR = "/cluster/work/bewi/members/gkoehn/data"
#DATADIR = "../data"

#####################
experiment="mark04"

# Metrics: Distances / Similarity Measure to use
metrics = ["AD","DL"]  # <-- configure distances here

#####################
# Cell Simulation Parameters
num_samples = 200 # <-- configure number of samples here

# Errors <--- set the error rates here
selected_error_cond = ['IDEAL', 'TYPICAL', 'LARGE']
all_error_cond = {
        member.name: member.value.dict()
        for member in yg.tree_inference.ErrorCombinations
}
errors = {
    name: all_error_cond[name]
    for name in selected_error_cond
}
n_mutations = [5] #, 10, 30, 50] # <-- configure number of mutations here
n_cells = [200] #, 1000] #, 5000] # <-- configure number of cells here

# Homozygous mutations [f: False / t: True]
observe_homozygous = "f" # <-- configure whether to observe homozygous mutations here

# cell attachment strategy [UXR: Uniform Exclude Root / UIR: Uniform Include Root]
cell_attachment_strategy = "UXR" # <-- configure cell attachment strategy here

#####################
# True Tree Parameters
# NOTE: Currently implements no STAR Trees
tree_types = ["r"] #, "s"] # <-- configure tree type here ["r","s","d"]
tree_seeds = jnp.arange(200) # 34] # <-- configure tree seed here

#####################
# Cell Simulation Seed
CS_seeds =  [76]

#####################


def make_all_mark04()->list[str]:
    """Make all final output file names."""
    filepaths = []
    filepath = f"{DATADIR}/{experiment}/plots/"
    # add +1 to n_mutation to account for the root mutation
    n_nodes = [n_mutation+1 for n_mutation in n_mutations]

    for CS_seed in CS_seeds:
        for tree_type in tree_types:
                for n_node in n_nodes:
                    for n_cell in n_cells:
                            for error_name, error in errors.items():
                                for metric in metrics:
                                    # AD is not defined for star trees - skip this case
                                    if tree_type == "s" and metric == "AD":
                                        continue
                                    filepaths.append(filepath+f"CS_{CS_seed}-T_{tree_type}_{n_node}_XX-{n_cell}_{error['fpr']}_{error['fnr']}_0.0_{observe_homozygous}_{cell_attachment_strategy}/{metric}_hist.svg")

    # add combined histogram
    filepath = f"{DATADIR}/{experiment}/plots/combined/"
    # add +1 to n_mutation to account for the root mutation
    n_nodes = [n_mutation + 1 for n_mutation in n_mutations]

    for CS_seed in CS_seeds:
        for tree_type in tree_types:
                for n_node in n_nodes:
                    for n_cell in n_cells:
                            for metric in metrics:
                                # AD is not defined for star trees - skip this case
                                if tree_type == "s" and metric == "AD":
                                    continue
                                filepaths.append(filepath + f"CS_{CS_seed}-T_{tree_type}_{n_node}_XX-{n_cell}_XX_XX_0.0_{observe_homozygous}_{cell_attachment_strategy}/combined_{metric}_hist.svg")

    return filepaths

rule mark04:
    """Make the distance histograms for each metric.
    
    Outputs:
        Histogram Plots as SVGs:
        - {DATADIR}/{experiment}/plots/CS_XX-{true_tree_id}-{n_cells,\d+}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}/{metric}_hist.svg
        where
         experiment = "mark01"
         true_tree_id = T_{tree_type}-{n_nodes}-{seed}
            tree_type = "r" random | "s" star | "d" deep
            n_nodes = number of nodes in the tree
            seed = seed used to generate the tree
         n_cells = number of cells in the tree
         CS_fpr = false positive rate of the cell simulation
         CS_fnr = false negative rate of the cell simulation
         CS_na = noise rate of the cell simulation
         observe_homozygous = whether to observe homozygous mutations
         cell_attachment_strategy = whether to attach cells uniformly including the root or excluding the root
         metric = log probability or similarity measure used to calculate the distance between the true tree and the HUNTRESS trees   
    """
    input:
        make_all_mark04()


rule make_combined_histograms_m4:
    """Make the distance histograms for each metric but all error conditions combined.
    
        Attention: this rule is static.
    """
    input:
        distances = [("{DATADIR}/mark01/distances/CS_{CS_seed}-T_{tree_type}_{n_nodes}_XX-{n_cells,\d+}_" + str(error["fpr"]) + "_" + str(error["fnr"]) + "_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}/{metric}.json") for error in errors.values()],
    output:
        hist ="{DATADIR}/mark04/plots/combined/CS_{CS_seed}-T_{tree_type}_{n_nodes}_XX-{n_cells,\d+}_XX_XX_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}/combined_{metric}_hist.svg"
    run:
        import pyggdrasil as yg
        import numpy as np
        from pathlib import Path
        import matplotlib.pyplot as plt
        # load the distances
        distances = [yg.serialize.read_metric_result(Path(str(fn))) for fn in input.distances]
        distances = np.array(distances)
        # make the histogram
        fig, axs = plt.subplots(1,1,tight_layout=True)
        fig.set_size_inches(7, 4)
        # Define colors for each histogram
        colors = ['g', 'b', 'r', 'purple']
        # make combined histogram for all error conditions
        plot_data = []
        plot_label = []
        for i in range(distances.shape[0]):
            hist_data = distances[i, 1, :]  # Select the 2nd position data for the i-th element
            error_name = list(errors.keys())[i]
            plot_label.append(error_name)
            plot_data.append(np.array(hist_data).astype(float))
        # plot the histogram
        axs.hist(plot_data,range=(0,1),color=colors,label=plot_label, histtype='bar')
        # Put a legend to the right of the current axis
        axs.legend(loc='center left',bbox_to_anchor=(1, 0.5))
        # set the axis labels
        axs.set_xlabel(f"Similarity: {wildcards.metric}")
        axs.set_ylabel("Frequency")
        # have the x-axis go increasing from left to right
        axs.invert_xaxis()
        # ensure proper layout
        fig.tight_layout()
        # save the histogram
        fig.savefig(Path(output.hist))


rule make_histograms_m4:
    """Make the distance histograms for each metric."""
    input:
        distances ="{DATADIR}/{experiment}/distances/CS_{CS_seed}-T_{tree_type}_{n_nodes}_XX-{n_cells,\d+}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}/{metric}.json"
    output:
        hist ="{DATADIR}/{experiment}/plots/CS_{CS_seed}-T_{tree_type}_{n_nodes}_XX-{n_cells,\d+}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}/{metric}_hist.svg"
    run:
        import pyggdrasil as yg
        from pathlib import Path
        import matplotlib.pyplot as plt
        # load the distances
        tree_ids, distances =  yg.serialize.read_metric_result(Path(input.distances))
        # make the histogram
        fig, axs = plt.subplots(1,1,tight_layout=True)
        # We can set the number of bins with the *bins* keyword argument.
        # TODO (Gordon): consider fetching the range from the metric fn in the future
        axs.hist(distances, bins=20, range=(0, 1))
        # set the axis labels
        axs.set_xlabel(f"Similarity: {wildcards.metric}")
        axs.set_ylabel("Frequency")
        # save the histogram
        fig.savefig(Path(output.hist))



rule calculate_huntress_distances_m4:
    """Calculate the distances between the true tree and the HUNTRESS tree for a list of true tree / HUNTRESS tree pairs"""
    input:
        true_trees = ["{DATADIR}/{experiment}/trees/T_{tree_type}_{n_nodes}_"+str(tree_seed)+".json" for tree_seed in tree_seeds],
        # TODO (Gordon): make huntress tree id like this
        huntrees_trees = ["{DATADIR}/{experiment}/huntress/HUN-CS_{CS_seed}-T_{tree_type}_{n_nodes}_"+str(tree_seed)+"-{n_cells}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}.json" for tree_seed in tree_seeds],
    output:
        distances = "{DATADIR}/{experiment}/distances/CS_{CS_seed}-T_{tree_type}_{n_nodes}_XX-{n_cells,\d+}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}/{metric}.json"
    run:
        import pyggdrasil as yg
        # load the true trees
        true_trees = [yg.serialize.read_tree_node(Path(fn)) for fn in input.true_trees]
        # load the huntress trees
        huntress_trees = [yg.serialize.read_tree_node(Path(fn)) for fn in input.huntrees_trees]
        # get the tree ids - for backwards identification
        huntress_tree_ids = [Path(fn).stem for fn in input.huntrees_trees]
        # get the metric function
        metric_fn = yg.analyze.Metrics.get(wildcards.metric)
        # calculate the distances and save along with the huntress tree id
        distances = [metric_fn(true_tree, huntress_tree) for true_tree, huntress_tree in zip(true_trees, huntress_trees)]  # assume both are ordered..
        # save the distances and the huntress tree ids
        yg.serialize.save_metric_result(axis=huntress_tree_ids, result=distances, out_fp=Path(output.distances), axis_name="huntress_tree_id")

