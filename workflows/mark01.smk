"""Experiment mark01

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
experiment="mark01"

# Metrics: Distances / Similarity Measure to use
metrics = ["MP3","AD"]  # also AD <-- configure distances here

#####################
# Cell Simulation Parameters
num_samples = 200 #200 # <-- configure number of samples here

# Errors <--- set the error rates here
errors = {
        member.name: member.value.dict()
        for member in yg.tree_inference.ErrorCombinations
}
n_mutations = [5, 10] #, 30, 50] # <-- configure number of mutations here
n_cells = [200] #, 1000] #, 5000] # <-- configure number of cells here

# Homozygous mutations [f: False / t: True]
observe_homozygous = "f" # <-- configure whether to observe homozygous mutations here

# cell attachment strategy [UXR: Uniform Exclude Root / UIR: Uniform Include Root]
cell_attachment_strategy = "UXR" # <-- configure cell attachment strategy here

#####################
# True Tree Parameters
tree_types = ["r"]#, "s"] # <-- configure tree type here ["r","s","d"]
tree_seeds = [42,]# 34] # <-- configure tree seed here

#####################
# Auxiliary variables
CS_seeds =  jnp.arange(num_samples)

#####################

def make_all_mark01()->list[str]:
    """Make all final output file names."""
    filepaths = []
    filepath = f"{DATADIR}/{experiment}/plots/CS_XX-"
    # add +1 to n_mutation to account for the root mutation
    n_nodes = [n_mutation+1 for n_mutation in n_mutations]

    for tree_type in tree_types:
        for tree_seed in tree_seeds:
            for n_node in n_nodes:

                # make true tree id
                if tree_type == "s": # star trees have no seed
                    tree_id = TreeId(tree_type=TreeType(tree_type), n_nodes=n_node)
                else:
                    tree_id = TreeId(tree_type=TreeType(tree_type), n_nodes=n_node, seed=tree_seed)

                for n_cell in n_cells:
                        for error_name, error in errors.items():
                            for metric in metrics:
                                # AD is not defined for star trees - skip this case
                                if tree_type == "s" and metric == "AD":
                                    continue
                                filepaths.append(filepath+f"{tree_id}-{n_cell}_{error['fpr']}_{error['fnr']}_0.0_{observe_homozygous}_{cell_attachment_strategy}/{metric}_hist.svg")

    return filepaths

rule mark01:
    """Make the distance histograms for each metric."""
    input:
        make_all_mark01()


rule make_histograms:
    """Make the distance histograms for each metric."""
    input:
        distances ="{DATADIR}/{experiment}/distances/CS_XX-{true_tree_id}-{n_cells,\d+}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}/{metric}.json"
    output:
        hist ="{DATADIR}/{experiment}/plots/CS_XX-{true_tree_id}-{n_cells,\d+}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}/{metric}_hist.svg"
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
        axs.set_xlabel(f"Distance/Similarity: {wildcards.metric}")
        axs.set_ylabel("Frequency")
        # save the histogram
        fig.savefig(Path(output.hist))



rule calculate_huntress_distances:
    """Calculate the distances between the true tree and the HUNTRESS trees."""
    input:
        true_tree = "{DATADIR}/{experiment}/trees/{true_tree_id}.json",
        # TODO (Gordon): make huntress tree id like this
        huntrees_trees = ["{DATADIR}/{experiment}/huntress/HUN-CS_"+ str(CS_seed) +"-{true_tree_id}-{n_cells}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}.json" for CS_seed in CS_seeds],
    output:
        distances = "{DATADIR}/{experiment}/distances/CS_XX-{true_tree_id}-{n_cells,\d+}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}/{metric}.json"
    run:
        import pyggdrasil as yg
        # load the true tree
        true_tree = yg.serialize.read_tree_node(Path(input.true_tree))
        # load the huntress trees
        huntress_trees = [yg.serialize.read_tree_node(Path(fn)) for fn in input.huntrees_trees]
        # get the tree ids - for backwards identification
        huntress_tree_ids = [Path(fn).stem for fn in input.huntrees_trees]
        # get the metric function
        metric_fn = yg.analyze.Metrics.get(wildcards.metric)
        # calculate the distances and save along with the huntress tree id
        distances = [metric_fn(true_tree, huntress_tree) for huntress_tree in huntress_trees]
        # save the distances and the huntress tree ids
        yg.serialize.save_metric_result(axis=huntress_tree_ids, result=distances, out_fp=Path(output.distances), axis_name="huntress_tree_id")

