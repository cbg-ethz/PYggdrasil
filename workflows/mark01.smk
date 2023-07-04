"""Experiment mark01

 assessing the HUNTRESS trees with distance
 metrics under the SCITE generative model"""

# imports
import jax.numpy as jnp
import pyggdrasil as yg

from pyggdrasil.tree_inference import CellSimulationId, TreeType, TreeId

#####################
experiment="mark01"

# Metrics: Distances / Similarity Measure to use
metrics = ["MP3"]  # also AD <-- configure distances here

#####################
# Cell Simulation Parameters
num_samples = 200 # <-- configure number of samples here

# Errors <--- set the error rates here
errors = {"ideal" : {"fpr": 1e-6, "fnr": 1e-6},
         "typical" : {"fpr": 1e-6, "fnr": 0.1},
         "large" : {"fpr": 0.1, "fnr": 0.1},
         "extreme" : {"fpr": 0.3, "fnr": 0.3}
         }
n_mutations = [5, 10, 30, 50] # <-- configure number of mutations here
n_cells = [200, 1000, 5000] # <-- configure number of cells here

# Homozygous mutations [f: False / t: True]
observe_homozygous = "f" # <-- configure whether to observe homozygous mutations here

# cell attachment strategy [UXR: Uniform Exclude Root / UIR: Uniform Include Root]
cell_attachment_strategy = "UXR" # <-- configure cell attachment strategy here

#####################
# True Tree Parameters
tree_types = ["r"] # <-- configure tree type here ["r","s","d"]
tree_seeds = [42, 34] # <-- configure tree seed here

#####################
# Auxiliary variables
CS_seeds =  jnp.arange(num_samples)

#####################

def make_all_mark01()->list[str]:
    """Make all final output file names."""
    filepaths = []
    filepath = f"{WORKDIR}/{experiment}/plots/CS_XX-"
    # add +1 to n_mutation to account for the root mutation
    n_nodes = [n_mutation+1 for n_mutation in n_mutations]

    # make tree ids
    tree_id_ls = []
    for tree_type in tree_types:
        for tree_seed in tree_seeds:
            for n_node in n_nodes:
                tree_id_ls.append(TreeId(tree_type=TreeType(tree_type), n_nodes=n_node, seed=tree_seed))

    for tree_id in tree_id_ls:
        for n_cell in n_cells:
                for error_name, error in errors.items():
                    for metric in metrics:
                        filepaths.append(filepath+f"{tree_id}-{n_cell}_{error['fpr']}_{error['fnr']}_0.0_{observe_homozygous}_{cell_attachment_strategy}/{metric}_hist.svg")
    return filepaths

rule mark01:
    """Make the distance histograms for each metric."""
    input:
        make_all_mark01()


rule make_histograms:
    """Make the distance histograms for each metric."""
    input:
        distances = "{WORKDIR}/{experiment}/distances/CS_XX-{true_tree_id}-{n_cells,\d+}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}/{metric}.json"
    output:
        hist = "{WORKDIR}/{experiment}/plots/CS_XX-{true_tree_id}-{n_cells,\d+}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}/{metric}_hist.svg"
    run:
        import pyggdrasil as yg
        from pathlib import Path
        import matplotlib.pyplot as plt
        # load the distances
        tree_ids, distances =  yg.serialize.read_metric_result(Path(input.distances))
        # make the histogram
        fig, axs = plt.subplots(1,1,tight_layout=True)
        # We can set the number of bins with the *bins* keyword argument.
        axs.hist(distances, bins='auto')
        # set the axis labels
        axs.set_xlabel(f"Distance/Similarity: {wildcards.metric}")
        axs.set_ylabel("Frequency")
        # save the histogram
        fig.savefig(Path(output.hist))



rule calculate_huntress_distances:
    """Calculate the distances between the true tree and the HUNTRESS trees."""
    input:
        true_tree = "{WORKDIR}/{experiment}/trees/{true_tree_id}.json",
        # TODO (Gordon): make huntress tree id like this
        huntrees_trees = ["{WORKDIR}/{experiment}/huntress/HUN-CS_"+ str(CS_seed) +"-{true_tree_id}-{n_cells}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}.json" for CS_seed in CS_seeds],
    output:
        distances = "{WORKDIR}/{experiment}/distances/CS_XX-{true_tree_id}-{n_cells,\d+}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}/{metric}.json"
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


# below rule input will trigger gen_cell_simulation rule, which will trigger tree generation rule
rule run_huntress:
    """Run HUNTRESS on the true tree.
    
    - Cell Simulation data requires
        - no missing entries
        - no homozygous mutations
    """
    # TODO (Gordon): this rule should be general enough to be used for all experiments, i.e. it should be moved to the common workflow
    input:
        mutation_data = "{WORKDIR}/{experiment}/mutations/{mutation_data_id}.json",
    output:
        huntrees_tree = "{WORKDIR}/{experiment}/huntress/HUN-{mutation_data_id}.json"
    run:
        import pyggdrasil as yg
        # load data of mutation matrix
        with open(input.mutation_data,"r") as f:
            cell_simulation_data = json.load(f)
        # TODO (Gordon): modify to allow non-simulated data
        cell_simulation_data = yg.tree_inference.get_simulation_data(cell_simulation_data)
        # get the mutation matrix
        mut_mat = cell_simulation_data["noisy_mutation_mat"]
        # get error rates from the cell simulation id
        # get name of file without extension
        data_fn = Path(input.mutation_data).stem
        # try to match the cell simulation id
        cell_sim_id = yg.tree_inference.CellSimulationId.from_str(data_fn)
        # run huntress
        huntress_tree = yg.tree_inference.huntress_tree_inference(mut_mat, cell_sim_id.fpr, cell_sim_id.fnr)
        # make TreeNode from Node
        huntress_treeNode = yg.TreeNode.convert_anytree_to_treenode(huntress_tree)
        # save the huntress tree
        yg.serialize.save_tree_node(huntress_treeNode, Path(output.huntrees_tree))

