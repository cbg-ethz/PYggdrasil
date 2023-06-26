"""Experiment mark01

 assessing the HUNTRESS trees with distance
 metrics under the SCITE generative model"""

# imports
import pyggdrasil as yg

from pyggdrasil.tree_inference import CellSimulationId

#####################
# Environment variables
WORKDIR = "../data"

#####################
experiment="mark01"

# Metrics: Distances / Similarity Measure to use
metrics = ["MP3","AD"]

#####################
# Cell Simulation Parameters
CS_seeds = [1,2,3,4,5,6,7,8,9,10] # should be 200

#####################
# Auxiliary variables
true_tree_id = "tbd"
# even if not needed make cell simulation id to check type and pydatic object

#####################

rule all:
    """Make the distance histograms for each metric."""
    input:
        expand(f"{WORKDIR}/{experiment}/plots/{{metric}}_hist.svg", metric=metrics)


rule make_histograms:
    """Make the distance histograms for each metric."""
    input:
        "{WORKDIR}/{experiment}/distances/CS_XX-{true_tree_id}-{n_cells,\d+}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}/{metric}.json"
    output:
        "{WORKDIR}/{experiment}/plots/CS_XX-{true_tree_id}-{n_cells,\d+}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}/{metric}_hist.svg"
    run:
        return NotImplementedError


rule calculate_huntress_distances:
    """Calculate the distances between the true tree and the HUNTRESS trees."""
    input:
        true_tree = "{WORKDIR}/{experiment}/trees/{true_tree_id}.json",
        # TODO (Gordon): make huntress tree id like this
        mutation_data = expand(
            f"{WORKDIR}/{experiment}/huntress/h-CS_{{CS_seed}}-{true_tree_id}-{n_cells}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}.json",
            CS_seed=CS_seeds
        )
    output:
        "{WORKDIR}/{experiment}/distances/CS_XX-{true_tree_id}-{n_cells,\d+}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}/{metric}.json"
    run:
        return NotImplementedError


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

