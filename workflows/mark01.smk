"""Experiment mark01

 assessing the HUNTRESS trees with distance
 metrics under the SCITE generative model"""

# imports
import pyggdrasil as yg

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
        expand(f"{WORKDIR}/{experiment}/plots/{{metrics}}_hist.svg", metric=metrics)


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
            "{WORKDIR}/{experiment}/huntress/h-CS_{CS_seed,\d+}-{true_tree_id}-{n_cells,\d+}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}.json",
            CS_seed=CS_seeds
        )
    output:
        "{WORKDIR}/{experiment}/distances/CS_XX-{true_tree_id}-{n_cells,\d+}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}/{metric}.json"
    run:
        return NotImplementedError

# below rule input will trigger gen_cell_simulation rule, which will trigger tree generation rule
rule run_huntress:
    """Run HUNTRESS on the true tree."""
    # TODO (Gordon): this rule should be general enough to be used for all experiments, i.e. it should be moved to the common workflow
    input:
        mutation_data = "{WORKDIR}/{experiment}/mutations/{mutation_data_id}.json",
    output:
        "{WORKDIR}/{experiment}/huntress/{mutation_data_id}.json"
    run:
        return NotImplementedError
