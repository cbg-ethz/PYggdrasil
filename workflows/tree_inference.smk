"""Snakemake rules for the tree inference pipeline."""

import json

from pyggdrasil.tree_inference import McmcConfig, MoveProbConfig, MoveProbConfigOptions, McmcConfigOptions


rule make_random_or_deep_tree:
    """Make a tree (TreeNode) of random/deep topology and save it as JSON."""
    input:
        script="../scripts/make_tree.py"

    wildcard_constraints:
        tree_type = "(r|d)"
    output:
        tree="../data/{experiment}/trees/T_{tree_type}_{n_nodes,\d+}_{tree_seed,\d+}.json"
    shell:
        """
        poetry run python {input.script} \
            --out_dir ../data/{wildcards.experiment}/trees \
            --seed {wildcards.tree_seed} \
            --n_nodes {wildcards.n_nodes} \
            --tree_type {wildcards.tree_type}
        """

rule make_star_tree:
    """"Make a tree (TreeNode) of star topology and save it as JSON."""
    input:
        script="../scripts/make_tree.py"
    output:
        tree="../data/{experiment}/trees/T_s_{n_nodes,\d+}.json"
    shell:
        """
        poetry run python {input.script} \
            --out_dir ../data/{wildcards.experiment}/trees \
            --n_nodes {wildcards.n_nodes} \
            --tree_type s
        """


rule gen_cell_simulation:
    """Generate a mutation matrix given a true tree and save to JSON."""
    input:
        script="../scripts/cell_simulation.py",
        true_tree = "../data/{experiment}/trees/{true_tree_id}.json"
    output:
        mutation_data = "../data/{experiment}/mutations/CS_{CS_seed,\d+}-{true_tree_id}-{n_cells,\d+}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}.json"
    shell:
        """
        poetry run python {input.script} \
        --seed {wildcards.CS_seed} \
        --true_tree_fp {input.true_tree} \
        --n_cells {wildcards.n_cells} \
        --fpr {wildcards.CS_fpr} \
        --fnr {wildcards.CS_fnr} \
        --na_rate {wildcards.CS_na} \
        --observe_homozygous {wildcards.observe_homozygous} \
        --strategy {wildcards.cell_attachment_strategy} \
        --out_dir ../data/{wildcards.experiment}/mutations \
        """


# choose the optimal move probabilities, by enums
#optimal_MP_conf = MoveProbConfigOptions.OPTIMAL


# Set the Model manually
#customMcmcConfig = McmcConfigOptions.TEST.value

rule make_mcmc_move_prob_config:
    """Make MCMC move probability config."""
    params:
        prune_and_reattach = "{prune_and_reattach}",
        swap_node_labels = "{swap_node_labels}",
        swap_subtrees = "{swap_subtrees}"
    wildcard_constraints:
        prune_and_reattach = "\d+\.\d+",
        swap_node_labels = "\d+\.\d+",
        swap_subtrees = "\d+\.\d+",
    output:
        mcmc_move_prob_config = "../data/{experiment}/mcmc/config/MPC_{prune_and_reattach}_{swap_node_labels}_{swap_subtrees}.json",
    run:
        # define the move probabilities manually
        custom_MP_conf = MoveProbConfig(
            prune_and_reattach=float(params.prune_and_reattach),
            swap_node_labels=float(params.swap_node_labels),
            swap_subtrees=float(params.swap_subtrees)
        )

        with open(output.mcmc_move_prob_config, "w") as f:
            json.dump(custom_MP_conf.dict(), f)



rule make_mcmc_config:
    """Make MCMC config."""
    params:
        fpr = "{mcmc_fpr}",
        fnr = "{mcmc_fnr}",
        n_samples = "{mcmc_n_samples}",
        burn_in = "{burn_in}",
        thinning = "{thinning}",
    wildcard_constraints:
        move_prob_config_id = "MPC.*",
        thinning = "\d+",
        burn_in = "\d+",
    input:
        mcmc_move_prob_config = "../data/{experiment}/mcmc/config/{move_prob_config_id}.json",
    output:
        mcmc_config = "../data/{experiment}/mcmc/config/MC_{mcmc_fpr}_{mcmc_fnr}_{mcmc_n_samples}_{burn_in}_{thinning}-{move_prob_config_id}.json",
    run:
        # load the move probabilities from the config file
        with open(input.mcmc_move_prob_config, "r") as f:
            move_prob_config = json.load(f)
        move_prob_config = MoveProbConfig(**move_prob_config)

        # define the move probabilities manually
        customMcmcConfig = McmcConfig(
            fpr=float(params.fpr),
            fnr=float(params.fnr),
            n_samples=int(params.n_samples),
            burn_in=int(params.burn_in),
            thinning=int(params.thinning),
            move_probs=move_prob_config
        )

        with open(output.mcmc_config, "w") as f:
            json.dump(customMcmcConfig.dict(), f)



rule mcmc:
    """"Run MCMC"""
    input:
        script = "../scripts/run_mcmc.py",
        mutation_data= "../data/{experiment}/mutations/{mutation_data_id}.json",
        init_tree = "../data/{experiment}/trees/{init_tree_id}.json",
        mcmc_config = "../data/{experiment}/mcmc/config/{mcmc_config_id}.json",
    wildcard_constraints:
        mcmc_config_id = "MC.*",
        init_tree_id = "T.*"
    output:
        mcmc_log = '../data/{experiment}/mcmc/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}.log',
        mcmc_samples = '../data/{experiment}/mcmc/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}.json',
    shell:
        """
        poetry run python {input.script} \
        --seed {wildcards.mcmc_seed} \
        --config_fp {input.mcmc_config} \
        --out_dir ../data/{wildcards.experiment}/mcmc \
        --data_fp {input.mutation_data} \
        --init_tree_fp {input.init_tree} 
        """