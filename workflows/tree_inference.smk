"""Snakemake rules for the tree inference pipeline."""

import json
import shutil
import jax

from pathlib import Path

import pyggdrasil as yg


from pyggdrasil.tree_inference import (
    McmcConfig,
    MoveProbConfig,
    MoveProbConfigOptions,
    McmcConfigOptions,
)


###############################################
## Relative path from DATADIR to the repo root

REPODIR = "/cluster/work/bewi/members/gkoehn/repos/PYggdrasil"
#REPODIR = ".."

###############################################


rule make_random_or_deep_tree:
    """Make a tree (TreeNode) of random/deep topology and save it as JSON."""
    input:
        # TODO (Gordon): make this flexible - repo/PYggdrasil is not always the working directory
        script=REPODIR + "/scripts/make_tree.py",
    wildcard_constraints:
        tree_type="(r|d)",
    output:
        tree="{DATADIR}/{experiment}/trees/T_{tree_type}_{n_nodes,\d+}_{tree_seed,\d+}.json",
    shell:
        """
        python {input.script} \
            --out_dir {DATADIR}/{wildcards.experiment}/trees \
            --seed {wildcards.tree_seed} \
            --n_nodes {wildcards.n_nodes} \
            --tree_type {wildcards.tree_type}
        """


rule make_star_tree:
    """"Make a tree (TreeNode) of star topology and save it as JSON."""
    input:
        script=REPODIR + "/scripts/make_tree.py",
    output:
        tree="{DATADIR}/{experiment}/trees/T_s_{n_nodes,\d+}.json",
    shell:
        """
        python {input.script} \
            --out_dir {DATADIR}/{wildcards.experiment}/trees \
            --n_nodes {wildcards.n_nodes} \
            --tree_type s
        """


rule gen_cell_simulation:
    """Generate a mutation matrix given a true tree and save to JSON."""
    input:
        script=REPODIR + "/scripts/cell_simulation.py",
        true_tree="{DATADIR}/{experiment}/trees/{true_tree_id}.json",
    output:
        mutation_data="{DATADIR}/{experiment}/mutations/CS_{CS_seed,\d+}-{true_tree_id}-{n_cells,\d+}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}.json",
    shell:
        """
        python {input.script} \
        --seed {wildcards.CS_seed} \
        --true_tree_fp {input.true_tree} \
        --n_cells {wildcards.n_cells} \
        --fpr {wildcards.CS_fpr} \
        --fnr {wildcards.CS_fnr} \
        --na_rate {wildcards.CS_na} \
        --observe_homozygous {wildcards.observe_homozygous} \
        --strategy {wildcards.cell_attachment_strategy} \
        --out_dir {DATADIR}/{wildcards.experiment}/mutations \
        """


rule make_mcmc_move_prob_config:
    """Make MCMC move probability config."""
    params:
        prune_and_reattach="{prune_and_reattach}",
        swap_node_labels="{swap_node_labels}",
        swap_subtrees="{swap_subtrees}",
    wildcard_constraints:
        prune_and_reattach="\d+\.\d+",
        swap_node_labels="\d+\.\d+",
        swap_subtrees="\d+\.\d+",
    output:
        mcmc_move_prob_config="{DATADIR}/{experiment}/mcmc/config/MPC_{prune_and_reattach}_{swap_node_labels}_{swap_subtrees}.json",
    run:
        # define the move probabilities manually
        custom_MP_conf = MoveProbConfig(
            prune_and_reattach=float(params.prune_and_reattach),
            swap_node_labels=float(params.swap_node_labels),
            swap_subtrees=float(params.swap_subtrees),
        )

        with open(output.mcmc_move_prob_config, "w") as f:
            json.dump(custom_MP_conf.dict(), f)


rule make_mcmc_config:
    """Make MCMC config."""
    params:
        fpr="{mcmc_fpr}",
        fnr="{mcmc_fnr}",
        n_samples="{mcmc_n_samples}",
        burn_in="{burn_in}",
        thinning="{thinning}",
    wildcard_constraints:
        move_prob_config_id="MPC.*",
        thinning="\d+",
        burn_in="\d+",
    input:
        mcmc_move_prob_config="{DATADIR}/{experiment}/mcmc/config/{move_prob_config_id}.json",
    output:
        mcmc_config="{DATADIR}/{experiment}/mcmc/config/MC_{mcmc_fpr}_{mcmc_fnr}_{mcmc_n_samples}_{burn_in}_{thinning}-{move_prob_config_id}.json",
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
            move_probs=move_prob_config,
        )

        with open(output.mcmc_config, "w") as f:
            json.dump(customMcmcConfig.dict(), f)


rule mcmc:
    """"Run MCMC"""
    input:
        script=REPODIR + "/scripts/run_mcmc.py",
        mutation_data="{DATADIR}/{experiment}/mutations/{mutation_data_id}.json",
        init_tree="{DATADIR}/{experiment}/trees/{init_tree_id}.json",
        mcmc_config="{DATADIR}/{experiment}/mcmc/config/{mcmc_config_id}.json",
    wildcard_constraints:
        mcmc_config_id = "MC.*",
        init_tree_id = "(HUN|T).*" # allowing both generated and huntress trees
    output:
        mcmc_log="{DATADIR}/{experiment}/mcmc/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}.log",
        mcmc_samples="{DATADIR}/{experiment}/mcmc/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}.json",
    shell:
        """
        python {input.script} \
        --seed {wildcards.mcmc_seed} \
        --config_fp {input.mcmc_config} \
        --out_dir {DATADIR}/{wildcards.experiment}/mcmc \
        --data_fp {input.mutation_data} \
        --init_tree_fp {input.init_tree} 
        """


# below rule input will trigger gen_cell_simulation rule, which will trigger tree generation rule
rule run_huntress:
    """Run HUNTRESS on the true tree.
    
    Output is saved in huntress directory, intentionally not in the tree directory.
    HUNTRESS output may vary in the number of mutations from the mutation matrix.

    - Cell Simulation data requires
        - no missing entries
        - no homozygous mutations
    """
    input:
        mutation_data="{DATADIR}/{experiment}/mutations/{mutation_data_id}.json",
    output:
        huntrees_tree="{DATADIR}/{experiment}/huntress/HUN-{mutation_data_id}.json"
    threads: 8
    run:
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
        huntress_tree = yg.tree_inference.huntress_tree_inference(mut_mat,cell_sim_id.fpr,cell_sim_id.fnr, n_threads=threads)
        # make TreeNode from Node
        huntress_treeNode = yg.TreeNode.convert_anytree_to_treenode(huntress_tree)
        # save the huntress tree
        yg.serialize.save_tree_node(huntress_treeNode,Path(output.huntrees_tree))


rule copy_simulated_huntress_r_d_tree:
    """Copy the simulated huntress tree to the tree directory, 
       with information about the number of nodes from the true tree.
       
       Validates that the number of nodes in the tree matches the number of nodes in the true tree.
       """
    input:
        huntrees_tree="{DATADIR}/{experiment}/huntress/HUN-CS_{CS_seed}-T_{tree_type}_{n_nodes}_{tree_seed}-{n_cells}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}.json"
    output:
        huntrees_tree="{DATADIR}/{experiment}/trees/T_h_{n_nodes}_CS_{CS_seed}-T_{tree_type}_{n_nodes}_{tree_seed}-{n_cells}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}.json"
    run:
        # validate the number of nodes in the tree
        init_tree_node = yg.serialize.read_tree_node(Path(input.huntrees_tree))
        # convert TreeNode to Tree
        init_tree = yg.tree_inference.Tree.tree_from_tree_node(init_tree_node)

        # assert that the number of mutations and the data matrix size match
        # no of nodes must equal the number of rows in the data matrix plus root truncated
        if not int(init_tree.labels.shape[0]) == int(wildcards.n_nodes):
            raise ValueError(f"Number of nodes in the tree {init_tree.labels.shape[0]} does not match the number of nodes in the filename {wildcards.n_nodes}, Huntress may have not included all mutations.")

        # copy and rename the file from the huntress tree directory to the tree directory
        shutil.copy(input.huntrees_tree,output.huntrees_tree)


rule copy_simulated_huntress_s_tree:
    """Copy the simulated huntress tree to the tree directory, 
       with information about the number of nodes from the true tree.
       
       Validates that the number of nodes in the tree matches the number of nodes in the true tree.
       """
    input:
        huntrees_tree="{DATADIR}/{experiment}/huntress/HUN-CS_{CS_seed}-T_s_{n_nodes}-{n_cells}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}.json"
    output:
        huntrees_tree="{DATADIR}/{experiment}/trees/T_h_{n_nodes}_CS_{CS_seed}-T_s_{n_nodes}-{n_cells}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}.json"
    run:
        # validate the number of nodes in the tree
        init_tree_node = yg.serialize.read_tree_node(Path(input.huntrees_tree))
        # convert TreeNode to Tree
        init_tree = yg.tree_inference.Tree.tree_from_tree_node(init_tree_node)

        # assert that the number of mutations and the data matrix size match
        # no of nodes must equal the number of rows in the data matrix plus root truncated
        if not int(init_tree.labels.shape[0]) == int(wildcards.n_nodes):
            raise ValueError(f"Number of nodes in the tree {init_tree.labels.shape[0]} does not match the number of nodes in the filename {wildcards.n_nodes}, Huntress may have not included all mutations.")

        # copy and rename the file from the huntress tree directory to the tree directory
        shutil.copy(input.huntrees_tree,output.huntrees_tree)


rule mcmc_evolve_tree:
    """Evolves a tree using mcmc moves from SCITE."""

    input:
        init_tree="{DATADIR}/{experiment}/trees/{init_tree_id}.json",
    output:
        evolved_tree="{DATADIR}/{experiment}/trees/T_m_{n_nodes}_{n_moves}_{mcmc_move_seed}_o{init_tree_id}.json"
    run:
        # load the initial tree
        init_tree_node = yg.serialize.read_tree_node(Path(input.init_tree))
        # get mcmc parameters from the filename
        n_moves = int(wildcards.n_moves)
        mcmc_seed = int(wildcards.mcmc_move_seed)
        rng = jax.random.PRNGKey(mcmc_seed)
        # evolve the tree
        evolved_tree_node = yg.tree_inference.evolve_tree_mcmc(init_tree_node,n_moves,rng)
        # save the evolved tree
        yg.serialize.save_tree_node(evolved_tree_node,Path(output.evolved_tree))
