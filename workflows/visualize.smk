"""Snakemake file defining the workflows for plotting"""

import json
import pyggdrasil as yg
import numpy as np
from pathlib import Path


rule plot_log_prob:
    """Plot the log-probability over iterations of an mcmc run"""
    input:
        log_prob="{DATADIR}/{experiment}/analysis/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}/log_prob.json",
    output:
        plot="{DATADIR}/{experiment}/plots/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}/log_prob.svg",
    run:
        in_fp = Path(input.log_prob)
        with open(in_fp) as f:
            data = json.load(f)
        out_fp = Path(output.plot)
        yg.visualize.save_log_p_iteration(data["iteration"], data["result"], out_fp)


rule plot_metrics:
    """Plot a metric over iterations of an mcmc run"""
    input:
        metric="{DATADIR}/{experiment}/analysis/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}/{base_tree_id}/{metric}.json",
    output:
        plot="{DATADIR}/{experiment}/plots/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}/{base_tree_id}/{metric}.svg",
    run:
        in_fp = Path(input.metric)
        with open(in_fp) as f:
            data = json.load(f)
        out_fp = Path(output.plot)
        yg.visualize.save_metric_iteration(
            data["iteration"],
            data["result"],
            metric_name=wildcards.metric,
            out_fp=out_fp,
        )


rule plot_initial_tree:
    """Plot the initial tree"""
    input:
        mcmc_data="{DATADIR}/{experiment}/mcmc/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}.json",
        initial_tree="{DATADIR}/{experiment}/trees/{init_tree_id}.json",
    wildcard_constraints:
        mcmc_config_id="MC.*",
        init_tree_id="T.*",
    output:
        plot="{DATADIR}/{experiment}/plots/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}/init_tree.svg",
    run:
        in_fp = Path(input.initial_tree)
        out_fp = Path(output.plot)
        # read in tree
        initial_tree = yg.serialize.read_tree_node(in_fp)
        # plot the tree
        yg.visualize.plot_tree_no_print(
            initial_tree, save_name=out_fp.name.__str__(), save_dir=out_fp.parent
        )


rule plot_top_three_trees:
    """Plot the top three trees from an mcmc run"""
    input:
        mcmc_data="{DATADIR}/{experiment}/mcmc/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}.json",
    output:
        plot1="{DATADIR}/{experiment}/plots/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}/top_tree_1.svg",
        plot2="{DATADIR}/{experiment}/plots/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}/top_tree_2.svg",
        plot3="{DATADIR}/{experiment}/plots/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}/top_tree_3.svg",
        info="{DATADIR}/{experiment}/plots/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}/top_tree_info.json",
    run:
        in_fp = Path(input.mcmc_data)
        out_fp = Path(output.plot1)
        # read in the mcmc data
        data = yg.serialize.read_mcmc_samples(in_fp)
        mcmc_samples = yg.analyze.to_pure_mcmc_data(data)
        # get the top three trees, by log-probability
        yg.visualize.save_top_trees_plots(mcmc_samples, out_fp.parent)


rule plot_true_tree:
    """Plot the true tree."""
    input:
        true_tree="{DATADIR}/{experiment}/trees/{true_tree_id}.json",
        mcmc_data="{DATADIR}/{experiment}/mcmc/MCMC_{mcmc_seed,\d+}-CS_{CS_seed,\d+}-{true_tree_id}-{n_cells,\d+}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}-i{init_tree_id}-{mcmc_config_id}.json",
    output:
        plot="{DATADIR}/{experiment}/plots/MCMC_{mcmc_seed,\d+}-CS_{CS_seed,\d+}-{true_tree_id}-{n_cells,\d+}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}-i{init_tree_id}-{mcmc_config_id}/true_tree.svg",
    run:
        in_fp = Path(input.true_tree)
        out_fp = Path(output.plot)
        # read in tree
        true_tree = yg.serialize.read_tree_node(in_fp)
        # plot the tree
        yg.visualize.plot_tree_no_print(
            true_tree, save_name=out_fp.name.__str__(), save_dir=out_fp.parent
        )


rule plot_tree:
    """Plot a raw tree"""
    input:
        tree="{DATADIR}/{experiment}/trees/{tree_id}.json",
    wildcard_constraints:
        tree_id = "(HUN|T)_(?:(?!/).)+" # allowing both generated and huntress trees
    output:
        plot="{DATADIR}/{experiment}/plots/{tree_id}.svg",
    run:
        in_fp = Path(input.tree)
        out_fp = Path(output.plot)
        # read in tree
        true_tree = yg.serialize.read_tree_node(in_fp)
        # plot the tree
        yg.visualize.plot_tree_no_print(
            true_tree, save_name=out_fp.name.__str__(), save_dir=out_fp.parent
        )


rule plot_tree_relabeled:
    """Plot a raw tree, but relabel the integer node labels to count from 1"""
    input:
        tree="{DATADIR}/{experiment}/trees/{tree_id}.json",
    output:
        plot="{DATADIR}/{experiment}/plots/{tree_id}_relabeled.svg",
    run:
        import pyggdrasil as yg
        in_fp = Path(input.tree)
        out_fp = Path(output.plot)
        # read in tree
        true_tree = yg.serialize.read_tree_node(in_fp)
        # relabel the tree
        old_labels = yg.tree_inference.Tree.tree_from_tree_node(true_tree).labels.tolist()
        new_labels = (x + 1 for x in old_labels)
        mapping_dict = dict(zip(old_labels,new_labels))
        # set print options
        print_options = dict()
        print_options["title"] = False
        print_options["data_tree"] = dict()
        # plot the tree
        yg.visualize.plot_tree(
            true_tree, save_name=out_fp.name.__str__(), save_dir=out_fp.parent, print_options=print_options, rename_labels=mapping_dict
        )
