"""Snakemake file defining the workflows for plotting"""

import json
import pyggdrasil as yg
from pathlib import Path

rule plot_log_prob:
    """Plot the log-probability over iterations of an mcmc run"""
    input:
        log_prob = '../data/{experiment}/analysis/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}/log_prob.json',

    output:
        plot = '../data/{experiment}/plots/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}/log_prob.svg',
    run:
        in_fp = Path(input.log_prob)
        with open(in_fp) as f:
            data = json.load(f)
        out_fp = Path(output.plot)
        yg.visualize.save_log_p_iteration(data['iteration'],data['result'], out_fp)


rule plot_metrics:
    """Plot a metric over iterations of an mcmc run"""
    input:
        metric = '../data/{experiment}/analysis/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}/{base_tree_id}/{metric}.json',
    output:
        plot = '../data/{experiment}/plots/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}/{base_tree_id}/{metric}.svg',
    run:
        in_fp = Path(input.metric)
        with open(in_fp) as f:
            data = json.load(f)
        out_fp = Path(output.plot)
        yg.visualize.save_metric_iteration(data['iteration'],data['result'], metric_name=wildcards.metric, out_fp=out_fp)


rule plot_initial_tree:
    """Plot the initial tree"""
    input:
        mcmc_data = '../data/{experiment}/mcmc/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}.json',
        initial_tree = '../data/{experiment}/trees/{init_tree_id}.json',
    wildcard_constraints:
        mcmc_config_id = "MC.*",
        init_tree_id = "T.*",
    output:
        plot = '../data/{experiment}/plots/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}/init_tree.svg',
    run:
        in_fp = Path(input.initial_tree)
        out_fp = Path(output.plot)
        # read in tree
        initial_tree = yg.serialize.read_tree_node(in_fp)
        # plot the tree
        yg.visualize.plot_tree_no_print(initial_tree, save_name=out_fp.name.__str__(), save_dir=out_fp.parent)



rule  plot_top_three_trees:
    """Plot the top three trees from an mcmc run"""
    input:
        mcmc_data = '../data/{experiment}/mcmc/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}.json',
    output:
        plot1='../data/{experiment}/plots/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}/top_tree_1.svg',
        plot2='../data/{experiment}/plots/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}/top_tree_2.svg',
        plot3='../data/{experiment}/plots/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}/top_tree_3.svg',
    run:
        in_fp = Path(input.mcmc_data)
        out_fp = Path(output.plot1)
        # read in the mcmc data
        data = yg.serialize.read_mcmc_samples(in_fp)
        mcmc_samples = yg.analyze.to_pure_mcmc_data(data)
        # get the top three trees, by log-probability
        yg.visualize.save_top_trees_plots(mcmc_samples, out_fp.parent)