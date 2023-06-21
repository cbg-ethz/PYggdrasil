"""Snakemake file defining the workflows for plotting"""

import json
import logging
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
