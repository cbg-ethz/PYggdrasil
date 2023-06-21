"""Workflow for analyzing MCMC runs."""

import json
import logging

import pyggdrasil as yg

from pathlib import Path

from pyggdrasil.tree_inference import McmcConfig, MoveProbConfig, MoveProbConfigOptions, McmcConfigOptions



rule analyze_metric:
    """Analyze MCMC run with a metric taking a base tree and
     an MCMC sample as input i.e. all distances /similarity metrics.
     Note: this includes ``is_true_tree``."""
    input:
        mcmc_samples = '../data/{experiment}/mcmc/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}.json',
        base_tree = '../data/{experiment}/trees/{base_tree_id}.json'

    output:
        result = '../data/{experiment}/analysis/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}/{base_tree_id}/{metric}.json',
        log = '../data/{experiment}/analysis/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}/{base_tree_id}/{metric}.log'
    run:
        # Set up logging
        # if logfile does not exist, create it
        log_path = Path(output.log)
        log_path.unlink(missing_ok=True)
        log_path.touch()
        logging.basicConfig(filename=log_path,level=logging.INFO)

        # load the data
        mcmc_samples = yg.serialize.read_mcmc_samples(Path(input.mcmc_samples))
        mcmc_data = yg.analyze.to_pure_mcmc_data(mcmc_samples)
        logging.info(f"Loaded MCMC samples")
        # load the tree
        base_tree = yg.serialize.read_tree_node(Path(input.base_tree))
        logging.info(f"Loaded base tree: {input.base_tree}")
        # define metric
        metric = yg.analyze.Metrics.get(wildcards.metric)
        logging.info(f"Loaded metric: {metric}")
        # compute the metric
        iteration, result = yg.analyze.analyze_mcmc_run(mcmc_data, metric, base_tree)
        # write the result
        fp = Path(output.result)
        yg.serialize.save_metric_result(iteration, result, fp)
        logging.info(f"Saved result to {fp}")


rule get_log_probs:
    """Extract log probabilities from MCMC samples for ease of plotting / analysis."""
    input:
        mcmc_samples='../data/{experiment}/mcmc/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}.json',
    output:
        result = '../data/{experiment}/analysis/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}/log_prob.json',
    run:
        # load the data
        mcmc_samples = yg.serialize.read_mcmc_samples(Path(input.mcmc_samples))
        mcmc_data = yg.analyze.to_pure_mcmc_data(mcmc_samples)

        # write the result
        fp = Path(output.result)

        # get log probs
        log_probs = mcmc_data.log_probabilities
        # convert log probs to list of float
        log_probs = [float(i) for i in log_probs]
        # get iteration
        iteration = mcmc_data.iterations
        # convert iterations of list of int
        iteration = [int(i) for i in iteration]

        # save
        yg.serialize.save_metric_result(iteration, log_probs, fp)