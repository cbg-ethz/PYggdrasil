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
        mcmc_samples = '{DATADIR}/{experiment}/mcmc/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}.json',
        base_tree = '{DATADIR}/{experiment}/trees/{base_tree_id}.json'
    wildcard_constraints:
        mcmc_config_id = "MC.*",
    output:
        result = '{DATADIR}/{experiment}/analysis/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}/{base_tree_id}/{metric}.json',
        log = '{DATADIR}/{experiment}/analysis/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}/{base_tree_id}/{metric}.log'
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
        mcmc_samples='{DATADIR}/{experiment}/mcmc/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}.json',
    output:
        result = '{DATADIR}/{experiment}/analysis/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}/log_prob.json',
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


rule true_trees_found:
    """Make human readable summary of whether true trees were found and in which iteration."""
    input:
        metric_result = '{DATADIR}/{experiment}/analysis/MCMC_{mcmc_seed,\d+}-CS_{CS_seed,\d+}-{true_tree_id}-{n_cells,\d+}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}-i{init_tree_id}-{mcmc_config_id}/{true_tree_id}/TrueTree.json',
    output:
        result = '{DATADIR}/{experiment}/analysis/MCMC_{mcmc_seed,\d+}-CS_{CS_seed,\d+}-{true_tree_id}-{n_cells,\d+}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}-i{init_tree_id}-{mcmc_config_id}/true_trees_found.txt',
    run:
        # make paths
        fp = Path(input.metric_result)
        fp_out = Path(output.result)
        # load data
        iteration, result = yg.serialize.read_metric_result(fp)
        # cast result str to bool
        result = [bool(i) for i in result]
        # get indices of results that are true
        true_indices = [i for i, x in enumerate(result) if x]
        # get iterations of true trees
        true_iterations = [iteration[i] for i in true_indices]
        # write to file
        with open(fp_out, 'w') as f:
            f.write(f"True trees found in iterations:")
            for i in true_iterations:
                f.write(f"\n{i}")
