"""Workflow for analyzing MCMC runs."""

import logging

import pyggdrasil as yg

from pathlib import Path


def get_mem_mb(wildcards, attempt):
    """Get adaptive memory in MB for a given rule."""
    return attempt * 2000

def get_mem_mb_large(wildcards, attempt):
    """Get adaptive memory in MB for a given rule."""
    return attempt * 8000


rule analyze_metric:
    """Analyze MCMC run with a metric taking a base tree and
     an MCMC sample as input i.e. all distances /similarity metrics.
     Note: this includes ``is_true_tree``."""
    input:
        mcmc_samples = '{DATADIR}/{experiment}/mcmc/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}.json',
        base_tree = '{DATADIR}/{experiment}/trees/{base_tree_id}.json'
    wildcard_constraints:
        mcmc_config_id = "MC.*",
        # metric wildcard cannot be log_prob
        metric=r"(?!(log_prob))\w+",
    resources:
        mem_mb=get_mem_mb,
    output:
        result="{DATADIR}/{experiment}/analysis/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}/{base_tree_id}/{metric}.json",
        log="{DATADIR}/{experiment}/analysis/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}/{base_tree_id}/{metric}.log",
    run:
        # set up logging
        log_path = Path(output.log)
        log_path.unlink(missing_ok=True)
        log_path.touch()
        logging.basicConfig(filename=log_path, level=logging.INFO)

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
        mcmc_samples="{DATADIR}/{experiment}/mcmc/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}.json",
    wildcard_constraints:
        mcmc_config_id = "MC_(?:(?!/).)+",
        init_tree_id = "(HUN|T)_(?:(?!/).)+"

    output:
        result="{DATADIR}/{experiment}/analysis/MCMC_{mcmc_seed,\d+}-{mutation_data_id}-i{init_tree_id}-{mcmc_config_id}/log_prob.json",
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
        metric_result="{DATADIR}/{experiment}/analysis/MCMC_{mcmc_seed,\d+}-CS_{CS_seed,\d+}-{true_tree_id}-{n_cells,\d+}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}-i{init_tree_id}-{mcmc_config_id}/{true_tree_id}/TrueTree.json",
    output:
        result="{DATADIR}/{experiment}/analysis/MCMC_{mcmc_seed,\d+}-CS_{CS_seed,\d+}-{true_tree_id}-{n_cells,\d+}_{CS_fpr}_{CS_fnr}_{CS_na}_{observe_homozygous}_{cell_attachment_strategy}-i{init_tree_id}-{mcmc_config_id}/true_trees_found.txt",
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
        with open(fp_out, "w") as f:
            f.write(f"True trees found in iterations:")
            for i in true_iterations:
                f.write(f"\n{i}")


rule calculate_rhats_4chains:
    """Calculate Rhat for 4 chains, each with a different init tree and seed but same data and config."""
    input:
        mcmc_metric_samples1="{DATADIR}/{experiment}/analysis/MCMC_{mcmc_seed1,\d+}-{mutation_data_id}-i{init_tree_id1}-{mcmc_config_id}/{base_tree_id}/{metric}.json",
        mcmc_metric_samples2="{DATADIR}/{experiment}/analysis/MCMC_{mcmc_seed2,\d+}-{mutation_data_id}-i{init_tree_id2}-{mcmc_config_id}/{base_tree_id}/{metric}.json",
        mcmc_metric_samples3="{DATADIR}/{experiment}/analysis/MCMC_{mcmc_seed3,\d+}-{mutation_data_id}-i{init_tree_id3}-{mcmc_config_id}/{base_tree_id}/{metric}.json",
        mcmc_metric_samples4="{DATADIR}/{experiment}/analysis/MCMC_{mcmc_seed4,\d+}-{mutation_data_id}-i{init_tree_id4}-{mcmc_config_id}/{base_tree_id}/{metric}.json",
    wildcard_constraints:
        mutation_data_id = "CS.*",
        mcmc_config_id= "MC_(?:(?!/).)+",
    resources:
        mem_mb=get_mem_mb,
    output:
        result="{DATADIR}/{experiment}/analysis/rhat/{base_tree_id}/{metric}/rhat4-MCMCseeds_s{mcmc_seed1}_s{mcmc_seed2}_s{mcmc_seed3}_s{mcmc_seed4}-{mutation_data_id}-iTrees_i{init_tree_id1}_i{init_tree_id2}_i{init_tree_id3}_i{init_tree_id4}-{mcmc_config_id}/rhat.json",
    run:
        import json
        import numpy as np
        # load the data
        # chain 1
        # load data
        fp = Path(input.mcmc_metric_samples1)
        iteration, result1 = yg.serialize.read_metric_result(fp)
        result1 = np.array(result1)
        # chain 2
        fp = Path(input.mcmc_metric_samples2)
        _, result2 = yg.serialize.read_metric_result(fp)
        result2 = np.array(result2)
        # chain 3
        fp = Path(input.mcmc_metric_samples3)
        _, result3 = yg.serialize.read_metric_result(fp)
        result3 = np.array(result3)
        # chain 4
        fp = Path(input.mcmc_metric_samples4)
        _, result4 = yg.serialize.read_metric_result(fp)
        result4 = np.array(result4)

        # calculate rhat - returns the 4-length array of rhats
        chains = np.array([result1, result2, result3, result4])
        rhat = yg.analyze.rhats(chains)

        # write the result
        fp = Path(output.result)
        # save with iteration numbers, struncate the first 3 iterations
        iteration = iteration[3:]
        yg.serialize.save_metric_result(iteration, list(rhat), fp)

