"""Experiment mark05

 assessing the MCMC tree dispersion per tree-tree metric
 as a baseline to understand the tree-tree metric."""

# imports
import jax.random as random
import pyggdrasil as yg
import seaborn as sns

import jax.numpy as jnp

from pathlib import Path

################################################################################
# Environment variables
#DATADIR = "/cluster/work/bewi/members/gkoehn/data"
DATADIR = "../data.nosync"


################################################################################
# Experimental Setup

n_samples = [1000]
n_nodes = [5, 10, 30, 50]

################################################################################

def expand_fp(n_nodes, n_samples) -> list[str]:
    """make the filepaths for the cornerplots"""
    all_files = []
    for nodes in n_nodes:
        for samples in n_samples:
            # mcmc trees
            all_files.append(
                f"{DATADIR}/mark05/plots/AD_DL_mcmc_n{nodes}_samples{samples}.svg"
            )
            # random trees
            all_files.append(
                f"{DATADIR}/mark05/plots/AD_DL_rand_n{nodes}_samples{samples}.svg"
            )

    return all_files


rule mark05:
    """Assessing the MCMC tree dispersion per tree-tree metric"""
    input:
        expand_fp(n_nodes, n_samples)


rule mark05_mcmc:
    """Make MCMC trees and compute tree-tree metrics to plot conern plots"""
    params:
        tree_seed = 43,
        tree_type = yg.tree_inference.TreeType.RANDOM,
        mcmc_seed = 43,

    output:
        cornerplot = f"{DATADIR}/mark05/plots/AD_DL_mcmc_n"+"{nodes}_samples{samples}.svg"
    run:
        # make the initial tree
        init_tree = yg.tree_inference.make_tree(
            int(wildcards.nodes),
            params.tree_type,
            params.tree_seed)

        # now evolve the tree `tree_samples` times
        rng = random.PRNGKey(params.mcmc_seed)
        trees =yg.tree_inference.evolve_tree_mcmc_all(
            init_tree,
            int(wildcards.samples),
            rng)

        # compute the tree-tree metrics
        metric_ad = yg.distances.AncestorDescendantSimilarity().calculate
        ad_values = [metric_ad(init_tree, tree) for tree in trees]  # type: ignore
        metric_dl = yg.distances.DifferentLineageSimilarity().calculate
        dl_values = [metric_dl(init_tree, tree) for tree in trees]  # type: ignore

        # plot corner plot with seaborn
        sns.set_theme(style="ticks")
        g = sns.JointGrid(x=ad_values, y=dl_values, marginal_ticks=True)
        g.plot_joint(sns.scatterplot, s=10, alpha=0.5)
        g.plot_marginals(sns.histplot, kde=True)
        g.set_axis_labels("AD", "DL", fontsize=16)
        g.ax_joint.set_xlim(0, 1)
        g.ax_joint.set_ylim(0, 1)
        g.ax_joint.grid(True)
        g.savefig(output.cornerplot)
        g.savefig(output.cornerplot.replace(".svg", ".png"), dpi=300)


rule mark05_random:
    """Test the corner plot"""
    params:
        tree_type = yg.tree_inference.TreeType.RANDOM,
    output:
        cornerplot = f"{DATADIR}/mark05/plots/AD_DL_rand_n"+"{nodes}_samples{samples}.svg"
    run:
        # make ``samples`` random trees
        trees = []
        seeds = jnp.arange(int(wildcards.samples) + 1)
        for i in range(int(wildcards.samples) + 1):
            seed = seeds[i]
            trees.append(yg.tree_inference.make_tree(
                int(wildcards.nodes),
                params.tree_type,
                int(seed)))

        # get the first tree as a ref tree
        ref_tree = trees[0]
        # remove the first tree from the list
        trees = trees[1:]

        # compute the tree-tree metrics
        metric_ad = yg.distances.AncestorDescendantSimilarity().calculate
        ad_values = [metric_ad(ref_tree, tree) for tree in trees]  # type: ignore
        metric_dl = yg.distances.DifferentLineageSimilarity().calculate
        dl_values = [metric_dl(ref_tree, tree) for tree in trees]  # type: ignore

        # plot corner plot with seaborn
        sns.set_theme(style="ticks")
        g = sns.JointGrid(x=ad_values, y=dl_values, marginal_ticks=True)
        g.plot_joint(sns.scatterplot, s=10, alpha=0.5)
        g.plot_marginals(sns.histplot, kde=True)
        g.set_axis_labels("AD", "DL", fontsize=16)
        g.ax_joint.set_xlim(0, 1)
        g.ax_joint.set_ylim(0, 1)
        g.ax_joint.grid(True)
        g.savefig(output.cornerplot)
        g.savefig(output.cornerplot.replace(".svg", ".png"), dpi=300)

