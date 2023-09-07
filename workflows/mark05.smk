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
        tree_seeds = [43,65,78,23,98],
        tree_type = yg.tree_inference.TreeType.RANDOM,
        mcmc_seed = 43,

    output:
        cornerplot = f"{DATADIR}/mark05/plots/AD_DL_mcmc_n"+"{nodes}_samples{samples}.svg"
    run:
        # make the initial trees
        initial_trees = []

        for seed in params.tree_seeds:
            initial_trees.append(yg.tree_inference.make_tree(
                int(wildcards.nodes),
                params.tree_type,
                int(seed)))

        trees_per_ref = []
        samples_pre_ref = int(wildcards.samples) // len(initial_trees)
        for init_tree in initial_trees:
            # now evolve the tree `tree_samples` times
            rng = random.PRNGKey(params.mcmc_seed)
            trees =yg.tree_inference.evolve_tree_mcmc_all(
                init_tree,
                samples_pre_ref,
                rng)
            trees_per_ref.append(trees)

        # get the tree-tree metrics
        metric_ad = yg.distances.AncestorDescendantSimilarity().calculate
        metric_dl = yg.distances.DifferentLineageSimilarity().calculate

        ad_values_per_ref = []
        dl_values_per_ref = []

        for ref_tree in initial_trees:
            ad_values = [metric_ad(ref_tree, tree) for tree in trees]  # type: ignore
            dl_values = [metric_dl(ref_tree, tree) for tree in trees]  # type: ignore
            ad_values_per_ref.append(ad_values)
            dl_values_per_ref.append(dl_values)

        # Create a list of colors for plotting
        colors = ['b', 'g', 'r', 'c', 'm']

        # Ensure the number of distributions matches the number of colors
        if len(initial_trees) > len(colors):
            raise ValueError("There are more distributions than colors.")

        # Initialize the seaborn theme
        sns.set_theme(style="ticks")

        # Initialize the JointGrid outside the loop
        g = sns.JointGrid(marginal_ticks=True)

        # Loop through ref_trees
        for i in range(len(initial_trees)):
            ad_values = ad_values_per_ref[i]
            dl_values = dl_values_per_ref[i]

            # Inside the loop, set the data for the JointGrid and specify color
            g.x = ad_values
            g.y = dl_values
            color = colors.pop(0)
            g.plot_joint(sns.scatterplot,s=10,alpha=0.5,color=color)
            g.plot_marginals(sns.histplot,kde=True,color=color)

        # Set labels and limits outside the loop
        g.set_axis_labels("AD","DL",fontsize=20)
        g.ax_joint.set_xlim(-0.05,1.05)
        g.ax_joint.set_ylim(-0.05,1.05)
        g.ax_joint.grid(True)

        # Set the tick number size for both x-axis and y-axis
        g.ax_joint.tick_params(axis='both',labelsize=14)
        g.ax_joint.xaxis.set_tick_params(labelsize=14)
        g.ax_joint.yaxis.set_tick_params(labelsize=14)

        # Save the plot
        g.savefig(output.cornerplot)  # svg
        g.savefig(output.cornerplot,dpi=300)


rule mark05_random:
    """Test the corner plot"""
    params:
        tree_type = yg.tree_inference.TreeType.RANDOM,
    output:
        cornerplot = f"{DATADIR}/mark05/plots/AD_DL_rand_n"+"{nodes}_samples{samples}.svg"
    run:
        # make ``samples`` random trees
        trees = []
        seeds = jnp.arange(int(wildcards.samples) + 5)
        for i in range(int(wildcards.samples) + 5):
            seed = seeds[i]
            trees.append(yg.tree_inference.make_tree(
                int(wildcards.nodes),
                params.tree_type,
                int(seed)))

        # get the first tree 5 trees as reference
        ref_trees = trees[:5]
        # remove the 5 trees from the list
        trees = trees[5:]

        # get the tree-tree metrics
        metric_ad = yg.distances.AncestorDescendantSimilarity().calculate
        metric_dl = yg.distances.DifferentLineageSimilarity().calculate

        ad_values_per_ref = []
        dl_values_per_ref = []

        for ref_tree in ref_trees:
            ad_values = [metric_ad(ref_tree, tree) for tree in trees]  # type: ignore
            dl_values = [metric_dl(ref_tree, tree) for tree in trees]  # type: ignore
            ad_values_per_ref.append(ad_values)
            dl_values_per_ref.append(dl_values)

        # Create a list of colors for plotting
        colors = ['b', 'g', 'r', 'c', 'm']

        # Ensure the number of distributions matches the number of colors
        if len(ref_trees) > len(colors):
            raise ValueError("There are more distributions than colors.")

        # Initialize the seaborn theme
        sns.set_theme(style="ticks")

        # Initialize the JointGrid outside the loop
        g = sns.JointGrid(marginal_ticks=True)

        # Loop through ref_trees
        for i in range(len(ref_trees)):
            ad_values = ad_values_per_ref[i]
            dl_values = dl_values_per_ref[i]

            # Inside the loop, set the data for the JointGrid and specify color
            g.x = ad_values
            g.y = dl_values
            color = colors.pop(0)
            g.plot_joint(sns.scatterplot,s=10,alpha=0.5,color=color)
            g.plot_marginals(sns.histplot,kde=True,color=color)

        # Set labels and limits outside the loop
        g.set_axis_labels("AD","DL",fontsize=18)
        g.ax_joint.set_xlim(-0.05,1.05)
        g.ax_joint.set_ylim(-0.05,1.05)
        g.ax_joint.grid(True)

        # Set the tick number size for both x-axis and y-axis
        g.ax_joint.tick_params(axis='both',labelsize=14)
        g.ax_joint.xaxis.set_tick_params(labelsize=14)
        g.ax_joint.yaxis.set_tick_params(labelsize=14)

        # Save the plot
        g.savefig(output.cornerplot) # svg
        g.savefig(output.cornerplot,dpi=300)

