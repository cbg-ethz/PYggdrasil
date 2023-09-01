"""Experiment mark05

 assessing the MCMC tree dispersion per tree-tree metric
 as a baseline to understand the tree-tree metric."""

# imports
import jax.random as random
import pyggdrasil as yg
import seaborn as sns

from pathlib import Path

################################################################################
# Environment variables
#DATADIR = "/cluster/work/bewi/members/gkoehn/data"
DATADIR = "../data.nosync"

rule mark05:
    """Assessing the MCMC tree dispersion per tree-tree metric"""
    input:
        cornerplot = f"{DATADIR}/mark05/plots/AD_DL_cornerplot.svg"


rule test_corner_plot:
    """Test the corner plot"""

    output:
        cornerplot = f"{DATADIR}/mark05/plots/AD_DL_cornerplot_test.svg"
    run:
        # generate two lists of random numbers
        seed = 43
        key = random.PRNGKey(seed)
        # split key
        key1, key2 = random.split(key)
        x = random.normal(key1,(1000,)) + 1
        y = random.normal(key2,(1000,)) + 1

        # plot corner plot with seaborn
        sns.set_theme(style="ticks")
        g = sns.JointGrid(x=x, y=y, marginal_ticks=True)
        g.plot_joint(sns.scatterplot, s=10, alpha=0.5)
        g.plot_marginals(sns.histplot, kde=True)
        g.set_axis_labels("AD", "DL", fontsize=16)
        #g.ax_joint.set_xticks([0, 1, 2, 3])
        #g.ax_joint.set_yticks([-3, -2, -1, 0, 1, 2, 3])
        g.ax_joint.set_xlim(0, 1)
        g.ax_joint.set_ylim(0, 1)
        g.ax_joint.grid(True)
        g.savefig(output.cornerplot)
        g.savefig(output.cornerplot.replace(".svg", ".png"), dpi=300)


rule mark04_exp:
    """Test the corner plot"""
    params:
        tree_nodes = 10,
        tree_seed = 43,
        tree_type = yg.tree_inference.TreeType.RANDOM,
        tree_samples = 100,
        mcmc_seed = 43,

    output:
        cornerplot = f"{DATADIR}/mark05/plots/AD_DL_cornerplot.svg"
    run:
        # make the initial tree
        init_tree = yg.tree_inference.make_tree(
            params.tree_nodes,
            params.tree_type,
            params.tree_seed)

        # now evolve the tree `tree_samples` times
        rng = random.PRNGKey(params.mcmc_seed)
        trees =yg.tree_inference.evolve_tree_mcmc_all(
            init_tree,
            params.tree_samples,
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




