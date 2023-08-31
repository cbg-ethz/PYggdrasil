"""Experiment mark05

 assessing the MCMC tree dispersion per tree-tree metric
 as a baseline to understand the tree-tree metric."""

# imports
import jax.random as random
import pyggdrasil as yg
import seaborn as sns

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
        cornerplot = f"{DATADIR}/mark05/plots/AD_DL_cornerplot.svg"
    run:
        # generate two lists of random numbers
        seed = 43
        key = random.PRNGKey(seed)
        # split key
        key1, key2 = random.split(key)
        x = random.normal(key1,(1000,))
        y = random.normal(key2,(1000,))

        # plot corner plot with seaborn
        sns.set_theme(style="ticks")
        g = sns.JointGrid(x=x, y=y, marginal_ticks=True)
        g.plot_joint(sns.scatterplot, s=10, alpha=0.5)
        g.plot_marginals(sns.histplot, kde=True)
        g.set_axis_labels("x", "y", fontsize=16)
        g.ax_joint.set_xticks([-3, -2, -1, 0, 1, 2, 3])
        g.ax_joint.set_yticks([-3, -2, -1, 0, 1, 2, 3])
        g.ax_joint.set_xlim(-3, 3)
        g.ax_joint.set_ylim(-3, 3)
        g.ax_joint.grid(True)
        g.savefig(output.cornerplot)
        g.savefig(output.cornerplot.replace(".svg", ".png"), dpi=300)

