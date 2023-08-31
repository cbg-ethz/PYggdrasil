"""Experiment mark05

 assessing the MCMC tree dispersion per tree-tree metric
 as a baseline to understand the tree-tree metric."""

# imports
import jax.numpy as jnp
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
