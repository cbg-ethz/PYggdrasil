"""Experiment mark_viz

 Implements visualization for the presentation of the
 implementations of trees and more.
 """

#####################
# Environment variables
DATADIR = "../data"
# DATADIR = "/cluster/work/bewi/members/gkoehn/data"
#####################

#####################
experiment="mark_viz"
#####################

rule mark_viz:
    input:
        f"{DATADIR}/{experiment}/plots/T_r_6_34.svg",
        f"{DATADIR}/{experiment}/plots/T_d_6_35.svg",
        f"{DATADIR}/{experiment}/plots/T_s_6.svg"
