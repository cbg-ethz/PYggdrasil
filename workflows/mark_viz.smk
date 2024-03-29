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
        f"{DATADIR}/{experiment}/plots/T_r_10_42_relabeled.svg",
        f"{DATADIR}/{experiment}/plots/T_d_10_42_relabeled.svg",
        f"{DATADIR}/{experiment}/plots/T_s_10_relabeled.svg",
        f"{DATADIR}/{experiment}/plots/T_r_6_42_relabeled.svg",
        f"{DATADIR}/{experiment}/plots/T_r_51_42_relabeled.svg",
        f"{DATADIR}/{experiment}/plots/T_r_6_45_relabeled.svg",
        f"{DATADIR}/{experiment}/plots/T_r_6_20_relabeled.svg",
        f"{DATADIR}/{experiment}/plots/T_r_6_34_relabeled.svg",
        f"{DATADIR}/{experiment}/plots/T_r_6_89_relabeled.svg",



