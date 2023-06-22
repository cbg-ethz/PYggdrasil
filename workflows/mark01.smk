"""Experiment mark01

 assessing the HUNTRESS trees with distance
 metrics under the SCITE generative model"""

# imports
import pyggdrasil as yg



#####################
experiment="mark01"

# Metrics: Distances / Similarity Measure to use
metrics = ["MP3","AD"]



#####################
# Auxiliary variables
true_tree_id = "tbd"

#####################

rule all:
    """Make the distance histograms for each metric."""
    input:
        expand(f"{WORKDIR}/{experiment}/plots/{{metrics}}_hist.svg", metric=metrics)


#rule make_histograms:
#    """Make the distance histograms for each metric."""
#    input:
#        f"WORKDIR/{experiment}/distances/{true_tree_id}/{metric}.trees"