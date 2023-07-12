"""Experiment mark03

 Investigate convergence of SCITE MCMC chains,
 given different initial points with tree
 distances"""

# generate a random true tree as the mutation history
# For several scenarios (again, FPR, FNR, number of mutations and cells - hardness of task):
# same as before
# Generate an initial trees - for each category 10 trees
# - Deep Tree
# - Random Tree
# - Star Tree (only one possible)
# - Random Tree -> Huntress (hence generate 10 random trees)
# - (True Random Tree -> 5 MCMC moves -> Initial Tree)
# 5 trees.
#
# Start MCMC runs from: deep, shallow, random, huntress
# 2000 iterations
# Plot the mcmc runs against log_prob / distance to iteration number coloured by their initial tree type
# count the average number of iteration until true tree is reached.


