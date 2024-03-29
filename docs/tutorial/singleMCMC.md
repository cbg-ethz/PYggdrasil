# Single MCMC Run

This tutorial shows how to run a single MCMC chain of SCITE using
PYggdrasil.

- We will generate our own ground-truth mutation history and generate a
  noisy single-cell mutation profile from it.
- We will then run a single MCMC chain to infer the mutation history
  from the noisy single-cell mutation profile.
- Visualize the results. The trees and the evolution of the MCMC.

## 0) Imports

``` python
## imports
import pyggdrasil as yg
import jax.numpy as jnp
import jax.random as random
from pathlib import Path
import matplotlib.pyplot as plt

# matplotlib inline
%matplotlib inline 
```

## 1) Generate a ground-truth mutation history and a noisy single-cell mutation profile

The below cell generates a random tree with 4 mutations, plus root. For
debugging we may use the *print_topo* to plot its topology.

``` python
# make true tree
tree_type = yg.tree_inference.TreeType.RANDOM
mutations = 4
nodes = mutations + 1
tree_seed = 42
true_tree = yg.tree_inference.make_tree(nodes, tree_type, tree_seed) 

true_tree.print_topo()
```

    4
    ├── 0
    ├── 2
    │   └── 1
    └── 3

## 2) Generate an initial tree to start the Markov Chain from

We also choose a random tree here.

``` python
# make random starting tree
tree_type = yg.tree_inference.TreeType.RANDOM
tree_seed = 487
starting_tree = yg.tree_inference.make_tree(nodes, tree_type, tree_seed)
starting_tree.print_topo()
```

    4
    ├── 1
    └── 3
        ├── 0
        └── 2

## 3) Generate a noisy single-cell mutation profile from the ground-truth tree

``` python
## generate some little nois data
# Set up the simulation model
csm = yg.tree_inference.CellSimulationModel(
    n_cells=1000,
    n_mutations=mutations,
    fpr=0.01,
    fnr=0.01,
    na_rate=0.0,
    observe_homozygous=False,
    strategy=yg.tree_inference.CellAttachmentStrategy.UNIFORM_EXCLUDE_ROOT,
)


# Generate Data
seed = 42
rng = random.PRNGKey(seed)
data = yg.tree_inference.gen_sim_data(
    csm,
    rng,
    true_tree
    )

mut_mat = jnp.array(data['noisy_mutation_mat'])
print(mut_mat)
```

    [[0 0 0 ... 0 0 1]
     [1 0 0 ... 0 0 0]
     [1 1 1 ... 0 0 0]
     [0 0 0 ... 1 1 0]]

## 4) Run the Markov Monte Carlo Chain

The below cell runs a single MCMC chain. We initialize it with
the initial tree from before. We configure the move probabilities
and error rates and run the MCMC chain for 100 iterations. 
The samples are saved to disk and loaded back into memory as chains may be very long.

``` python
## Run MCMC
# converting initial tree from TreeNode to Tree format
init_tree_t = yg.tree_inference.Tree.tree_from_tree_node(starting_tree)

## file handling
# set up save location
save_dir = Path("")
# make directory if it doesn't exist
save_dir.mkdir(parents=True, exist_ok=True)
save_name = "mcmc_test"
full_save_name = save_dir / f"{save_name}.json"
# make file / empty it if it exists
with open(full_save_name, "w") as f:
    f.write("")

# set the move probabilities and error rates
move_probs = yg.tree_inference.MoveProbabilities()
error_rates = yg.tree_inference.ErrorCombinations.IDEAL.value

# run mcmc sampler
yg.tree_inference.mcmc_sampler(
    rng_key=rng,
    data=mut_mat,
    error_rates=(error_rates.fpr, error_rates.fnr),
    move_probs=move_probs,
    num_samples=100,
    num_burn_in=0,
    out_fp=full_save_name,
    thinning=1,
    init_tree=init_tree_t,
)

# load the data from disk
mcmc_data = yg.serialize.read_mcmc_samples(save_dir / f"{save_name}.json")
#delete file
(full_save_name).unlink()
```

## 5) Visualize the results

In the following, we would like to plot the evolution of the MCMC chain
and the trees that were sampled. First, we convert the data from the serialized
format to a pureMCMCdata format. This is a simple data class that 
contains the trees and the log probabilities of the trees.

``` python
# unpack the data - reads in the serialized trees to Tree objects
# takes some time as tree objects are built and checked for validity
mcmc_samples = yg.analyze.to_pure_mcmc_data(mcmc_data)
```

Now, we may plot it.

``` python
plt.plot(mcmc_samples.iterations, mcmc_samples.log_probabilities)
plt.xlabel("Iteration")
plt.ylabel("Log-probability")
plt.grid()
plt.show()
```

<img src="../singleMCMC_files/figure-commonmark/log-prob-iter-output-1.png"
id="log-prob-iter"
alt="The evolution of the log-probability of the trees over the MCMC iterations." />

The log-probability quickly improved ! Seem like we have sampled a
quicke good tree most of the time.

Let’s have a look at the last tree in the chain. The last tree appears
to have a high log-probability.

``` python
# get last tree
last_tree = mcmc_samples.trees[-1]
# print topology
last_tree.print_topo()
```

    4
    ├── 0
    ├── 2
    │   └── 1
    └── 3

Is it perhaps the true tree?

``` python
# compare the true tree to the last tree
yg.compare_trees(last_tree, true_tree)
```

    True

Now note that the last tree does not need to be a good tree.
SCITE is just likely to spend more iterations exploring more
likely trees. Here, the last tree just turns out to be a
tree with the highest log-probability.

To acutally retrive a mutation tree from the posterior one 
would have to make a point estimate Maximum A Posteriori (MAP) tree,
 i.e. sampled the most times.  See SCITE paper for details.
