# Analyzing MCMC Runs

In this notebook we will analyze the samples of a MCMC.

We analye the evolution of MCMC runs with tree similarities in
combination with the Gelman-Rubin statistic.

## Imports

<details>
<summary>Code</summary>

``` python
## imports
import pyggdrasil as yg
import jax.numpy as jnp
import numpy as np
import jax.random as random
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# matplotlib inline
%matplotlib inline 
```

</details>

## Run MCMC

Below we run 4 Markov Chains, for 100 iterations each, with different
initial trees.

### Generate a ground-truth mutation history and a noisy single-cell mutation profile

The below cell generates a random tree with 4 mutations, plus root. For
debugging we may use the *print_topo* to plot its topology.

<details>
<summary>Code</summary>

``` python
# make true tree
tree_type = yg.tree_inference.TreeType.RANDOM
mutations = 10
nodes = mutations + 1
tree_seed = 42
true_tree = yg.tree_inference.make_tree(nodes, tree_type, tree_seed) 
```

</details>

## Generate an initial tree to start the Markov Chain from

We also choose a random tree here.

<details>
<summary>Code</summary>

``` python
inital_trees = []
tree_type = yg.tree_inference.TreeType.RANDOM
for i in range(4):
    tree_seed = i
    inital_trees.append(yg.tree_inference.make_tree(nodes, tree_type, tree_seed))
```

</details>

## Generate a noisy single-cell mutation profile from the ground-truth tree

<details>
<summary>Code</summary>

``` python
## generate some little nois data
# Set up the simulation model
csm = yg.tree_inference.CellSimulationModel(
    n_cells=1000,
    n_mutations=mutations,
    fpr=0.01,
    fnr=0.2,
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
```

</details>

## Run the Markov Monte Carlo Chain

The below cell runs a 4 differnt MCMC chain. We initialize ti with the
initial tree from before. We configure the move probabilities and error
rates and run the MCMC chain for 100 iterations. The sampels are saved
to disk and loaded back into memory as chains may be very long.

``` python
mcmc_datas = []
n = 1
# run 4 chains, each with a different initial tree
for starting_tree in inital_trees:
        print("Starting MCMC for tree: ", n)
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
        # append the data to the list
        mcmc_datas.append(mcmc_data)
        #delete file
        (full_save_name).unlink()
        print("finished MCMC")
        n += 1
```

    Starting MCMC for tree:  1
    finished MCMC
    Starting MCMC for tree:  2
    finished MCMC
    Starting MCMC for tree:  3
    finished MCMC
    Starting MCMC for tree:  4
    finished MCMC

``` python
# unpack the data - reads in the serialized trees to Tree objects
# takes some time as tree objects are built and checked for validity
for i in range(len(mcmc_datas)):
    mcmc_datas[i] = yg.analyze.to_pure_mcmc_data(mcmc_datas[i])
```

## Let’s plot the log-probability of the trees over the iterations

``` python
for i in range(len(mcmc_datas)):
    plt.plot(mcmc_datas[i].iterations, mcmc_datas[i].log_probabilities)
plt.xlabel("Iteration")
plt.ylabel("Log-probability")
plt.grid()
plt.show()
```

<img
src="analyzeMCMC_files/figure-commonmark/log-prob-iter-output-1.png"
id="log-prob-iter"
alt="The evolution of the log-probability of the trees over the MCMC iterations." />

## Let’s calculate the tree similarity over the iterations

``` python
metrics = ["AD","DL"]
base_tree = true_tree

# Create an empty list to store the results
results = []

for metric_name in metrics:
    metric = yg.analyze.Metrics.get(metric_name)

    for i in range(len(mcmc_datas)):
        iteration, result = yg.analyze.analyze_mcmc_run(mcmc_datas[i], metric, base_tree)
        print("Finished analyzing chain: ", i)

        # Append the result to the results list as a dictionary
        results.append({"Iteration": iteration, metric_name: result})

# Convert the results list to a pandas DataFrame
results = pd.DataFrame(results)

# Display the first rows
print(results.head())
```

    INFO:pyggdrasil.analyze._calculation:Using base tree: 10: None
    ├── 2: None
    ├── 3: None
    ├── 6: None
    │   ├── 0: None
    │   └── 5: None
    ├── 8: None
    │   ├── 1: None
    │   ├── 4: None
    │   └── 7: None
    └── 9: None
    .
    INFO:pyggdrasil.analyze._calculation:Iteration 1.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 2.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 3.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 4.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 5.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 6.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 7.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 8.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 9.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 10.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 11.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 12.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 13.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 14.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 15.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 16.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 17.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 18.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 19.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 20.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 21.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 22.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 23.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 24.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 25.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 26.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 27.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 28.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 29.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 30.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 31.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 32.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 33.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 34.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 35.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 36.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 37.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 38.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 39.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 40.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 41.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 42.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 43.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 44.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 45.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 46.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 47.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 48.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 49.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 50.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 51.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 52.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 53.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 54.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 55.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 56.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 57.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 58.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 59.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 60.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 61.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 62.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 63.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 64.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 65.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 66.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 67.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 68.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 69.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 70.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 71.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 72.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 73.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 74.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 75.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 76.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 77.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 78.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 79.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 80.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 81.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 82.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 83.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 84.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 85.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 86.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 87.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 88.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 89.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 90.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 91.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 92.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 93.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 94.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 95.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 96.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 97.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 98.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 99.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 100.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Using base tree: 10: None
    ├── 2: None
    ├── 3: None
    ├── 6: None
    │   ├── 0: None
    │   └── 5: None
    ├── 8: None
    │   ├── 1: None
    │   ├── 4: None
    │   └── 7: None
    └── 9: None
    .
    INFO:pyggdrasil.analyze._calculation:Iteration 1.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 2.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 3.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 4.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 5.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 6.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 7.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 8.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 9.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 10.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 11.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 12.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 13.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 14.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 15.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 16.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 17.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 18.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 19.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 20.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 21.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 22.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 23.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 24.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 25.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 26.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 27.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 28.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 29.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 30.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 31.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 32.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 33.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 34.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 35.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 36.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 37.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 38.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 39.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 40.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 41.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 42.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 43.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 44.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 45.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 46.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 47.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 48.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 49.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 50.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 51.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 52.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 53.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 54.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 55.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 56.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 57.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 58.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 59.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 60.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 61.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 62.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 63.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 64.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 65.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 66.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 67.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 68.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 69.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 70.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 71.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 72.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 73.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 74.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 75.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 76.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 77.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 78.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 79.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 80.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 81.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 82.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 83.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 84.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 85.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 86.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 87.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 88.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 89.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 90.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 91.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 92.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 93.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 94.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 95.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 96.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 97.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 98.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 99.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 100.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Using base tree: 10: None
    ├── 2: None
    ├── 3: None
    ├── 6: None
    │   ├── 0: None
    │   └── 5: None
    ├── 8: None
    │   ├── 1: None
    │   ├── 4: None
    │   └── 7: None
    └── 9: None
    .
    INFO:pyggdrasil.analyze._calculation:Iteration 1.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 2.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 3.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 4.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 5.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 6.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 7.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 8.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 9.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 10.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 11.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 12.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 13.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 14.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 15.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 16.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 17.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 18.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 19.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 20.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 21.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 22.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 23.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 24.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 25.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 26.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 27.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 28.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 29.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 30.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 31.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 32.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 33.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 34.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 35.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 36.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 37.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 38.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 39.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 40.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 41.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 42.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 43.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 44.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 45.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 46.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 47.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 48.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 49.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 50.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 51.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 52.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 53.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 54.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 55.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 56.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 57.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 58.0: 0.6.
    INFO:pyggdrasil.analyze._calculation:Iteration 59.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 60.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 61.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 62.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 63.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 64.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 65.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 66.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 67.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 68.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 69.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 70.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 71.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 72.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 73.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 74.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 75.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 76.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 77.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 78.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 79.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 80.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 81.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 82.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 83.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 84.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 85.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 86.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 87.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 88.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 89.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 90.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 91.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 92.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 93.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 94.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 95.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 96.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 97.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 98.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 99.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 100.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Using base tree: 10: None
    ├── 2: None
    ├── 3: None
    ├── 6: None
    │   ├── 0: None
    │   └── 5: None
    ├── 8: None
    │   ├── 1: None
    │   ├── 4: None
    │   └── 7: None
    └── 9: None
    .
    INFO:pyggdrasil.analyze._calculation:Iteration 1.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 2.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 3.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 4.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 5.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 6.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 7.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 8.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 9.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 10.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 11.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 12.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 13.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 14.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 15.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 16.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 17.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 18.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 19.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 20.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 21.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 22.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 23.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 24.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 25.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 26.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 27.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 28.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 29.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 30.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 31.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 32.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 33.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 34.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 35.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 36.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 37.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 38.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 39.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 40.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 41.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 42.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 43.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 44.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 45.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 46.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 47.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 48.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 49.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 50.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 51.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 52.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 53.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 54.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 55.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 56.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 57.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 58.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 59.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 60.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 61.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 62.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 63.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 64.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 65.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 66.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 67.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 68.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 69.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 70.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 71.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 72.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 73.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 74.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 75.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 76.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 77.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 78.0: 0.4.
    INFO:pyggdrasil.analyze._calculation:Iteration 79.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 80.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 81.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 82.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 83.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 84.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 85.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 86.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 87.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 88.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 89.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 90.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 91.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 92.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 93.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 94.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 95.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 96.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 97.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 98.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 99.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 100.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Using base tree: 10: None
    ├── 2: None
    ├── 3: None
    ├── 6: None
    │   ├── 0: None
    │   └── 5: None
    ├── 8: None
    │   ├── 1: None
    │   ├── 4: None
    │   └── 7: None
    └── 9: None
    .
    INFO:pyggdrasil.analyze._calculation:Iteration 1.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 2.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 3.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 4.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 5.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 6.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 7.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 8.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 9.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 10.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 11.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 12.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 13.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 14.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 15.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 16.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 17.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 18.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 19.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 20.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 21.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 22.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 23.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 24.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 25.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 26.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 27.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 28.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 29.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 30.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 31.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 32.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 33.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 34.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 35.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 36.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 37.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 38.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 39.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 40.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 41.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 42.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 43.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 44.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 45.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 46.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 47.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 48.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 49.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 50.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 51.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 52.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 53.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 54.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 55.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 56.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 57.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 58.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 59.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 60.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 61.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 62.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 63.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 64.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 65.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 66.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 67.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 68.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 69.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 70.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 71.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 72.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 73.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 74.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 75.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 76.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 77.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 78.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 79.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 80.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 81.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 82.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 83.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 84.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 85.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 86.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 87.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 88.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 89.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 90.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 91.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 92.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 93.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 94.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 95.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 96.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 97.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 98.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 99.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 100.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Using base tree: 10: None
    ├── 2: None
    ├── 3: None
    ├── 6: None
    │   ├── 0: None
    │   └── 5: None
    ├── 8: None
    │   ├── 1: None
    │   ├── 4: None
    │   └── 7: None
    └── 9: None
    .
    INFO:pyggdrasil.analyze._calculation:Iteration 1.0: 0.7.
    INFO:pyggdrasil.analyze._calculation:Iteration 2.0: 0.7.
    INFO:pyggdrasil.analyze._calculation:Iteration 3.0: 0.7.
    INFO:pyggdrasil.analyze._calculation:Iteration 4.0: 0.7.
    INFO:pyggdrasil.analyze._calculation:Iteration 5.0: 0.725.
    INFO:pyggdrasil.analyze._calculation:Iteration 6.0: 0.725.
    INFO:pyggdrasil.analyze._calculation:Iteration 7.0: 0.725.
    INFO:pyggdrasil.analyze._calculation:Iteration 8.0: 0.725.
    INFO:pyggdrasil.analyze._calculation:Iteration 9.0: 0.75.
    INFO:pyggdrasil.analyze._calculation:Iteration 10.0: 0.75.
    INFO:pyggdrasil.analyze._calculation:Iteration 11.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 12.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 13.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 14.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 15.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 16.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 17.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 18.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 19.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 20.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 21.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 22.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 23.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 24.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 25.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 26.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 27.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 28.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 29.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 30.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 31.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 32.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 33.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 34.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 35.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 36.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 37.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 38.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 39.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 40.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 41.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 42.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 43.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 44.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 45.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 46.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 47.0: 0.85.
    INFO:pyggdrasil.analyze._calculation:Iteration 48.0: 0.85.
    INFO:pyggdrasil.analyze._calculation:Iteration 49.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 50.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 51.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 52.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 53.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 54.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 55.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 56.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 57.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 58.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 59.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 60.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 61.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 62.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 63.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 64.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 65.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 66.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 67.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 68.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 69.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 70.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 71.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 72.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 73.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 74.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 75.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 76.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 77.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 78.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 79.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 80.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 81.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 82.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 83.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 84.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 85.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 86.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 87.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 88.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 89.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 90.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 91.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 92.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 93.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 94.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 95.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 96.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 97.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 98.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 99.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 100.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Using base tree: 10: None
    ├── 2: None
    ├── 3: None
    ├── 6: None
    │   ├── 0: None
    │   └── 5: None
    ├── 8: None
    │   ├── 1: None
    │   ├── 4: None
    │   └── 7: None
    └── 9: None
    .
    INFO:pyggdrasil.analyze._calculation:Iteration 1.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 2.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 3.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 4.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 5.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 6.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 7.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 8.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 9.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 10.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 11.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 12.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 13.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 14.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 15.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 16.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 17.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 18.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 19.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 20.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 21.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 22.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 23.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 24.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 25.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 26.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 27.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 28.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 29.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 30.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 31.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 32.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 33.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 34.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 35.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 36.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 37.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 38.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 39.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 40.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 41.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 42.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 43.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 44.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 45.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 46.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 47.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 48.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 49.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 50.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 51.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 52.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 53.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 54.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 55.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 56.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 57.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 58.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 59.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 60.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 61.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 62.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 63.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 64.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 65.0: 0.9.
    INFO:pyggdrasil.analyze._calculation:Iteration 66.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 67.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 68.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 69.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 70.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 71.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 72.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 73.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 74.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 75.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 76.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 77.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 78.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 79.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 80.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 81.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 82.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 83.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 84.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 85.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 86.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 87.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 88.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 89.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 90.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 91.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 92.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 93.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 94.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 95.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 96.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 97.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 98.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 99.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Iteration 100.0: 0.95.
    INFO:pyggdrasil.analyze._calculation:Using base tree: 10: None
    ├── 2: None
    ├── 3: None
    ├── 6: None
    │   ├── 0: None
    │   └── 5: None
    ├── 8: None
    │   ├── 1: None
    │   ├── 4: None
    │   └── 7: None
    └── 9: None
    .
    INFO:pyggdrasil.analyze._calculation:Iteration 1.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 2.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 3.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 4.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 5.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 6.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 7.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 8.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 9.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 10.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 11.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 12.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 13.0: 0.8.
    INFO:pyggdrasil.analyze._calculation:Iteration 14.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 15.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 16.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 17.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 18.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 19.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 20.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 21.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 22.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 23.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 24.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 25.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 26.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 27.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 28.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 29.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 30.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 31.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 32.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 33.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 34.0: 0.825.
    INFO:pyggdrasil.analyze._calculation:Iteration 35.0: 0.85.
    INFO:pyggdrasil.analyze._calculation:Iteration 36.0: 0.85.
    INFO:pyggdrasil.analyze._calculation:Iteration 37.0: 0.85.
    INFO:pyggdrasil.analyze._calculation:Iteration 38.0: 0.85.
    INFO:pyggdrasil.analyze._calculation:Iteration 39.0: 0.85.
    INFO:pyggdrasil.analyze._calculation:Iteration 40.0: 0.85.
    INFO:pyggdrasil.analyze._calculation:Iteration 41.0: 0.85.
    INFO:pyggdrasil.analyze._calculation:Iteration 42.0: 0.85.
    INFO:pyggdrasil.analyze._calculation:Iteration 43.0: 0.85.
    INFO:pyggdrasil.analyze._calculation:Iteration 44.0: 0.85.
    INFO:pyggdrasil.analyze._calculation:Iteration 45.0: 0.85.
    INFO:pyggdrasil.analyze._calculation:Iteration 46.0: 0.85.
    INFO:pyggdrasil.analyze._calculation:Iteration 47.0: 0.85.
    INFO:pyggdrasil.analyze._calculation:Iteration 48.0: 0.85.
    INFO:pyggdrasil.analyze._calculation:Iteration 49.0: 0.85.
    INFO:pyggdrasil.analyze._calculation:Iteration 50.0: 0.85.
    INFO:pyggdrasil.analyze._calculation:Iteration 51.0: 0.85.
    INFO:pyggdrasil.analyze._calculation:Iteration 52.0: 0.85.
    INFO:pyggdrasil.analyze._calculation:Iteration 53.0: 0.85.
    INFO:pyggdrasil.analyze._calculation:Iteration 54.0: 0.85.
    INFO:pyggdrasil.analyze._calculation:Iteration 55.0: 0.85.
    INFO:pyggdrasil.analyze._calculation:Iteration 56.0: 0.85.
    INFO:pyggdrasil.analyze._calculation:Iteration 57.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 58.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 59.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 60.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 61.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 62.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 63.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 64.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 65.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 66.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 67.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 68.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 69.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 70.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 71.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 72.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 73.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 74.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 75.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 76.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 77.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 78.0: 0.875.
    INFO:pyggdrasil.analyze._calculation:Iteration 79.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 80.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 81.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 82.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 83.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 84.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 85.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 86.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 87.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 88.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 89.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 90.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 91.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 92.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 93.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 94.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 95.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 96.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 97.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 98.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 99.0: 0.925.
    INFO:pyggdrasil.analyze._calculation:Iteration 100.0: 0.925.

    Finished analyzing chain:  0
    Finished analyzing chain:  1
    Finished analyzing chain:  2
    Finished analyzing chain:  3
    Finished analyzing chain:  0
    Finished analyzing chain:  1
    Finished analyzing chain:  2
    Finished analyzing chain:  3
                                               Iteration  \
    0  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14...   
    1  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14...   
    2  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14...   
    3  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14...   
    4  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14...   

                                                      AD  \
    0  [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.8, 0.8, ...   
    1  [0.4, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.6, 0.8, ...   
    2  [0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4, 0.4, ...   
    3  [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, ...   
    4                                                NaN   

                                                      DL  
    0                                                NaN  
    1                                                NaN  
    2                                                NaN  
    3                                                NaN  
    4  [0.875, 0.875, 0.875, 0.875, 0.875, 0.875, 0.8...  
