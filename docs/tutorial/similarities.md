# Tree Similarities

In this tutorial we generate a bunch of trees and compute their pairwise
similarities and viszalize them.

The visualizations are built with [networkX](https://networkx.org/) and
[matplotlib](https://matplotlib.org/). Quite some specification was done
to make the visualizations look nice.

Setting up the envrionment:

<details>
<summary>Code</summary>

``` python
## imports
import pyggdrasil as yg
from pathlib import Path
import matplotlib.pyplot as plt

# matplotlib inline
%matplotlib inline 
```

</details>

## Generate trees

### Random Tree

``` python
tree_type = yg.tree_inference.TreeType.RANDOM
tree_seed = 487
nodes = 10
random_tree = yg.tree_inference.make_tree(nodes, tree_type, tree_seed)
random_tree.print_topo()
```

    9
    ├── 6
    │   ├── 0
    │   └── 3
    └── 8
        ├── 4
        │   └── 1
        ├── 5
        └── 7
            └── 2

Now let’s visualize this properly.

``` python
save_dir = Path("tree_sim_figs")
save_dir.mkdir(parents=True, exist_ok=True)
save_name = "random_tree"
yg.visualize.plot_tree_no_print(random_tree, save_name, save_dir)
```

![Random Tree](../tree_sim_figs/random_tree.svg)

### Star Tree

``` python
tree_type = yg.tree_inference.TreeType.STAR
tree_seed = 487
nodes = 10
star_tree = yg.tree_inference.make_tree(nodes, tree_type, tree_seed)
```

Now let’s visualize this properly.

<details>
<summary>Code</summary>

``` python
save_name = "star_tree"
yg.visualize.plot_tree_no_print(star_tree, save_name, save_dir)
```

</details>

![Star Tree](../tree_sim_figs/star_tree.svg)

### Deep Tree

``` python
tree_type = yg.tree_inference.TreeType.DEEP
tree_seed = 487
nodes = 10
deep_tree = yg.tree_inference.make_tree(nodes, tree_type, tree_seed)
```

Now let’s visualize this properly.

<details>
<summary>Code</summary>

``` python
save_name = "deep_tree"
yg.visualize.plot_tree_no_print(deep_tree, save_name, save_dir)
```

</details>

![Deep Tree](../tree_sim_figs/deep_tree.svg)

**Note:** PYggdrasil inplements two more advanced tree generation
methods.

1.  MCMC tree generation - takes a tree and evolves it by a fixed number
    random moves implemnted with SCITE.
2.  HUNTRESS inference - takes a cell-mutation profile and infers a tree
    with HUNTRESS.

## Compute Similarities

What similarities to care for? We can compute the following
similarities:

- Ancestor-Descendant (AD) Similarity
- Different-Lineage (DL) Similarity

``` python
# random tree to star tree
AD_star = yg.distances.AncestorDescendantSimilarity().calculate(random_tree, star_tree)
DL_star = yg.distances.DifferentLineageSimilarity().calculate(random_tree, star_tree)

print(f"AD Similarity: {AD_star}")
print(f"DL Similarity: {DL_star}")
```

    AD Similarity: 0.0
    DL Similarity: 1.0

- AD : 0.0 makes sense, since the star tree has no internal nodes, so no
  nodes are ancestors of other nodes. (AD does not consider the root
  node)
- DL : 1.0 makes sense, since the star tree has no internal nodes, so
  all nodes are in different lineages.

<details>
<summary>Code</summary>

``` python
# random tree to deep tree
AD_deep = yg.distances.AncestorDescendantSimilarity().calculate(random_tree, deep_tree)
DL_deep = yg.distances.DifferentLineageSimilarity().calculate(random_tree, deep_tree)

print(f"AD Similarity: {AD_deep}")
print(f"DL Similarity: {DL_deep}")
```

</details>

    AD Similarity: 0.11111111111111116
    DL Similarity: 0.0

- AD: some chronological order is preserved, but not all.
- DL: 0.0 makes sense, as all nodes are in the same lineage.

Let’s have another random tree for fun:

<details>
<summary>Code</summary>

``` python
tree_type = yg.tree_inference.TreeType.RANDOM
tree_seed = 4897
nodes = 10
random_tree2 = yg.tree_inference.make_tree(nodes, tree_type, tree_seed)
save_name = "random_tree2"
yg.visualize.plot_tree_no_print(random_tree, save_name, save_dir)
```

</details>

![Random Tree2](../tree_sim_figs/random_tree2.svg)

<details>
<summary>Code</summary>

``` python
# random tree to another random tree
AD_random = yg.distances.AncestorDescendantSimilarity().calculate(random_tree, random_tree2)
DL_random = yg.distances.DifferentLineageSimilarity().calculate(random_tree, random_tree2)

print(f"AD Similarity: {AD_random}")
print(f"DL Similarity: {DL_random}")
```

</details>

    AD Similarity: 0.0
    DL Similarity: 0.8148148148148148

We see a more balanced mix of AD and DL similarities. Here by chance a
AD of 0 again. Well, these are small trees.
