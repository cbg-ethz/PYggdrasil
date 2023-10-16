# API

- [Analyze](analyze.md) provides analysis tools of tree samples and inference runs.
- [Distances](distances.md) provides tree-tree distance and similarity measures.
- [Interface](interface.md) provides dataclasses for processing MCMC data.
- [Serialize](serialize.md) contains utilities for serialization and  deserialization of trees, MCMC rus and analysis.
- [Tree Inference](tree_inference.md) implements the mutation tree infernce; scDNA mutation profiles synthesis.
- [Visualize](visualize.md) implements visualization of trees and MCMC runs and diagostics theirof.

This package handels trees with _Anytree_, see our adaption in the class _TreeNode_ below.  For the inference we convert trees to binary adjacency matrices see [Tree Inference](tree_inference.md), for reasons of performance.

::: pyggdrasil.TreeNode
::: pyggdrasil.compare_trees


