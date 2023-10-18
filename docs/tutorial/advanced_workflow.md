# Advanced Workflows

As part of my thesis at ETH Zürich I,
[gordonkoehn](https://github.com/gordonkoehn), designed a series of
advanced [Snakemake](https://snakemake.readthedocs.io/en/stable/)
workflows as reproducible experiments.

We investigated particular aspects of SCITE’s MCMC inference:

- *warm-up* of the MCMC chain
- *multimodal* posterior distributions
- *convergence diagnsotics* in the mutation tree space

The workflows are available in:

- *mark01* : assessing the HUNTRESS trees with distance metrics under
  the SCITE generative model

- *mark02* : investigate convergence of SCITE MCMC chains, given
  different initial points with tree distances.

- *mark03* : investigate convergence of SCITE MCMC chains, given
  different initial points with tree distances.

- *mark04* : assessing the HUNTRESS trees with distance metrics under
  the SCITE generative model.

- *mark05* : assessing the MCMC tree dispersion per tree-tree metric as
  a baseline to understand the tree-tree metric.

The outcome an analyis of the results is available in the
[thesis](TODO:%20(gordonkoehn)%20add%20thesis%20here.).
