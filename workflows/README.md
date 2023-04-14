[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# PYggdrasil/workflows

Implements workflows to test and evaluate **PYggdrasil** using [snakemake](https://snakemake.readthedocs.io/en/stable/).

**Note: These workflows are in early development.** 

## Usage
To run all rules run in this directory
```bash
    snakemake -c 1
```
which runs the _all_ rule in the Snakemake file.

## Environment
We recommend the setup via [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html), 
for quick setup. 

To set up the environment run:
```bash
micromamba create -f ../environment.yml
```

## Code organisation

The workflows are in this directory, but generate output files are in 
```bash
../data/
```

