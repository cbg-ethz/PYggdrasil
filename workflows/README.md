[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# PYggdrasil/workflows

Implements workflows to test and evaluate **PYggdrasil** using [snakemake](https://snakemake.readthedocs.io/en/stable/).


## Usage
Before running any rules, set the `WORKDIR` in `Snakefile` and the path to the `PYggdrasil` package in the 
`tree_inference.smk` file.

Shared and experimental workflows are implemented. Shared workflows can be called upon by string matching i.e.
```bash
    snakemake -c <n_cores> <WORKDIR>/<EXPERIMENT>/T_r_34_23.json
```
which runs created a random Tree of 34 nodes and generation seed 23, see `tree_inference._file_id` for details.
Where `<n_cores>` is the number of cores to use, and `<WORKDIR>/<EXPERIMENT>` is the directory of the output.

Or run full experiments, titled m `markXX` by

```bash
    cd workflows
    snakemake -c <n_cores> markXX 
```


To inspect the DAG of the snakemake rule, e.g. `mark00`, run
```bash
snakemake --dag mark00 | dot -Tpng > dag.png
```

## Environment
Whilst we use [Poetry](https://python-poetry.org/) as a dependency manager, we recommend to used [Conda](https://docs.conda.io/en/latest/) to as the environment manager.

_PYggdrasil_'s dependencies are best installed via [Poetry](https://python-poetry.org/) in an envrioment created by [Conda](https://docs.conda.io/en/latest/).

This seperation allows to disect the dependencies of the package from the dependencies of the ``workflows``,
such as [snakemake](https://snakemake.readthedocs.io/en/stable/).

We recommend a setup via `mini-conda.`, i.e on linux:

```commandline
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```
Note that the above installs miniconda in your home directory.

Further, enhance by adding mamba to the conda environment, for faster resolving of dependencies:
```commandline
conda install mamba -n base -c conda-forge
```

Then, create a new environment for the project:
```commandline
cd PYggdrasil/
mamba env create -f environment.yml
```



Then add in all project specific dependencies via:
```commandline
cd PYggdrasil/
conda activate PYggdrasil
pip install -e .
```
This should install all the dependencies, and make the package available in the environment `PYggdrasil` that is currently active by running the prior command.



