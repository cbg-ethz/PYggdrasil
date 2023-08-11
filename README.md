[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![build](https://github.com/cbg-ethz/PYggdrasil/actions/workflows/test.yml/badge.svg)](https://github.com/cbg-ethz/PYggdrasil/actions/workflows/test.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<!-- TODO (Gordon): Add snakefmt back in when/if fixed. See https://github.com/snakemake/snakefmt/issues/197 [![Code style: snakefmt](https://img.shields.io/badge/code%20style-snakefmt-000000.svg)](https://github.com/snakemake/snakefmt) -->

# PYggdrasil

Python package for inference and analysis of mutation trees.


## Usage

```python
import pyggdrasil as yg
```


## Contributing

### Setting up the repository

To build package and maintain dependencies we use [Poetry](https://python-poetry.org/).
In particular, it's good to install it and become familiar with its basic functionalities by reading the documentation. 

To set up the environment (together with development tools) run:
```bash
$ poetry install --with dev
$ poetry run pre-commit install
```

Then, you will be able to run tests:
```bash
$ poetry run pytest
```
... or check the types:
```bash
$ poetry run pyright
```

Alternatively, you may prefer to work with the right Python environment using:
```bash
$ poetry shell
$ pytest
```

### Existing code quality checks
The code quality checks run during on GitHub can be seen in ``.github/workflows/test.yml``.

We are using:
  - [Ruff](https://github.com/charliermarsh/ruff) to lint the code.
  - [Black](https://github.com/psf/black) to format the code.
  - [Pyright](https://github.com/microsoft/pyright) to check the types.
  - [Pytest](https://docs.pytest.org/) to run the unit tests.
  - [Interrogate](https://interrogate.readthedocs.io/) to check the documentation.
<!-- TODO (Gordon): Add snakefmt back in when/if fixed. See https://github.com/snakemake/snakefmt/issues/197 -->
<!-- [Snakefmt](https://github.com/snakemake/snakefmt) to format Snakemake workflows.-->


### Workflow

We use [Feature Branch Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow),
in which modifications of the code should happen via small pull requests.

We recommend submitting small pull requests and starting with drafts outlining proposed changes.

### Code organisation

- The package code is in ``src/pyggdrasil/`` and is partitioned into subpackages.
- The unit tests are in ``tests/`` and the structure of this directory should reflect the one of the package.
- Experimental workflows are in ``workflows/``, with a description of how to set up the environment in ``workflows/README.md``

