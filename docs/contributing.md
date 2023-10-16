# Contributing

Thank you for your time to contribute to this project!
Below we present some guidelines.

## Reporting a bug

If you find a bug, please [submit a new issue](https://github.com/cbg-ethz/PYggdrasil/issues).

To be able to reproduce a bug, we will usually need the following information:

  - Versions of Python packages used (in particular version of this library).
  - A minimal code snippet allowing us to reproduce the bug.
  - What is the desired behaviour in the reported case?
  - What is the actual behaviour?


## Submitting a pull request

**Do:**

  - Do use [Google Style Guide](https://google.github.io/styleguide/pyguide.html). We use [black](https://github.com/psf/black) for code formatting.
  - Do write unit tests. We use [pytest](https://docs.pytest.org/).
  - Do write docstrings. We use [Material for Mkdocs](https://squidfunk.github.io/mkdocs-material/) to generate the documentation.
  - Do write high-level documentation as examples and tutorials, illustrating introduced features.
  - Do consider submitting a *draft* pull request with a description of proposed changes.
  - Do check the [Development section](#development).

**Don't:**

  - Don't include license information. This project is MIT licensed and by submitting your pull request you implicitly and irrevocably agree to use this.
  - Don't implement too many ideas in a single pull request. Multiple features should be implemented in separate pull requests.


## Development

### Workflow

We use [Feature Branch Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow),
in which modifications of the code should happen via small pull requests.

We recommend submitting small pull requests and starting with drafts outlining proposed changes.

### Installation & dependencies
To install the repository together with the dependencies run:
```
$ git clone git@github.com:cbg-ethz/PYggdrasil.git  # Clone the repository
$ poetry install --with dev                         # Install the dependencies
$ poetry run pre-commit install                     # Install pre-commit hooks
$ poetry run pytest                                 # Check if unit tests are passing
```


### Testing & Checking

Then, you will be able to run tests:
```bash
$ poetry run pytest
```
... or check the types:
```bash
$ poetry run pyright
```

### Existing code quality checks
The code quality checks run during on GitHub can be seen in ``.github/workflows/test.yml``.

We are using:
  - [Ruff](https://github.com/charliermarsh/ruff) to lint the code.
  - [Black](https://github.com/psf/black) to format the code.
  - [Pyright](https://github.com/microsoft/pyright) to check the types.
  - [Pytest](https://docs.pytest.org/) to run the unit tests.
  - [Interrogate](https://interrogate.readthedocs.io/) to check the documentation.

Alternatively, you may prefer to work with the right Python environment using:
```bash
$ poetry shell
$ pytest
```

### Building documentation locally
You can build the documentation on your machine using:
```
$ poetry run mkdocs serve
```
and opening the generated link using web browser.

### Code organisation

- The package code is in ``src/pyggdrasil/`` and is partitioned into subpackages.
- The unit tests are in ``tests/`` and the structure of this directory should reflect the one of the package.
- Experimental workflows are in ``workflows/``, with a description of how to set up the environment in ``workflows/README.md``

