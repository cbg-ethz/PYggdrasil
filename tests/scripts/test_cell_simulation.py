"""Tests the cell_simulation script."""
# cell_simulation.py

import os

import numpy as np
import pytest
import json


IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


def mock_data(tmp_path):
    """Test plot_tree. - check for output."""

    save_name = "seed_42_n_trees_3_n_cells_100_n_mutations_8_alpha_0.01_"
    save_name = save_name + "beta_0.02_na_rate_0.01_observe_homozygous_True_"
    save_name = save_name + "strategy_UNIFORM_INCLUDE_ROOT_tree_3.json"

    save_dir = tmp_path / "mock"

    # run cell_simulation.py with the following arguments
    command = "poetry run python scripts/cell_simulation.py --seed 42 "
    command = (
        command + f" --out_dir {save_dir} --n_trees 3 --n_cells 100 --n_mutations 8 "
    )
    command = (
        command
        + "--strategy UNIFORM_INCLUDE_ROOT --alpha 0.01 --beta 0.02 --na_rate 0.01 "
    )
    command = command + "--observe_homozygous True --verbose"

    os.system(command)

    file = save_dir / save_name

    txt = file.read_text()

    # load json file
    data = json.loads(txt)
    return data


@pytest.mark.filterwarnings("ignore::DeprecationWarning:")
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Skipping this test in GitHub Actions.")
# contents of test_image.py
def test_dims_mock_data(tmp_path):
    """Test that the dimensions of the mock data are correct."""

    data = mock_data(tmp_path)

    # check that the dimensions of the data are correct
    assert np.array(data["adjacency_matrix"]).shape == (8 + 1, 8 + 1)
    assert np.array(data["noisy_mutation_mat"]).shape == (
        8 + 1,
        100,
    )  # TODO: to be altered if we truncate the matrix
    assert np.array(data["perfect_mutation_mat"]).shape == (
        8 + 1,
        100,
    )  # TODO: to be altered if we truncate the matrix
