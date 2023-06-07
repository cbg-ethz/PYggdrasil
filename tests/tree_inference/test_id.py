"""Tests for id creation for file naming."""

import pytest


from pyggdrasil.tree_inference import (
    CellAttachmentStrategy,
    McmcConfig,
    MoveProbConfig,
    TreeId,
    CellSimulationId,
    TreeType,
    McmcRunId,
)


@pytest.fixture
def mcmc_config() -> McmcConfig:
    """Returns an MCMC config id."""

    move_probs = MoveProbConfig()
    mcmc_config = McmcConfig(move_probs=move_probs, n_samples=1000)

    return mcmc_config


@pytest.fixture
def tree_id() -> TreeId:
    """Returns a tree id."""

    tree_type = TreeType.DEEP
    n_nodes = 10
    seed = 123

    return TreeId(tree_type, n_nodes, seed)


@pytest.fixture
def cell_simulation_id(tree_id) -> CellSimulationId:
    """Returns a cell simulation id."""

    seed = 42
    n_cells = 1000
    fpr = 0.01
    fnr = 0.02
    na_rate = 0.03
    observe_homozygous = True
    strategy = CellAttachmentStrategy.UNIFORM_EXCLUDE_ROOT

    return CellSimulationId(
        seed, tree_id, n_cells, fpr, fnr, na_rate, observe_homozygous, strategy
    )


@pytest.fixture
def mcmc_run_id(cell_simulation_id, tree_id, mcmc_config) -> McmcRunId:
    """Returns an MCMC run id."""

    seed = 42
    data = cell_simulation_id
    init_tree_id = tree_id
    mcmc_config = mcmc_config

    return McmcRunId(seed, data, init_tree_id, mcmc_config)


def test_tree_id(tree_id) -> None:
    """Tests for tree id."""
    assert str(tree_id) == "T_d_10_123"


def test_cell_simulation_id(cell_simulation_id) -> None:
    """Tests for cell simulation id."""
    assert str(cell_simulation_id) == "CS_42-T_d_10_123-1000_0.01_0.02_0.03_t_UXR"


def test_mcmc_run_id(mcmc_run_id) -> None:
    """Tests for MCMC run id."""

    expected_id = "MCMC_42-CS_42-T_d_10_123-1000_0.01_0.02_0.03_t_UXR-iT_d_10_123-MC_"
    expected_id = expected_id + "1.24e-06_0.097_1000_0_1-MPC_0.1_0.65_0.25"

    assert str(mcmc_run_id) == expected_id


def test_tree_id_from_str(tree_id) -> None:
    """Tests for tree id."""
    test_id = TreeId.from_str(str(tree_id))

    assert test_id.tree_type == tree_id.tree_type
    assert test_id.n_nodes == tree_id.n_nodes
    assert test_id.seed == tree_id.seed
