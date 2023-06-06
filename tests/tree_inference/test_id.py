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

    tree_type = TreeType.STAR
    n_nodes = 10
    seed = 123

    return TreeId(tree_type, n_nodes, seed)


@pytest.fixture
def cell_simulation_id(tree_id) -> CellSimulationId:
    """Returns a cell simulation id."""

    seed = 42
    n_cells = 1000
    n_mutations = 100
    fpr = 0.01
    fnr = 0.02
    na_rate = 0.03
    observe_homozygous = True
    strategy = CellAttachmentStrategy.UNIFORM_EXCLUDE_ROOT

    return CellSimulationId(
        seed,
        tree_id,
        n_cells,
        n_mutations,
        fpr,
        fnr,
        na_rate,
        observe_homozygous,
        strategy,
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
    assert str(tree_id) == "T_s_10_123"


def test_cell_simulation_id(cell_simulation_id) -> None:
    """Tests for cell simulation id."""
    assert str(cell_simulation_id) == "CS_42-T_s_10_123-1000_100_0.01_0.02_0.03_t_UIR"


def test_mcmc_run_id(mcmc_run_id) -> None:
    """Tests for MCMC run id."""

    expected_id = (
        "MCMC_42_CS_42-T_s_10_123-1000_100_0.01_0.02_0.03_t_UIR-iT_s_10_123-MC_"
    )
    expected_id = expected_id + "1.24e-06_0.097_1000_0_1_MPC_0.1_0.65_0.25"

    assert str(mcmc_run_id) == expected_id