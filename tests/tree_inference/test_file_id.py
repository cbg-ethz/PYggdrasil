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
    ErrorCombinations,
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
def cell_simulation_id_scientific_notation(tree_id) -> CellSimulationId:
    """Returns a cell simulation id."""

    seed = 42
    n_cells = 1000
    fpr = 1.24e-06
    fnr = 1e-01
    na_rate = 1e-06
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
    test_id: TreeId = TreeId.from_str(str(tree_id))  # type: ignore

    assert test_id.tree_type == tree_id.tree_type
    assert test_id.n_nodes == tree_id.n_nodes
    assert test_id.seed == tree_id.seed


def test_cell_simulation_id_from_str(cell_simulation_id) -> None:
    """Tests for cell simulation id."""
    csi = CellSimulationId.from_str(str(cell_simulation_id))
    assert csi.seed == cell_simulation_id.seed
    assert csi.tree_id.tree_type == cell_simulation_id.tree_id.tree_type
    assert csi.tree_id.n_nodes == cell_simulation_id.tree_id.n_nodes
    assert csi.tree_id.seed == cell_simulation_id.tree_id.seed
    assert csi.n_cells == cell_simulation_id.n_cells
    assert csi.fpr == cell_simulation_id.fpr
    assert csi.fnr == cell_simulation_id.fnr
    assert csi.na_rate == cell_simulation_id.na_rate
    assert csi.observe_homozygous == cell_simulation_id.observe_homozygous
    assert csi.strategy == cell_simulation_id.strategy


def test_cell_simulation_id_from_str_scientific_notation(
    cell_simulation_id_scientific_notation,
) -> None:
    """Tests for cell simulation id."""
    csi = CellSimulationId.from_str(str(cell_simulation_id_scientific_notation))
    assert csi.seed == cell_simulation_id_scientific_notation.seed
    assert (
        csi.tree_id.tree_type
        == cell_simulation_id_scientific_notation.tree_id.tree_type
    )
    assert csi.tree_id.n_nodes == cell_simulation_id_scientific_notation.tree_id.n_nodes
    assert csi.tree_id.seed == cell_simulation_id_scientific_notation.tree_id.seed
    assert csi.n_cells == cell_simulation_id_scientific_notation.n_cells
    assert csi.fpr == cell_simulation_id_scientific_notation.fpr
    assert csi.fnr == cell_simulation_id_scientific_notation.fnr
    assert csi.na_rate == cell_simulation_id_scientific_notation.na_rate
    assert (
        csi.observe_homozygous
        == cell_simulation_id_scientific_notation.observe_homozygous
    )
    assert csi.strategy == cell_simulation_id_scientific_notation.strategy


def test_huntrees_tree_id_from_str() -> None:
    """Tests for tree id."""

    str = "T_h_6_CS_42-T_r_6_42-200_0.1_0.1_0.0_f_UXR"

    test_id: TreeId = TreeId.from_str(str)  # type: ignore

    assert test_id.tree_type == TreeType("h")
    assert test_id.n_nodes == 6


def test_mcmc_tree_id_from_str() -> None:
    """Tests for tree id."""

    test_str = "iT_m_6_5_99_oT_r_6_42"

    test_id: TreeId = TreeId.from_str(test_str)  # type: ignore

    assert test_id.tree_type == TreeType.MCMC
    assert test_id.n_nodes == 6


def test_mcmc_id_from_string_manual() -> None:
    test_str = (
        "MCMC_35-CS_42-T_r_31_42-1000_1e-06_1e-06_0.0_f"
        "_UXR-iT_r_31_35-MC_1e-06_1e-06_2000_0_1-MPC_0.1_0.65_0.25"
    )

    true_tree_id = TreeId.from_str("T_r_31_42")

    cs_id = CellSimulationId(
        42,
        true_tree_id,  # type: ignore
        1000,
        1e-06,
        1e-06,
        0.0,
        False,
        CellAttachmentStrategy.UNIFORM_EXCLUDE_ROOT,
    )

    init_tree_id = TreeId.from_str("T_r_31_35")

    move_probs = MoveProbConfig()
    err = ErrorCombinations.IDEAL.value
    mcmc_config = McmcConfig(
        fnr=err.fnr, fpr=err.fpr, move_probs=move_probs, n_samples=2000
    )

    test_id: McmcRunId = McmcRunId(35, cs_id, init_tree_id, mcmc_config)  # type: ignore

    assert str(test_id) == test_str
