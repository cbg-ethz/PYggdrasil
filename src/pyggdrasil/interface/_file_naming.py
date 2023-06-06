"""Provides classes for naming files Tree,
Cell Simulation and MCMC run files uniquely """

from enum import Enum
from typing import Union

from pyggdrasil.tree_inference import CellAttachmentStrategy, McmcConfig


class TreeType(Enum):
    """Enum representing valid tree types implemented in pyggdrasil.

    Allowed values:
      - RANDOM (random tree)
      - STAR (star tree)
      - DEEP (deep tree)
      - HUNTRESS (Huntress tree) - inferred from real / cell simulation data
    """

    RANDOM = "R"
    STAR = "S"
    DEEP = "D"
    HUNTRESS = "H"


class MutationDataId:
    """Class representing a mutation data id.

    In case we want to infer a tree from real data,
    we need to provide a mutation data id.
    """

    id: str

    raise NotImplementedError


class CellSimulationId:
    """Class representing a cell simulation id."""

    seed: int
    tree_id: str
    n_cells: int
    n_mutations: int
    fpr: float
    fnr: float
    na_rate: float
    observe_homozygous: bool
    strategy: CellAttachmentStrategy

    id: str

    def __init__(
        self,
        seed: int,
        tree_id: str,
        n_cells: int,
        n_mutations: int,
        fpr: float,
        fnr: float,
        na_rate: float,
        observe_homozygous: bool,
        strategy: CellAttachmentStrategy,
    ):
        self.seed = seed
        self.tree_id = tree_id
        self.n_cells = n_cells
        self.n_mutations = n_mutations
        self.fpr = fpr
        self.fnr = fnr
        self.na_rate = na_rate
        self.observe_homozygous = observe_homozygous
        self.strategy = strategy

        self.id = self._create_id()

    def _create_id(self) -> str:
        pass


class TreeId:
    """Class representing a tree id.

    A tree id is a unique identifier for a tree.

    tree_type: TreeType - type of tree
    n_nodes: int - number of nodes in the tree
    seed: int - seed used to generate the tree
    cell_simulation_id: str - id of the cell simulation used to generate the tree
    """

    tree_type: TreeType
    n_nodes: int
    seed: int
    cell_simulation_id: CellSimulationId

    id: str

    def __init__(
        self,
        tree_type: TreeType,
        n_nodes: int,
        seed: int = None,
        cell_simulation_id: CellSimulationId = None,
    ):
        """Initializes a tree id.

        Args:
            tree_type: TreeType
            n_nodes: int
            seed: int
        """

        self.tree_type = tree_type
        self.n_nodes = n_nodes
        self.seed = seed
        self.cell_simulation_id = cell_simulation_id

        self.id = self._create_id()

    def _create_id(self) -> str:
        pass


class McmcRunId:
    """Class representing an MCMC run id."""

    seed: int
    data: Union[CellSimulationId, MutationDataId]
    init_tree_id: TreeId
    mcmc_config_id: McmcConfig

    id: str

    def __init__(
        self,
        seed: int,
        data: Union[CellSimulationId, MutationDataId],
        init_tree_id: TreeId,
        mcmc_config_id: McmcConfig,
    ):
        """Initializes an MCMC run id.

        Args:
            seed: int
            data: Union[CellSimulationId, MutationDataId]
            init_tree_id: TreeId
            mcmc_config_id: McmcConfig
        """

        self.seed = seed
        self.data = data
        self.init_tree_id = init_tree_id
        self.mcmc_config_id = mcmc_config_id

        self.id = self._create_id()

    def _create_id(self) -> str:
        pass
