"""Provides classes for naming files Tree,
Cell Simulation and MCMC run files uniquely """

from enum import Enum
from typing import Union, Optional


from pyggdrasil.tree_inference import CellAttachmentStrategy, McmcConfig


class TreeType(Enum):
    """Enum representing valid tree types implemented in pyggdrasil.

    Allowed values:
      - RANDOM (random tree)
      - STAR (star tree)
      - DEEP (deep tree)
      - HUNTRESS (Huntress tree) - inferred from real / cell simulation data
    """

    RANDOM = "r"
    STAR = "s"
    DEEP = "d"
    HUNTRESS = "h"


class MutationDataId:
    """Class representing a mutation data id.

    In case we want to infer a tree from real data,
    we need to provide a mutation data id.
    """

    id = str


class TreeId:
    """Class representing a tree id.

    A tree id is a unique identifier for a tree.

    tree_type: TreeType - type of tree
    n_nodes: int - number of nodes in the tree
    seed: int - seed used to generate the tree
    cell_simulation_id: str - if the tree was generated from a cell
                                simulation, i.e. Huntress
    """

    tree_type: TreeType
    n_nodes: int
    seed: Optional[int]
    cell_simulation_id: Union[MutationDataId, None]

    id: str

    def __init__(
        self,
        tree_type: TreeType,
        n_nodes: int,
        seed: Optional[int] = None,
        cell_simulation_id: Optional[MutationDataId] = None,
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
        """Creates a unique id for the tree,
        by concatenating the values of the attributes"""

        self.id = "T"
        self.id = self.id + "_" + str(self.tree_type.value)
        self.id = self.id + "_" + str(self.n_nodes)

        if self.seed is not None:
            self.id = self.id + "_" + str(self.seed)

        if self.cell_simulation_id is not None:
            self.id = self.id + "_" + str(self.cell_simulation_id)

        return self.id

    def __str__(self) -> str:
        return self.id


class CellSimulationId(MutationDataId):
    """Class representing a cell simulation id.

    Note: that the Tree_id contains the number of mutations i.e. nodes-1"""

    seed: int
    tree_id: TreeId
    n_cells: int
    fpr: float
    fnr: float
    na_rate: float
    observe_homozygous: bool
    strategy: CellAttachmentStrategy

    id: str

    def __init__(
        self,
        seed: int,
        tree_id: TreeId,
        n_cells: int,
        fpr: float,
        fnr: float,
        na_rate: float,
        observe_homozygous: bool,
        strategy: CellAttachmentStrategy,
    ):
        """Initializes a cell simulation id."""
        self.seed = seed
        self.tree_id = tree_id
        self.n_cells = n_cells
        self.fpr = fpr
        self.fnr = fnr
        self.na_rate = na_rate
        self.observe_homozygous = observe_homozygous
        self.strategy = strategy

        self.id = self._create_id()

    def _create_id(self) -> str:
        """Creates a unique id for the cell simulation,
        by concatenating the values of the attributes"""

        self.id = "CS_" + str(self.seed)
        self.id = self.id + "-" + str(self.tree_id)
        self.id = self.id + "-" + str(self.n_cells)
        self.id = self.id + f"_{self.fpr:.3}"
        self.id = self.id + f"_{self.fnr:.3}"
        self.id = self.id + f"_{self.na_rate:.3}"
        self.id = self.id + "_" + str(self.observe_homozygous).lower()[0]

        if self.strategy is CellAttachmentStrategy.UNIFORM_EXCLUDE_ROOT:
            self.id = self.id + "_" + "UXR"
        elif self.strategy is CellAttachmentStrategy.UNIFORM_INCLUDE_ROOT:
            self.id = self.id + "_" + "UIR"

        return self.id

    def __str__(self) -> str:
        return self.id


class McmcRunId:
    """Class representing an MCMC run id."""

    seed: int
    data: MutationDataId
    init_tree_id: TreeId
    mcmc_config_id: McmcConfig

    id: str

    def __init__(
        self,
        seed: int,
        data: Union[CellSimulationId, MutationDataId],
        init_tree_id: TreeId,
        mcmc_config: McmcConfig,
    ):
        """Initializes an MCMC run id.

        Args:
            seed: int
            data: Union[CellSimulationId, MutationDataId]
            init_tree_id: TreeId
            mcmc_config: McmcConfig
        """

        self.seed = seed
        self.data = data
        self.init_tree_id = init_tree_id
        self.mcmc_config = mcmc_config

        self.id = self._create_id()

    def _create_id(self) -> str:
        """Creates a unique id for the MCMC run,
        by concatenating the values of the attributes"""

        self.id = "MCMC"
        self.id = self.id + "_" + str(self.seed)
        self.id = self.id + "-" + str(self.data)
        self.id = self.id + "-i" + str(self.init_tree_id)
        self.id = self.id + "-" + str(self.mcmc_config.id())

        return self.id

    def __str__(self) -> str:
        return self.id
