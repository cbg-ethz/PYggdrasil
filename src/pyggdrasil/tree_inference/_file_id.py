"""Provides classes for naming files Tree,
Cell Simulation and MCMC run files uniquely """

import re

from typing import Union, Optional


from pyggdrasil.tree_inference import CellAttachmentStrategy, McmcConfig, TreeType


class MutationDataId:
    """Class representing a mutation data id.

    In case we want to infer a tree from real data,
    we need to provide a mutation data id.
    """

    id = str

    def __init__(self, id: str):
        """Initializes a mutation data id."""
        self.id = id


class TreeId:
    """Class representing a tree id.

    A tree id is a unique identifier for a tree.

    tree_type: TreeType - type of tree
    n_nodes: int - number of nodes in the tree
    seed: int - seed used to generate the tree, not required for star tree
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

        if tree_type == TreeType.STAR and seed is not None:
            raise AssertionError("Star tree cannot have a seed")

        self.tree_type = tree_type
        self.n_nodes = n_nodes
        self.seed = seed
        self.cell_simulation_id = cell_simulation_id

        self.id = self._create_id()

    def _create_id(self) -> str:
        """Creates a unique id for the tree,
        by concatenating the values of the attributes"""

        str_rep = "T"
        str_rep = str_rep + "_" + str(self.tree_type.value)
        str_rep = str_rep + "_" + str(self.n_nodes)

        if self.seed is not None:
            str_rep = str_rep + "_" + str(self.seed)

        if self.cell_simulation_id is not None:
            str_rep = str_rep + "_" + str(self.cell_simulation_id)

        return str_rep

    def __str__(self) -> str:
        return self.id

    @classmethod
    def from_str(cls, str_id: str):
        """Creates a tree id from a string representation of the id.

        Args:
            str_id: str
        """
        # split string by underscore and assign to attributes
        split_elements = str_id.split("_")
        seed = None
        rest_id = None
        if len(split_elements) == 3:
            _, tree_type, n_nodes = split_elements
        elif len(split_elements) == 4:
            _, tree_type, n_nodes, seed = split_elements
        elif len(split_elements) >= 5:
            _, tree_type, n_nodes, *rest = split_elements
            rest_id = "_".join(rest)
        else:
            raise AssertionError("Tree id has invalid format")

        if seed is not None:
            tree_id = TreeId(TreeType(tree_type), int(n_nodes), int(seed))
            return tree_id
        else:
            if rest_id is not None:
                # check if tree is MCMC tree
                if tree_type == TreeType.MCMC.value:
                    try:
                        tree_id = McmcTreeId.from_str(str_id)
                        return tree_id
                    except AssertionError:
                        raise AssertionError(
                            "Tree id has invalid format for an MCMC tree"
                        )

                # check if tree is Huntress tree
                elif tree_type == TreeType.HUNTRESS.value:
                    try:
                        mutation_data = CellSimulationId.from_str(rest_id)
                    except AssertionError:
                        mutation_data = MutationDataId(rest_id)

                    tree_id = TreeId(
                        TreeType(tree_type), int(n_nodes), None, mutation_data
                    )
                    return tree_id
            else:
                tree_id = TreeId(TreeType(tree_type), int(n_nodes))
                return tree_id


class McmcTreeId(TreeId):
    """Class for tree ids of trees evolved by MCMC moves under SCITE.

    MCMC move probabilities are not specified in the id!
    ID is not unique, fully reproducible only with the MCMC config.
    Assumed default values for MCMC config.
    """

    tree_type: TreeType
    n_moves: int
    n_nodes: int
    mcmc_rng_seed: int
    initial_tree_id: TreeId

    def __init__(
        self,
        n_moves: int,
        n_nodes: int,
        mcmc_rng_seed: int,
        initial_tree_id: TreeId,
        tree_type: TreeType = TreeType.MCMC,
    ):
        self.initial_tree_id = initial_tree_id
        self.n_nodes = n_nodes
        self.n_moves = n_moves
        self.mcmc_rng_seed = mcmc_rng_seed
        self.tree_type = tree_type
        super().__init__(TreeType.MCMC, n_nodes)

        self.id = self._create_id()

    def _create_id(self) -> str:
        """Creates a unique id for the tree,
        by concatenating the values of the attributes"""

        str_rep = "T"
        str_rep = str_rep + "_" + str(self.tree_type.value)
        str_rep = str_rep + "_" + str(self.n_nodes)
        str_rep = str_rep + "_" + str(self.n_moves)
        str_rep = str_rep + "_" + str(self.mcmc_rng_seed)
        str_rep = str_rep + "_o" + str(self.initial_tree_id)

        return str_rep

    def __str__(self) -> str:
        return self.id

    @classmethod
    def from_str(cls, str_id: str):
        """Creates a tree id from a string representation of the id.

        Args:
            str_id: str
        """

        # Define the regular expression pattern to match the variables
        pattern = r"T_m_(\d+)_(\d+)_(\d+)_o(T_[a-zA-Z]_\d+_\d+)"

        # Use re.findall() to extract the matched variables
        matches = re.findall(pattern, str_id)

        # The 'matches' variable now contains the extracted variables.
        # Let's unpack the matches to get individual variable values.
        if matches:
            n_nodes, n_moves, mcmc_move_seed, initial_tree_id = matches[0]

            tree_id = McmcTreeId(
                int(n_moves),
                int(n_nodes),
                int(mcmc_move_seed),
                TreeId.from_str(initial_tree_id),
            )

            return tree_id
        else:
            raise AssertionError("MCMC tree id has invalid format")


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
        super().__init__(id="")
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

        str_rep = "CS_" + str(self.seed)
        str_rep = str_rep + "-" + str(self.tree_id)
        str_rep = str_rep + "-" + str(self.n_cells)
        str_rep = str_rep + f"_{self.fpr:.3}"
        str_rep = str_rep + f"_{self.fnr:.3}"
        str_rep = str_rep + f"_{self.na_rate:.3}"
        str_rep = str_rep + "_" + str(self.observe_homozygous).lower()[0]

        if self.strategy is CellAttachmentStrategy.UNIFORM_EXCLUDE_ROOT:
            str_rep = str_rep + "_" + "UXR"
        elif self.strategy is CellAttachmentStrategy.UNIFORM_INCLUDE_ROOT:
            str_rep = str_rep + "_" + "UIR"

        return str_rep

    def __str__(self) -> str:
        return self.id

    @classmethod
    def from_str(cls, str_id: str):
        """Creates a CellSimulation id from a string representation of the id.
        Args:
            str_id: str

        Raises:
            AssertionError if the string representation is not valid
        """
        # split string by underscore and assign to attributes
        # CS_1-T_d_10-100_0.01_0.01_0.01_true_UXR
        parts = str_id.split("-")
        cs_part1 = parts[0]
        tree_id = parts[1]
        cs_part2 = "-".join(parts[2:])

        # check prefix and postfix
        assert cs_part1.startswith("CS_")
        assert tree_id.startswith("T_")
        assert cs_part2.endswith("UXR") or cs_part2.endswith("UIR")
        # split cs_part2 by underscore and assign to attributes
        cs_part2 = cs_part2.split("_")
        seed = int(cs_part1.split("_")[1])
        n_cells = int(cs_part2[0])
        # check that cs_part[1] is not 0.00 but 0.0
        if cs_part2[1] == "0.00" or cs_part2[2] == "0.00" or cs_part2[3] == "0.00":
            raise AssertionError(
                "String representations of floats must not have trailing zeros"
            )
        fpr = float(cs_part2[1])
        fnr = float(cs_part2[2])
        na_rate = float(cs_part2[3])
        observe_homozygous = cs_part2[4] == "t"
        strategy = cs_part2[5]
        if strategy == "UXR":
            strategy = CellAttachmentStrategy.UNIFORM_EXCLUDE_ROOT
        elif strategy == "UIR":
            strategy = CellAttachmentStrategy.UNIFORM_INCLUDE_ROOT
        else:
            raise ValueError("Invalid strategy")
        # create tree id
        tree_id = TreeId.from_str(tree_id)

        return cls(
            seed,
            tree_id,
            n_cells,
            fpr,
            fnr,
            na_rate,
            observe_homozygous,
            strategy,
        )


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

        str_rep = "MCMC"
        str_rep = str_rep + "_" + str(self.seed)
        str_rep = str_rep + "-" + str(self.data)
        str_rep = str_rep + "-i" + str(self.init_tree_id)
        str_rep = str_rep + "-" + str(self.mcmc_config.id())

        return str_rep

    def __str__(self) -> str:
        return self.id
