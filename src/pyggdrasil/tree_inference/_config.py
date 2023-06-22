"""Config classes for MCMC sampler,
 cell simulation and tree inference, distance calculation."""

from enum import Enum
from pydantic import BaseModel, confloat, conint, root_validator


class MoveProbConfig(BaseModel):
    """Move probabilities for MCMC sampler."""

    prune_and_reattach: confloat(gt=0, lt=1) = 0.1  # type: ignore
    swap_node_labels: confloat(gt=0, lt=1) = 0.65  # type: ignore
    swap_subtrees: confloat(gt=0, lt=1) = 0.25  # type: ignore

    @root_validator
    @classmethod
    def move_prob_validator(cls, field_values):
        """Probabilities sum to 1."""
        total = (
            field_values["prune_and_reattach"]
            + field_values["swap_node_labels"]
            + field_values["swap_subtrees"]
        )
        # near enough to 1
        if abs(total - 1) > 1e-6:
            raise ValueError("Move probabilities must sum to 1")
        return field_values

    def id(self) -> str:
        """String representation of move probabilities."""
        str_rep = "MPC_" + f"{self.prune_and_reattach:.2}"
        str_rep = str_rep + f"_{self.swap_node_labels:.2}"
        str_rep = str_rep + f"_{self.swap_subtrees:.2}"
        return str_rep


class MoveProbConfigOptions(Enum):
    """Move probability configurations.

    Implements configurations are DEFAULT and OPTIMAL from
    SCITE paper, supplement p.15.

    Default values:
        prune_and_reattach=0.1,
        swap_node_labels=0.65,
        swap_subtrees=0.25

    Optimal values:
        prune_and_reattach=0.55,
        swap_node_labels=0.4,
        swap_subtrees=0.05

        (`Optimal values find ML tree up to 2 or 3 times faster`)
    """

    DEFAULT = MoveProbConfig()
    OPTIMAL = MoveProbConfig(
        prune_and_reattach=0.55, swap_node_labels=0.4, swap_subtrees=0.05
    )


class McmcConfig(BaseModel):
    """Config for MCMC sampler."""

    move_probs: MoveProbConfig = MoveProbConfigOptions.DEFAULT.value
    fpr: confloat(gt=0, lt=1) = 1.24e-06  # type: ignore
    fnr: confloat(gt=0, lt=1) = 0.097  # type: ignore
    n_samples: conint(gt=-1) = 12000  # type: ignore
    burn_in: conint(gt=-1) = 0  # type: ignore
    thinning: conint(gt=0) = 1  # type: ignore

    def id(self) -> str:
        """String representation of MCMC config."""
        str_rep = "MC"
        str_rep = str_rep + f"_{self.fpr:.3}"
        str_rep = str_rep + f"_{self.fnr:.3}"
        str_rep = str_rep + "_" + str(self.n_samples)
        str_rep = str_rep + "_" + str(self.burn_in)
        str_rep = str_rep + "_" + str(self.thinning)
        str_rep = str_rep + "-" + str(self.move_probs.id())
        return str_rep


class McmcConfigOptions(Enum):
    """MCMC run configurations.

    Implements configurations are DEFAULT and TEST.

    DEFAULT:
        move_probs=MoveProbConfigOptions.DEFAULT
        fpr=1.24e-06,
        fnr=0.097,
        n_samples=12000,
        burn_in=0,
        thinning=1

    TEST:
        move_probs=MoveProbConfigOptions.DEFAULT
        fpr=1.24e-06,
        fnr=0.097,
        n_samples=100,
        burn_in=0,
        thinning=1

    """

    DEFAULT = (McmcConfig(),)
    TEST = McmcConfig(n_samples=1000, fpr=0.4, fnr=0.4)
