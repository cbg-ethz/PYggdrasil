"""Config classes for MCMC sampler,
 cell simulation and tree inference, distance calculation."""

from pydantic import BaseModel, validator


class MoveProbConfig(BaseModel):
    """Move probabilities for MCMC sampler."""

    prune_and_reattach: float = 0.1
    swap_node_labels: float = 0.65
    swap_subtrees: float = 0.25

    @validator("prune_and_reattach", "swap_node_labels", "swap_subtrees")
    def move_prob_validator(cls, v):
        """Probabilities sum to 1."""
        total = sum(v.values())
        if total != 1:
            raise ValueError("Move probabilities must sum to 1")
        return v

    def id(self) -> str:
        """String representation of move probabilities."""
        str_rep = "MPC_" + str(self.prune_and_reattach)
        str_rep = str_rep + "_" + str(self.swap_node_labels)
        str_rep = str_rep + "_" + str(self.swap_subtrees)
        return str_rep


class McmcConfig(BaseModel):
    """Config for MCMC sampler."""

    move_probs: MoveProbConfig
    fpr: float = 1.24e-06
    fnr: float = 0.097
    n_samples: int
    burn_in: int = 0
    thinning: int = 1

    @validator("fpr")
    def fpr_validator(cls, v):
        """Validate move probabilities."""
        if v <= 0 or v > 1:
            raise ValueError("fpr must be between 0 and 1")
        return v

    @validator("fnr")
    def fnr_validator(cls, v):
        """Validate move probabilities."""
        if v <= 0 or v > 1:
            raise ValueError("fnr must be between 0 and 1")
        return v

    def id(self) -> str:
        """String representation of MCMC config."""
        str_rep = "MC"
        str_rep = str_rep + "_" + str(self.fpr)
        str_rep = str_rep + "_" + str(self.fnr)
        str_rep = str_rep + "_" + str(self.n_samples)
        str_rep = str_rep + "_" + str(self.burn_in)
        str_rep = str_rep + "_" + str(self.thinning)
        str_rep = str_rep + "_" + str(self.move_probs.id())
        return str_rep
