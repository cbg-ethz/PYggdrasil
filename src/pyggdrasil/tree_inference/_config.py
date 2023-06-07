"""Config classes for MCMC sampler,
 cell simulation and tree inference, distance calculation."""

from pydantic import BaseModel, validator, confloat, conint


class MoveProbConfig(BaseModel):
    """Move probabilities for MCMC sampler."""

    prune_and_reattach: confloat(gt=0, lt=1) = 0.1  # type: ignore
    swap_node_labels: confloat(gt=0, lt=1) = 0.65  # type: ignore
    swap_subtrees: confloat(gt=0, lt=1) = 0.25  # type: ignore

    @validator("prune_and_reattach", "swap_node_labels", "swap_subtrees")
    def move_prob_validator(cls, v):
        """Probabilities sum to 1."""
        total = sum(v.values())
        # near enough to 1
        if abs(total - 1) > 1e-6:
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
    fpr: confloat(gt=0, lt=1) = 1.24e-06  # type: ignore
    fnr: confloat(gt=0, lt=1) = 0.097  # type: ignore
    n_samples: conint(gt=-1)  # type: ignore
    burn_in: conint(gt=-1) = 0  # type: ignore
    thinning: conint(gt=0) = 1  # type: ignore

    def id(self) -> str:
        """String representation of MCMC config."""
        str_rep = "MC"
        str_rep = str_rep + "_" + str(self.fpr)
        str_rep = str_rep + "_" + str(self.fnr)
        str_rep = str_rep + "_" + str(self.n_samples)
        str_rep = str_rep + "_" + str(self.burn_in)
        str_rep = str_rep + "_" + str(self.thinning)
        str_rep = str_rep + "-" + str(self.move_probs.id())
        return str_rep
