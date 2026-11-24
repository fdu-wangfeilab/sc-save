"""Compatibility shim re-exporting Flow Matching components from ``sc_save.nn.flow_matching``."""

import torch

from sc_save.nn.flow_matching import (
    FM,
    CondOTPath,
    FlowMatcher,
    MLPTimeStep,
    OTPlanSampler,
    SimpleSDESolver,
    VEPath,
    VPPath,
)


def pad_t_like_x(t: torch.Tensor | float | int, x: torch.Tensor) -> torch.Tensor | float | int:
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


__all__ = [
    "CondOTPath",
    "FM",
    "FlowMatcher",
    "MLPTimeStep",
    "OTPlanSampler",
    "SimpleSDESolver",
    "VEPath",
    "VPPath",
    "pad_t_like_x",
]
