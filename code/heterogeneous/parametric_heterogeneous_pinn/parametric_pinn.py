# -*- coding: utf-8 -*-
"""Parametric PINN for heterogeneous four-zone transport.

Input: (x*, t*, CFL_1..CFL_4) concatenated into one MLP.
Forward signature matches the PINO operator: (x_star, t_star, branch_input).
"""

from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_ARCHITECTURE = [6, 36, 36, 36, 36, 1]


def _build_mlp(arch: list[int], activation: type[nn.Module]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(arch) - 1):
        in_f, out_f = arch[i], arch[i + 1]
        layers.append(nn.Linear(in_f, out_f))
        if i < len(arch) - 2:
            layers.append(activation())
        else:
            layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


def _init_mlp_gain(mlp: nn.Sequential, gain: float) -> None:
    for layer in mlp:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class ParametricPINN(nn.Module):
    """Maps (x*, t*, CFL_1..CFL_4) -> C* in [0, 1]."""

    def __init__(
        self,
        architecture: list[int] | None = None,
        activation: type[nn.Module] = nn.Tanh,
    ):
        super().__init__()
        arch = list(architecture) if architecture is not None else list(DEFAULT_ARCHITECTURE)
        self.architecture = arch
        self.net = _build_mlp(arch, activation)
        gain = nn.init.calculate_gain("tanh")
        _init_mlp_gain(self.net, gain)

    def forward(
        self,
        x_star: torch.Tensor,
        t_star: torch.Tensor,
        branch_input: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat([x_star, t_star, branch_input], dim=1)
        return self.net(inputs)


def build_parametric_pinn(
    architecture: list[int] | None = None,
    activation: type[nn.Module] = nn.Tanh,
) -> ParametricPINN:
    return ParametricPINN(architecture=architecture, activation=activation)
