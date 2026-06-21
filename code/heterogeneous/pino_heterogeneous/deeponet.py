# -*- coding: utf-8 -*-
"""DeepONet parametric operator (branch + trunk).

branch_input: dimensionless zone CFL values (CFL_i = u_i T_max / L).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _build_mlp(arch: list[int], activation: type[nn.Module]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(arch) - 1):
        in_f, out_f = arch[i], arch[i + 1]
        layers.append(nn.Linear(in_f, out_f))
        if i < len(arch) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


def _init_mlp_gain(mlp: nn.Sequential, gain: float) -> None:
    for layer in mlp:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class DeepONetParametric(nn.Module):
    def __init__(self, branch_arch, trunk_arch, activation):
        super().__init__()
        self.branch = _build_mlp(branch_arch, activation)
        self.trunk = _build_mlp(trunk_arch, activation)

        gain = nn.init.calculate_gain("tanh")
        _init_mlp_gain(self.branch, gain)
        _init_mlp_gain(self.trunk, gain)

    def forward(
        self,
        x_star,
        t_star,
        branch_input,
        case_idx=None,
        *,
        tau: float = 0.05,
        hard_bc: bool = False,
    ):
        if case_idx is None:
            b_vec = self.branch(branch_input)
        else:
            b_vec = self.branch(branch_input)[case_idx]
        pts = torch.cat([x_star, t_star], dim=1)
        t_vec = self.trunk(pts)
        n_raw = (b_vec * t_vec).sum(dim=-1, keepdim=True)
        if not hard_bc:
            return torch.sigmoid(n_raw)
        r = 1.0 - torch.exp(-t_star / tau)
        a = (1.0 - x_star) * r
        b = x_star * (1.0 - x_star) * r
        return a + b * n_raw


def build_deeponet(
    branch_arch: list[int],
    trunk_arch: list[int],
    activation: type[nn.Module],
) -> nn.Module:
    latent = branch_arch[-1]
    if trunk_arch[-1] != latent:
        raise ValueError(
            f"Branch and trunk latent dims must match for the dot product: "
            f"branch output {latent} != trunk output {trunk_arch[-1]} "
            f"(branch={branch_arch}, trunk={trunk_arch})"
        )
    return DeepONetParametric(branch_arch, trunk_arch, activation)
