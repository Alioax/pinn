# -*- coding: utf-8 -*-
"""DeepONet parametric operator (branch + trunk).

branch_input: dimensionless zone CFL values (CFL_i = u_i T_max / L).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.zone_velocity import zone_index_xstar


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

    def forward(self, x_star, t_star, branch_input):
        pts = torch.cat([x_star, t_star], dim=1)
        b_vec = self.branch(branch_input)
        t_vec = self.trunk(pts)
        return torch.sigmoid((b_vec * t_vec).sum(dim=-1, keepdim=True))


class DeepONetZoneTrunks(nn.Module):
    """Shared branch; one trunk per zone selected by zone_index(x*)."""

    def __init__(self, branch_arch, trunk_arch, activation, *, n_zones: int = 4):
        super().__init__()
        self.n_zones = n_zones
        self.branch = _build_mlp(branch_arch, activation)
        self.trunks = nn.ModuleList(
            [_build_mlp(trunk_arch, activation) for _ in range(n_zones)]
        )

        gain = nn.init.calculate_gain("tanh")
        _init_mlp_gain(self.branch, gain)
        for trunk in self.trunks:
            _init_mlp_gain(trunk, gain)

    def forward(self, x_star, t_star, branch_input):
        pts = torch.cat([x_star, t_star], dim=1)
        b_vec = self.branch(branch_input)
        stacked = torch.stack([trunk(pts) for trunk in self.trunks], dim=1)
        zidx = zone_index_xstar(x_star).long().view(-1)
        row_idx = torch.arange(stacked.size(0), device=stacked.device)
        t_vec = stacked[row_idx, zidx]
        return torch.sigmoid((b_vec * t_vec).sum(dim=-1, keepdim=True))


def build_deeponet(
    trunk_mode: str,
    branch_arch: list[int],
    trunk_arch: list[int],
    activation: type[nn.Module],
) -> nn.Module:
    if trunk_mode == "single":
        return DeepONetParametric(branch_arch, trunk_arch, activation)
    if trunk_mode == "zone":
        return DeepONetZoneTrunks(branch_arch, trunk_arch, activation)
    raise ValueError(f"Unknown trunk_mode: {trunk_mode!r}")
