# -*- coding: utf-8 -*-
"""DeepONet parametric operator (branch + trunk).

branch_input: dimensionless zone CFL values (CFL_i = u_i T_max / L).
"""

import torch
import torch.nn as nn


class DeepONetParametric(nn.Module):
    def __init__(self, branch_arch, trunk_arch, activation):
        super().__init__()
        branch_layers = []
        for i in range(len(branch_arch) - 1):
            in_f, out_f = branch_arch[i], branch_arch[i + 1]
            branch_layers.append(nn.Linear(in_f, out_f))
            if i < len(branch_arch) - 2:
                branch_layers.append(activation())
        self.branch = nn.Sequential(*branch_layers)

        trunk_layers = []
        for i in range(len(trunk_arch) - 1):
            in_f, out_f = trunk_arch[i], trunk_arch[i + 1]
            trunk_layers.append(nn.Linear(in_f, out_f))
            if i < len(trunk_arch) - 2:
                trunk_layers.append(activation())
        self.trunk = nn.Sequential(*trunk_layers)

        gain = nn.init.calculate_gain("tanh")
        for mod in (self.branch, self.trunk):
            for layer in mod:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=gain)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x_star, t_star, branch_input):
        pts = torch.cat([x_star, t_star], dim=1)
        b_vec = self.branch(branch_input)
        t_vec = self.trunk(pts)
        return torch.sigmoid((b_vec * t_vec).sum(dim=-1, keepdim=True))
