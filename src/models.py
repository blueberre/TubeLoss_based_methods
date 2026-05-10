"""
src/models.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TubePINet(nn.Module):
    def __init__(self, in_dim: int, hid: int = 32, n_layers: int = 1):
        super().__init__()
        layers = [nn.Linear(in_dim, hid), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hid, hid), nn.ReLU()]
        layers += [nn.Linear(hid, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        o  = self.net(x)
        lo = o[:, 0]
        hi = lo + F.softplus(o[:, 1])
        return lo, hi

class QuantileNet(nn.Module):
    def __init__(self, in_dim: int, hid: int = 32, n_layers: int = 1):
        super().__init__()
        layers = [nn.Linear(in_dim, hid), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hid, hid), nn.ReLU()]
        layers += [nn.Linear(hid, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)