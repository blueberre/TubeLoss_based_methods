"""
src/losses.py
"""

import torch


def tube_loss(
    y: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    delta: float,
    *,
    t: float = 0.90,
    r: float = 0.50,
) -> torch.Tensor:
    
    thresh = r * hi + (1.0 - r) * lo           # weighted midpoint

    inside     = ((y >= lo) & (y <= hi)).float()
    upper_half = (y >= thresh).float() * inside
    lower_half = (1.0 - upper_half) * inside

    base = (
        t       * torch.clamp(y - hi, min=0.0)         # above interval
        + t     * torch.clamp(lo - y, min=0.0)         # below interval
        + (1-t) * (hi - y) * upper_half                # in upper half
        + (1-t) * (y - lo) * lower_half                # in lower half
    ).mean()

    width_penalty = delta * torch.abs(hi - lo).mean()
    return base + width_penalty


def pinball_loss(
    y: torch.Tensor,
    q_hat: torch.Tensor,
    q: float,
) -> torch.Tensor:
    e = y - q_hat
    return torch.mean(torch.where(e >= 0, q * e, (q - 1.0) * e))
