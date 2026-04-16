"""
neuro/training/losses.py
------------------------
Triplet Loss для обучения Siamese-энкодера (PyTorch).

Loss = max(0, dist(anchor, positive) - dist(anchor, negative) + margin)
"""

import torch


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """
    Triplet Margin Loss.

    Для L2-нормализованных векторов:
        dist²(a, b) = 2 - 2·dot(a, b) = sum((a - b)²)
    """
    pos_dist = torch.sum((anchor - positive) ** 2, dim=-1)
    neg_dist = torch.sum((anchor - negative) ** 2, dim=-1)
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
    return loss.mean()
