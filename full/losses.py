"""
Losses: BCE and optional MacroSoftF1. build_loss(mode="bce" | "bce_softf1").
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MacroSoftF1Loss(nn.Module):
    """Macro soft F1: vectorized, average over all classes. Loss = 1 - mean(soft_f1_per_class)."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)  # [B, K]
        tp = (probs * targets).sum(dim=0)   # [K]
        fp = (probs * (1 - targets)).sum(dim=0)
        fn = ((1 - probs) * targets).sum(dim=0)
        soft_f1 = 2 * tp / (2 * tp + fp + fn + self.eps)
        return (1.0 - soft_f1.mean())


def build_loss(mode: str, lambda_soft_f1: float = 1.0, pos_weight: torch.Tensor | None = None) -> nn.Module:
    """
    mode: "bce" -> BCEWithLogitsLoss(pos_weight=pos_weight if provided).
    mode: "bce_softf1" -> BCEWithLogitsLoss + lambda_soft_f1 * MacroSoftF1Loss.
    Returns a module that forward(logits, targets) returns a scalar loss.
    """
    if mode == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if mode == "bce_softf1":
        class CombinedLoss(nn.Module):
            def __init__(self, lam: float, pos_weight_inner: torch.Tensor | None):
                super().__init__()
                # Use the same pos_weight as in pure BCE mode to handle class imbalance.
                self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_inner)
                self.soft_f1 = MacroSoftF1Loss()
                self.lam = lam

            def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                return self.bce(logits, targets) + self.lam * self.soft_f1(logits, targets)

        return CombinedLoss(lambda_soft_f1, pos_weight)

    raise ValueError(f"loss mode must be 'bce' or 'bce_softf1'; got {mode!r}")
