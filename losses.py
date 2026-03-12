from __future__ import annotations

import torch
import torch.nn as nn


class MacroSoftF1Loss(nn.Module):
    """
    Macro soft F1 loss for multi-label classification.
    
    Computes soft F1 score per class using sigmoid probabilities, then averages.
    
    If ignore_absent_classes=True, averages only over classes that have at least
    one positive target in the batch. This is useful for tiny-overfit/debug mode.
    
    If ignore_absent_classes=False, averages over all classes (closer to original paper).
    
    Loss = 1 - mean(soft F1 per class)
    
    Based on: https://arxiv.org/abs/1901.08128
    """
    
    def __init__(self, eps: float = 1e-8, ignore_absent_classes: bool = True):
        """
        Args:
            eps: Small epsilon for numerical stability
            ignore_absent_classes: If True, only average over classes with positive targets in batch.
                                  If False, average over all classes (original paper behavior).
        """
        super().__init__()
        self.eps = eps
        self.ignore_absent_classes = ignore_absent_classes
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Compute macro soft F1 loss.
        
        Args:
            logits: [B, num_classes] classification logits
            targets: [B, num_classes] binary targets (0 or 1)
        
        Returns:
            Tuple of (loss, num_valid_classes):
                - loss: Scalar loss value
                - num_valid_classes: Number of classes with at least one positive target
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)  # [B, num_classes]
        
        # Vectorized computation of TP, FP, FN for all classes
        # probs: [B, num_classes], targets: [B, num_classes]
        tp = (probs * targets).sum(dim=0)  # [num_classes]
        fp = (probs * (1 - targets)).sum(dim=0)  # [num_classes]
        fn = ((1 - probs) * targets).sum(dim=0)  # [num_classes]
        
        # Soft F1 for all classes: [num_classes]
        soft_f1 = 2 * tp / (2 * tp + fp + fn + self.eps)
        
        if self.ignore_absent_classes:
            # Mask: only classes with at least one positive target
            valid_mask = targets.sum(dim=0) > 0  # [num_classes]
            num_valid_classes = valid_mask.sum().item()
            
            # If no valid classes, return 0.0 loss
            if num_valid_classes == 0:
                return torch.tensor(0.0, device=logits.device, dtype=logits.dtype), 0
            
            # Average soft F1 only over valid classes
            valid_soft_f1 = soft_f1[valid_mask]  # [num_valid_classes]
            mean_soft_f1 = valid_soft_f1.mean()
        else:
            # Average over all classes (original paper behavior)
            mean_soft_f1 = soft_f1.mean()
            num_valid_classes = logits.size(1)
        
        # Loss = 1 - mean soft F1
        loss = 1.0 - mean_soft_f1
        
        return loss, num_valid_classes
