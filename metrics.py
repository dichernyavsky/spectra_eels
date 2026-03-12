from __future__ import annotations

import torch
import numpy as np


def logits_to_preds(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Convert logits to binary predictions using sigmoid and threshold.
    
    Args:
        logits: [B, num_classes] logits
        threshold: threshold for binary classification
    
    Returns:
        [B, num_classes] binary predictions (0 or 1)
    """
    probs = torch.sigmoid(logits)
    return (probs >= threshold).long()


def multilabel_f1_scores(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute micro and macro F1, precision, recall for multi-label classification.
    
    Args:
        logits: [B, num_classes] logits
        targets: [B, num_classes] binary targets
        threshold: threshold for binary predictions
    
    Returns:
        Dictionary with micro/macro precision, recall, F1
    """
    preds = logits_to_preds(logits, threshold)
    
    # Flatten for micro metrics
    preds_flat = preds.flatten()
    targets_flat = targets.flatten().long()
    
    # Micro metrics (global)
    tp = (preds_flat * targets_flat).sum().float()
    fp = (preds_flat * (1 - targets_flat)).sum().float()
    fn = ((1 - preds_flat) * targets_flat).sum().float()
    
    micro_precision = tp / (tp + fp + 1e-12)
    micro_recall = tp / (tp + fn + 1e-12)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-12)
    
    # Macro metrics (per-class average)
    num_classes = logits.size(1)
    class_precisions = []
    class_recalls = []
    class_f1s = []
    
    for k in range(num_classes):
        pred_k = preds[:, k]
        target_k = targets[:, k].long()
        
        tp_k = (pred_k * target_k).sum().float()
        fp_k = (pred_k * (1 - target_k)).sum().float()
        fn_k = ((1 - pred_k) * target_k).sum().float()
        
        prec_k = tp_k / (tp_k + fp_k + 1e-12)
        rec_k = tp_k / (tp_k + fn_k + 1e-12)
        f1_k = 2 * prec_k * rec_k / (prec_k + rec_k + 1e-12)
        
        class_precisions.append(prec_k.item())
        class_recalls.append(rec_k.item())
        class_f1s.append(f1_k.item())
    
    macro_precision = np.mean(class_precisions)
    macro_recall = np.mean(class_recalls)
    macro_f1 = np.mean(class_f1s)
    
    return {
        "micro_precision": micro_precision.item(),
        "micro_recall": micro_recall.item(),
        "micro_f1": micro_f1.item(),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def per_class_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    """
    Compute per-class precision, recall, and F1.
    
    Args:
        logits: [B, num_classes] logits
        targets: [B, num_classes] binary targets
        threshold: threshold for binary predictions
    
    Returns:
        Dictionary with per-class metrics as tensors of shape [num_classes]
    """
    preds = logits_to_preds(logits, threshold)
    num_classes = logits.size(1)
    
    precisions = []
    recalls = []
    f1s = []
    
    for k in range(num_classes):
        pred_k = preds[:, k]
        target_k = targets[:, k].long()
        
        tp_k = (pred_k * target_k).sum().float()
        fp_k = (pred_k * (1 - target_k)).sum().float()
        fn_k = ((1 - pred_k) * target_k).sum().float()
        
        prec_k = tp_k / (tp_k + fp_k + 1e-12)
        rec_k = tp_k / (tp_k + fn_k + 1e-12)
        f1_k = 2 * prec_k * rec_k / (prec_k + rec_k + 1e-12)
        
        precisions.append(prec_k)
        recalls.append(rec_k)
        f1s.append(f1_k)
    
    return {
        "precision": torch.stack(precisions),
        "recall": torch.stack(recalls),
        "f1": torch.stack(f1s),
    }


def multilabel_weighted_f1_scores(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute weighted precision, recall, and F1 for multi-label classification.
    
    Weight for each class = support (number of positive samples) of that class.
    
    Args:
        logits: [B, num_classes] logits
        targets: [B, num_classes] binary targets
        threshold: threshold for binary predictions
    
    Returns:
        Dictionary with weighted precision, recall, F1
    """
    preds = logits_to_preds(logits, threshold)
    num_classes = logits.size(1)
    
    # Compute per-class metrics
    class_precisions = []
    class_recalls = []
    class_f1s = []
    class_weights = []
    
    for k in range(num_classes):
        pred_k = preds[:, k]
        target_k = targets[:, k].long()
        
        tp_k = (pred_k * target_k).sum().float()
        fp_k = (pred_k * (1 - target_k)).sum().float()
        fn_k = ((1 - pred_k) * target_k).sum().float()
        
        prec_k = tp_k / (tp_k + fp_k + 1e-12)
        rec_k = tp_k / (tp_k + fn_k + 1e-12)
        f1_k = 2 * prec_k * rec_k / (prec_k + rec_k + 1e-12)
        
        # Weight = support (number of positive samples)
        weight_k = target_k.sum().float()
        
        class_precisions.append(prec_k.item())
        class_recalls.append(rec_k.item())
        class_f1s.append(f1_k.item())
        class_weights.append(weight_k.item())
    
    # Convert to numpy for weighted average
    class_precisions = np.array(class_precisions)
    class_recalls = np.array(class_recalls)
    class_f1s = np.array(class_f1s)
    class_weights = np.array(class_weights)
    
    # Normalize weights
    total_weight = class_weights.sum()
    if total_weight > 0:
        normalized_weights = class_weights / total_weight
    else:
        normalized_weights = np.ones(num_classes) / num_classes
    
    # Weighted averages
    weighted_precision = np.average(class_precisions, weights=normalized_weights)
    weighted_recall = np.average(class_recalls, weights=normalized_weights)
    weighted_f1 = np.average(class_f1s, weights=normalized_weights)
    
    return {
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }


def exact_match_ratio(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Compute exact match ratio (subset accuracy) for multi-label classification.
    
    Exact match = fraction of samples where the entire multi-label vector is predicted correctly.
    
    Args:
        logits: [B, num_classes] logits
        targets: [B, num_classes] binary targets
        threshold: threshold for binary predictions
    
    Returns:
        Exact match ratio (0.0 to 1.0)
    """
    preds = logits_to_preds(logits, threshold)
    
    # Check if entire vector matches for each sample
    exact_matches = (preds == targets.long()).all(dim=1).float()
    
    return exact_matches.mean().item()


def threshold_sweep(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thresholds: torch.Tensor | np.ndarray | None = None,
) -> dict[str, float | float]:
    """
    Perform threshold sweep to find best threshold for multi-label classification.
    
    Args:
        logits: [B, num_classes] logits
        targets: [B, num_classes] binary targets
        thresholds: Optional array of thresholds to test. If None, uses np.arange(0.05, 0.96, 0.05)
    
    Returns:
        Dictionary with:
            - "best_threshold": best threshold value (by macro F1)
            - "best_macro_f1": macro F1 at best threshold
            - "macro_f1_at_0.5": macro F1 at threshold 0.5
            - "best_micro_f1": micro F1 at best threshold
            - "best_macro_recall": macro recall at best threshold
            - "best_macro_precision": macro precision at best threshold
            - "best_balanced_threshold": threshold minimizing |precision - recall|
            - "best_balanced_macro_f1": macro F1 at balanced threshold
            - "best_balanced_macro_precision": macro precision at balanced threshold
            - "best_balanced_macro_recall": macro recall at balanced threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    
    if isinstance(thresholds, np.ndarray):
        thresholds = torch.from_numpy(thresholds).float()
    
    best_threshold = 0.5
    best_macro_f1 = -1.0
    best_micro_f1 = 0.0
    best_macro_recall = 0.0
    best_macro_precision = 0.0
    
    best_balanced_threshold = 0.5
    best_balanced_macro_f1 = 0.0
    best_balanced_macro_precision = 0.0
    best_balanced_macro_recall = 0.0
    min_precision_recall_diff = float('inf')
    
    # Compute metrics at threshold 0.5
    metrics_05 = multilabel_f1_scores(logits, targets, threshold=0.5)
    macro_f1_at_05 = metrics_05["macro_f1"]
    
    # Sweep thresholds
    for thresh in thresholds:
        metrics = multilabel_f1_scores(logits, targets, threshold=thresh.item())
        
        # Track best by macro F1
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_threshold = thresh.item()
            best_micro_f1 = metrics["micro_f1"]
            best_macro_recall = metrics["macro_recall"]
            best_macro_precision = metrics["macro_precision"]
        
        # Track best balanced (minimize |precision - recall|)
        precision_recall_diff = abs(metrics["macro_precision"] - metrics["macro_recall"])
        if precision_recall_diff < min_precision_recall_diff:
            min_precision_recall_diff = precision_recall_diff
            best_balanced_threshold = thresh.item()
            best_balanced_macro_f1 = metrics["macro_f1"]
            best_balanced_macro_precision = metrics["macro_precision"]
            best_balanced_macro_recall = metrics["macro_recall"]
    
    return {
        "best_threshold": best_threshold,
        "best_macro_f1": best_macro_f1,
        "macro_f1_at_0.5": macro_f1_at_05,
        "best_micro_f1": best_micro_f1,
        "best_macro_recall": best_macro_recall,
        "best_macro_precision": best_macro_precision,
        "best_balanced_threshold": best_balanced_threshold,
        "best_balanced_macro_f1": best_balanced_macro_f1,
        "best_balanced_macro_precision": best_balanced_macro_precision,
        "best_balanced_macro_recall": best_balanced_macro_recall,
    }
