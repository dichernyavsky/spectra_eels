"""
Metrics for multi-label EELS classification, following Annys et al.:
- predictions are obtained via sigmoid + threshold
- supports micro/macro/weighted precision/recall/F1
- exact match rate (EMR)
- RMSE between probabilities and ground truth.
"""
import torch


def _probs_and_preds(
    logits: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert logits to probabilities and binary predictions."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    return probs, preds


def _micro_prf(
    preds: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Micro-averaged precision/recall/F1 over all labels."""
    t = targets.float()
    tp = (preds * t).sum()
    fp = (preds * (1.0 - t)).sum()
    fn = ((1.0 - preds) * t).sum()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


def _per_class_prf(
    preds: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-class precision/recall/F1 (no reduction)."""
    t = targets.float()
    tp = (preds * t).sum(dim=0)
    fp = (preds * (1.0 - t)).sum(dim=0)
    fn = ((1.0 - preds) * t).sum(dim=0)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


def _macro_prf(
    preds: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Macro-averaged precision/recall/F1 (mean over classes)."""
    p_c, r_c, f1_c = _per_class_prf(preds, targets, eps=eps)
    return p_c.mean(), r_c.mean(), f1_c.mean()


def _weighted_prf(
    preds: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Weighted precision/recall/F1:
    per-class scores weighted by class support (occurrence in targets).
    """
    t = targets.float()
    p_c, r_c, f1_c = _per_class_prf(preds, t, eps=eps)
    support = t.sum(dim=0)  # [K]
    total = support.sum()
    if total > 0:
        w = support / total
    else:
        w = torch.ones_like(support, device=t.device) / max(1, t.size(1))
    precision = (p_c * w).sum()
    recall = (r_c * w).sum()
    f1 = (f1_c * w).sum()
    return precision, recall, f1


def exact_match(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Exact match rate: fraction of samples where full label vector matches."""
    match = (preds == targets.float()).all(dim=1).float()
    return match.mean()


def rmse(
    probs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Root mean squared error between probabilities and ground truth labels."""
    t = targets.float()
    return torch.sqrt(torch.mean((probs - t) ** 2))


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute multi-label metrics for given logits and ground truth.

    Returns a dict with:
    - micro_precision, micro_recall, micro_f1
    - macro_precision, macro_recall, macro_f1
    - weighted_precision, weighted_recall, weighted_f1
    - exact_match
    - rmse
    """
    device = logits.device
    targets = targets.to(device)
    probs, preds = _probs_and_preds(logits, threshold=threshold)

    micro_p, micro_r, micro_f = _micro_prf(preds, targets)
    macro_p, macro_r, macro_f = _macro_prf(preds, targets)
    weighted_p, weighted_r, weighted_f = _weighted_prf(preds, targets)
    emr = exact_match(preds, targets)
    rmse_val = rmse(probs, targets)

    return {
        "micro_precision": float(micro_p.item()),
        "micro_recall": float(micro_r.item()),
        "micro_f1": float(micro_f.item()),
        "macro_precision": float(macro_p.item()),
        "macro_recall": float(macro_r.item()),
        "macro_f1": float(macro_f.item()),
        "weighted_precision": float(weighted_p.item()),
        "weighted_recall": float(weighted_r.item()),
        "weighted_f1": float(weighted_f.item()),
        "exact_match": float(emr.item()),
        "rmse": float(rmse_val.item()),
    }


def threshold_sweep(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thresholds: torch.Tensor | None = None,
) -> dict:
    """
    Sweep over thresholds on validation set.

    For each threshold computes all metrics via compute_metrics.
    Returns:
    - threshold_article_weighted: threshold minimizing |weighted_precision - weighted_recall|
    - metrics_at_article_weighted: metrics dict at threshold_article_weighted
    - threshold_article_micro: threshold minimizing |micro_precision - micro_recall|
    - metrics_at_article_micro: metrics dict at threshold_article_micro
    - threshold_best_weighted_f1: threshold maximizing weighted_f1
    - metrics_at_best_weighted_f1: metrics dict at threshold_best_weighted_f1
    """
    if thresholds is None:
        thresholds = torch.arange(
            0.05, 0.95 + 1e-6, 0.01, device=logits.device, dtype=logits.dtype
        )

    # Weighted-based article threshold (as in the paper: occurrence-weighted summary).
    best_article_thr_weighted = 0.5
    best_article_gap_weighted = float("inf")
    best_article_metrics_weighted = None

    # Micro-based variant for analysis.
    best_article_thr_micro = 0.5
    best_article_gap_micro = float("inf")
    best_article_metrics_micro = None

    best_wf1_thr = 0.5
    best_wf1 = -1.0
    best_wf1_metrics = None

    for th in thresholds:
        th_val = float(th.item())
        m = compute_metrics(logits, targets, threshold=th_val)

        gap_weighted = abs(m["weighted_precision"] - m["weighted_recall"])
        if gap_weighted < best_article_gap_weighted:
            best_article_gap_weighted = gap_weighted
            best_article_thr_weighted = th_val
            best_article_metrics_weighted = m

        gap_micro = abs(m["micro_precision"] - m["micro_recall"])
        if gap_micro < best_article_gap_micro:
            best_article_gap_micro = gap_micro
            best_article_thr_micro = th_val
            best_article_metrics_micro = m

        if m["weighted_f1"] > best_wf1:
            best_wf1 = m["weighted_f1"]
            best_wf1_thr = th_val
            best_wf1_metrics = m

    # Fallback in case something went wrong
    if best_article_metrics_weighted is None:
        best_article_thr_weighted = 0.5
        best_article_metrics_weighted = compute_metrics(
            logits, targets, threshold=best_article_thr_weighted
        )
    if best_article_metrics_micro is None:
        best_article_thr_micro = best_article_thr_weighted
        best_article_metrics_micro = best_article_metrics_weighted
    if best_wf1_metrics is None:
        best_wf1_thr = best_article_thr_weighted
        best_wf1_metrics = best_article_metrics_weighted

    return {
        "threshold_article_weighted": best_article_thr_weighted,
        "metrics_at_article_weighted": best_article_metrics_weighted,
        "threshold_article_micro": best_article_thr_micro,
        "metrics_at_article_micro": best_article_metrics_micro,
        "threshold_best_weighted_f1": best_wf1_thr,
        "metrics_at_best_weighted_f1": best_wf1_metrics,
    }
