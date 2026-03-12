"""
Evaluation script aligned with original repository logic.

Computes multi-label metrics in the style of the original repo:
- micro precision/recall/F1
- macro precision/recall/F1
- weighted precision/recall/F1
- exact match ratio
- threshold sweep
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from dataset import (
    SpectrumPreprocessConfig,
    make_split_dataset,
    make_dataloader,
    get_split_file_paths,
)
from model import EELSPerElementAttentionModel
from metrics import (
    multilabel_f1_scores,
    multilabel_weighted_f1_scores,
    exact_match_ratio,
    threshold_sweep,
)
from stats import compute_class_stats_identification


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_nonzero_mask: bool = True,
) -> dict[str, float]:
    """Evaluate model and compute all metrics."""
    model.eval()
    
    all_logits = []
    all_targets = []
    
    for batch in dataloader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        
        nonzero_mask = None
        if use_nonzero_mask and "nonzero_mask" in batch:
            nonzero_mask = batch["nonzero_mask"].to(device)
        
        output = model(x, nonzero_mask=nonzero_mask)
        logits = output["logits"]
        
        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Threshold sweep
    sweep_results = threshold_sweep(all_logits, all_targets)
    best_threshold = sweep_results["best_threshold"]
    
    # Compute metrics at best threshold
    metrics = multilabel_f1_scores(all_logits, all_targets, threshold=best_threshold)
    weighted_metrics = multilabel_weighted_f1_scores(all_logits, all_targets, threshold=best_threshold)
    exact_match = exact_match_ratio(all_logits, all_targets, threshold=best_threshold)
    
    # Also compute at threshold 0.5
    metrics_05 = multilabel_f1_scores(all_logits, all_targets, threshold=0.5)
    weighted_metrics_05 = multilabel_weighted_f1_scores(all_logits, all_targets, threshold=0.5)
    exact_match_05 = exact_match_ratio(all_logits, all_targets, threshold=0.5)
    
    results = {
        **metrics,
        **weighted_metrics,
        "exact_match": exact_match,
        **sweep_results,
        # At threshold 0.5
        "micro_f1_at_0.5": metrics_05["micro_f1"],
        "macro_f1_at_0.5": metrics_05["macro_f1"],
        "weighted_f1_at_0.5": weighted_metrics_05["weighted_f1"],
        "exact_match_at_0.5": exact_match_05,
    }
    
    return results


def print_metrics(metrics: dict[str, float], split: str) -> None:
    """Print formatted metrics."""
    print(f"\n{'='*60}")
    print(f"{split.upper()} Metrics")
    print(f"{'='*60}")
    
    print(f"\nAt best threshold ({metrics['best_threshold']:.3f}):")
    print(f"  Micro F1:     {metrics['micro_f1']:.4f}")
    print(f"  Macro F1:     {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:  {metrics['weighted_f1']:.4f}")
    print(f"  Exact Match:  {metrics['exact_match']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {metrics['macro_recall']:.4f}")
    
    print(f"\nAt threshold 0.5:")
    print(f"  Micro F1:     {metrics['micro_f1_at_0.5']:.4f}")
    print(f"  Macro F1:     {metrics['macro_f1_at_0.5']:.4f}")
    print(f"  Weighted F1:  {metrics['weighted_f1_at_0.5']:.4f}")
    print(f"  Exact Match:  {metrics['exact_match_at_0.5']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model aligned with original repo")
    parser.add_argument("--root", type=str, default="EELS", help="Path to EELS dataset root")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--use_nonzero_mask", action="store_true", help="Use nonzero mask")
    args = parser.parse_args()
    
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create model
    model = EELSPerElementAttentionModel(
        num_classes=80,
        spectrum_length=3072,
        channels=256,
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Model loaded successfully")
    
    # Create dataset
    preprocess_cfg = SpectrumPreprocessConfig(
        add_channel_dim=True,
        return_nonzero_mask=args.use_nonzero_mask,
        return_nonzero_bounds=False,
    )
    
    dataset = make_split_dataset(
        root=args.root,
        split=args.split,
        task="identification",
        preprocess_cfg=preprocess_cfg,
        return_metadata=False,
        return_index=False,
    )
    
    print(f"\nDataset size: {len(dataset):,}")
    
    # Compute dataset statistics
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    
    split_files = get_split_file_paths(args.root, split=args.split)
    stats = compute_class_stats_identification(split_files)
    
    print(f"Total samples: {stats['total_samples']:,}")
    print(f"Positive samples per class:")
    print(f"  Min: {stats['pos_counts'].min()}")
    print(f"  Max: {stats['pos_counts'].max()}")
    print(f"  Mean: {stats['pos_counts'].mean():.1f}")
    print(f"  Median: {stats['pos_counts'].median():.1f}")
    
    # Create dataloader
    dataloader = make_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("Evaluating Model")
    print("="*60)
    
    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        use_nonzero_mask=args.use_nonzero_mask,
    )
    
    print_metrics(metrics, args.split)
    
    print("\n" + "="*60)
    print("Evaluation Complete")
    print("="*60)


if __name__ == "__main__":
    main()
