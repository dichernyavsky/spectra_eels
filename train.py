from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
    per_class_metrics,
    threshold_sweep,
)
from stats import compute_class_stats_identification
from losses import MacroSoftF1Loss


@dataclass
class TrainConfig:
    root: str = "EELS"
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    threshold: float = 0.5
    use_nonzero_mask: bool = True
    save_dir: str = "checkpoints"
    seed: int = 42
    overfit_small_batch: bool = False
    overfit_num_samples: int = 128
    grad_clip_norm: float = 1.0
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    tiny_overfit_mode: bool = False
    tiny_overfit_num_samples: int = 32
    tiny_overfit_epochs: int = 50
    lambda_soft_f1: float = 1.0
    lambda_soft_f1_tiny_overfit: float = 0.1
    tiny_overfit_bce_only: bool = False
    selection_metric: str = "weighted_f1"  # "weighted_f1", "macro_f1", "micro_f1"


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    bce_criterion: nn.Module,
    soft_f1_criterion: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_nonzero_mask: bool = True,
    grad_clip_norm: float = 1.0,
    max_batches: int | None = None,
    lambda_soft_f1: float = 1.0,
) -> dict[str, float]:
    """Train for one epoch with profiling."""
    model.train()
    total_loss = 0.0
    total_num_valid_classes = 0
    num_batches_with_softf1 = 0
    num_batches = 0
    total_data_time = 0.0
    total_compute_time = 0.0
    total_samples = 0
    
    data_start = time.time()
    for batch_idx, batch in enumerate(train_loader):
        # Measure data loading time
        data_time = time.time() - data_start
        total_data_time += data_time
        
        # Early stopping if max_batches reached
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        compute_start = time.time()
        
        x = batch["x"].to(device)  # [B, 1, 3072]
        y = batch["y"].to(device)  # [B, 80]
        
        nonzero_mask = None
        if use_nonzero_mask and "nonzero_mask" in batch:
            nonzero_mask = batch["nonzero_mask"].to(device)
        
        # Forward
        output = model(x, nonzero_mask=nonzero_mask)
        logits = output["logits"]
        
        # Combined loss: BCE + lambda * soft F1
        bce_loss = bce_criterion(logits, y)
        if soft_f1_criterion is not None:
            soft_f1_loss, num_valid_classes = soft_f1_criterion(logits, y)
            loss = bce_loss + lambda_soft_f1 * soft_f1_loss
            total_num_valid_classes += num_valid_classes
            if num_valid_classes > 0:
                num_batches_with_softf1 += 1
        else:
            loss = bce_loss
            num_valid_classes = 0
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        
        # Clear CUDA cache periodically to avoid OOM (every 10 batches)
        if device.type == "cuda" and batch_idx > 0 and batch_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        # Measure compute time
        compute_time = time.time() - compute_start
        total_compute_time += compute_time
        
        total_loss += loss.item()
        num_batches += 1
        total_samples += x.size(0)
        
        # Start timing next data load
        data_start = time.time()
    
    avg_data_time = total_data_time / num_batches if num_batches > 0 else 0.0
    avg_compute_time = total_compute_time / num_batches if num_batches > 0 else 0.0
    total_time = total_data_time + total_compute_time
    
    batches_per_sec = num_batches / total_time if total_time > 0 else 0.0
    samples_per_sec = total_samples / total_time if total_time > 0 else 0.0
    
    avg_num_valid_classes = total_num_valid_classes / num_batches_with_softf1 if num_batches_with_softf1 > 0 else 0.0
    
    return {
        "train_loss": total_loss / num_batches if num_batches > 0 else 0.0,
        "avg_data_time": avg_data_time,
        "avg_compute_time": avg_compute_time,
        "batches_per_sec": batches_per_sec,
        "samples_per_sec": samples_per_sec,
        "num_batches": num_batches,
        "avg_num_valid_softf1_classes": avg_num_valid_classes,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    use_nonzero_mask: bool = True,
    max_batches: int | None = None,
    compute_train_metrics: bool = False,
    do_threshold_sweep: bool = True,
    return_attention: bool = False,
) -> dict[str, float]:
    """Evaluate on validation set with profiling."""
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []
    all_attention = [] if return_attention else None
    num_batches = 0
    total_data_time = 0.0
    total_compute_time = 0.0
    total_samples = 0
    
    data_start = time.time()
    for batch_idx, batch in enumerate(val_loader):
        # Measure data loading time
        data_time = time.time() - data_start
        total_data_time += data_time
        
        # Early stopping if max_batches reached
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        compute_start = time.time()
        
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        
        nonzero_mask = None
        if use_nonzero_mask and "nonzero_mask" in batch:
            nonzero_mask = batch["nonzero_mask"].to(device)
        
        output = model(x, nonzero_mask=nonzero_mask, return_attention=return_attention)
        logits = output["logits"]
        
        # Store attention for collapse inspection
        if return_attention and "attention" in output:
            all_attention.append(output["attention"].cpu())
        
        loss = criterion(logits, y)
        
        # Measure compute time
        compute_time = time.time() - compute_start
        total_compute_time += compute_time
        
        total_loss += loss.item()
        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())
        num_batches += 1
        total_samples += x.size(0)
        
        # Start timing next data load
        data_start = time.time()
    
    # Concatenate all predictions and targets
    all_logits = torch.cat(all_logits, dim=0) if all_logits else torch.empty(0, 80)
    all_targets = torch.cat(all_targets, dim=0) if all_targets else torch.empty(0, 80)
    
    # Concatenate attention if collected
    if all_attention and len(all_attention) > 0:
        all_attention = torch.cat(all_attention, dim=0)  # [B, K, N]
    else:
        all_attention = None
    
    # Compute metrics
    if len(all_logits) > 0:
        # Threshold sweep for best threshold
        if do_threshold_sweep:
            sweep_results = threshold_sweep(all_logits, all_targets)
            best_threshold = sweep_results["best_threshold"]
            metrics = multilabel_f1_scores(all_logits, all_targets, threshold=best_threshold)
            metrics.update(sweep_results)
        else:
            metrics = multilabel_f1_scores(all_logits, all_targets, threshold=threshold)
            metrics["best_threshold"] = threshold
            metrics["best_macro_f1"] = metrics["macro_f1"]
            metrics["best_macro_precision"] = metrics["macro_precision"]
            metrics["best_macro_recall"] = metrics["macro_recall"]
            metrics["macro_f1_at_0.5"] = multilabel_f1_scores(all_logits, all_targets, threshold=0.5)["macro_f1"]
            # Set balanced threshold to same as best threshold when sweep is disabled
            metrics["best_balanced_threshold"] = threshold
            metrics["best_balanced_macro_f1"] = metrics["macro_f1"]
            metrics["best_balanced_macro_precision"] = metrics["macro_precision"]
            metrics["best_balanced_macro_recall"] = metrics["macro_recall"]
        
        # Compute weighted metrics and exact match
        weighted_metrics = multilabel_weighted_f1_scores(all_logits, all_targets, threshold=metrics["best_threshold"])
        metrics.update(weighted_metrics)
        metrics["exact_match"] = exact_match_ratio(all_logits, all_targets, threshold=metrics["best_threshold"])
        
        per_class = per_class_metrics(all_logits, all_targets, threshold=metrics["best_threshold"])
        
        # Compute average logits/probs for positive and negative targets
        probs = torch.sigmoid(all_logits)
        positive_mask = all_targets > 0
        negative_mask = all_targets == 0
        
        if positive_mask.any():
            metrics["avg_logit_positive"] = all_logits[positive_mask].mean().item()
            metrics["avg_prob_positive"] = probs[positive_mask].mean().item()
        else:
            metrics["avg_logit_positive"] = 0.0
            metrics["avg_prob_positive"] = 0.0
        
        if negative_mask.any():
            metrics["avg_logit_negative"] = all_logits[negative_mask].mean().item()
            metrics["avg_prob_negative"] = probs[negative_mask].mean().item()
        else:
            metrics["avg_logit_negative"] = 0.0
            metrics["avg_prob_negative"] = 0.0
    else:
        metrics = {
            "micro_f1": 0.0,
            "macro_f1": 0.0,
            "macro_recall": 0.0,
            "macro_precision": 0.0,
            "best_threshold": threshold,
            "best_macro_f1": 0.0,
            "macro_f1_at_0.5": 0.0,
            "best_balanced_threshold": threshold,
            "best_balanced_macro_f1": 0.0,
            "best_balanced_macro_precision": 0.0,
            "best_balanced_macro_recall": 0.0,
            "best_macro_precision": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1": 0.0,
            "exact_match": 0.0,
            "avg_logit_positive": 0.0,
            "avg_prob_positive": 0.0,
            "avg_logit_negative": 0.0,
            "avg_prob_negative": 0.0,
        }
        per_class = {
            "precision": torch.zeros(80),
            "recall": torch.zeros(80),
            "f1": torch.zeros(80),
        }
    
    avg_data_time = total_data_time / num_batches if num_batches > 0 else 0.0
    avg_compute_time = total_compute_time / num_batches if num_batches > 0 else 0.0
    total_time = total_data_time + total_compute_time
    
    batches_per_sec = num_batches / total_time if total_time > 0 else 0.0
    samples_per_sec = total_samples / total_time if total_time > 0 else 0.0
    
    metrics["val_loss"] = total_loss / num_batches if num_batches > 0 else 0.0
    metrics["per_class_precision"] = per_class["precision"]
    metrics["per_class_recall"] = per_class["recall"]
    metrics["per_class_f1"] = per_class["f1"]
    metrics["avg_data_time"] = avg_data_time
    metrics["avg_compute_time"] = avg_compute_time
    metrics["batches_per_sec"] = batches_per_sec
    metrics["samples_per_sec"] = samples_per_sec
    metrics["num_batches"] = num_batches
    
    # Compute attention collapse metrics if attention is available
    if all_attention is not None:
        # all_attention: [B, K, N]
        eps = 1e-8
        # Compute entropy per sample and class: H = -sum(alpha_i * log(alpha_i + eps))
        attention_log = torch.log(all_attention + eps)  # [B, K, N]
        entropy = -(all_attention * attention_log).sum(dim=-1)  # [B, K]
        metrics["attention_entropy_mean"] = entropy.mean().item()
        
        # Compute max attention weight per sample and class
        max_attention = all_attention.max(dim=-1).values  # [B, K]
        metrics["attention_max_mean"] = max_attention.mean().item()
    else:
        metrics["attention_entropy_mean"] = 0.0
        metrics["attention_max_mean"] = 0.0
    
    # Store logits/probs/targets for debug mode
    if compute_train_metrics:
        metrics["logits"] = all_logits
        metrics["probs"] = probs if len(all_logits) > 0 else torch.empty(0, 80)
        metrics["targets"] = all_targets
    
    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    config: TrainConfig,
    save_path: Path,
    best_threshold: float = 0.5,
) -> None:
    """Save model checkpoint."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "best_threshold": best_threshold,
        "config": asdict(config),
    }
    
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> dict:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load checkpoint on
    
    Returns:
        Dictionary with checkpoint contents
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint


def main(config: TrainConfig):
    """Main training function."""
    set_seed(config.seed)
    
    device = torch.device(config.device)
    print(f"Using device: {device}")
    
    # Clear CUDA cache if using GPU
    if device.type == "cuda":
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        total = torch.cuda.get_device_properties(device).total_memory / 1024**2
        print(f"GPU memory: {allocated:.2f} MB allocated, {reserved:.2f} MB reserved, {total:.2f} MB total")
        
        # For tiny_overfit, suggest CPU if GPU memory is low
        if config.tiny_overfit_mode and reserved > total * 0.8:
            print("⚠️  WARNING: GPU memory is low. Consider using CPU for tiny_overfit mode.")
    
    # Create save directory
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Preprocess config
    preprocess_cfg = SpectrumPreprocessConfig(
        add_channel_dim=True,
        return_nonzero_mask=config.use_nonzero_mask,
        return_nonzero_bounds=False,
    )
    
    # Create datasets
    print("\n" + "="*60)
    print("Creating datasets...")
    print("="*60)
    
    train_ds = make_split_dataset(
        root=config.root,
        split="train",
        task="identification",
        preprocess_cfg=preprocess_cfg,
        return_metadata=False,
        return_index=False,
    )
    
    val_ds = make_split_dataset(
        root=config.root,
        split="val",
        task="identification",
        preprocess_cfg=preprocess_cfg,
        return_metadata=False,
        return_index=False,
    )
    
    # Tiny overfit mode: very small subset for debugging
    if config.tiny_overfit_mode:
        print(f"\n🔬 TINY OVERFIT MODE: Using {config.tiny_overfit_num_samples} samples")
        base_train_ds = train_ds
        subset_indices = torch.randperm(len(base_train_ds))[:config.tiny_overfit_num_samples]
        train_ds = torch.utils.data.Subset(base_train_ds, subset_indices)
        val_ds = torch.utils.data.Subset(base_train_ds, subset_indices)  # Same subset for val
        config.epochs = config.tiny_overfit_epochs
        print(f"  Epochs set to: {config.epochs}")
    
    # Overfit mode: use small subset
    elif config.overfit_small_batch:
        print(f"\n⚠️  OVERFIT MODE: Using {config.overfit_num_samples} samples")
        base_train_ds = train_ds
        subset_indices = torch.randperm(len(base_train_ds))[:config.overfit_num_samples]
        train_ds = torch.utils.data.Subset(base_train_ds, subset_indices)
        val_ds = torch.utils.data.Subset(base_train_ds, subset_indices)  # Same subset for val
    
    print(f"Train dataset size: {len(train_ds):,}")
    print(f"Val dataset size: {len(val_ds):,}")
    
    # Create dataloaders
    train_loader = make_dataloader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    
    val_loader = make_dataloader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    
    # Compute pos_weight (only for non-tiny-overfit mode)
    print("\n" + "="*60)
    print("Computing class statistics...")
    print("="*60)
    
    train_files = get_split_file_paths(config.root, split="train")
    stats = compute_class_stats_identification(train_files)
    # Create tensor on CPU first, then move to device if needed
    pos_weight = torch.tensor(stats["pos_weight"], dtype=torch.float32)
    if not config.tiny_overfit_mode:
        pos_weight = pos_weight.to(device)
    
    if config.tiny_overfit_mode:
        print("  (pos_weight disabled in tiny_overfit_mode)")
    else:
        print(f"Pos weight range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")
        print(f"Pos weight mean: {pos_weight.mean():.2f}")
    
    # Create model
    print("\n" + "="*60)
    print("Creating model...")
    print("="*60)
    
    model = EELSPerElementAttentionModel(
        num_classes=80,
        spectrum_length=3072,
        channels=256,
    ).to(device)
    
    # Initialize output bias based on class prevalence (disabled in tiny_overfit_mode)
    if config.tiny_overfit_mode:
        print("  (bias initialization disabled in tiny_overfit_mode, using zeros)")
    else:
        # Create tensor on CPU first, then move to device
        prevalence = torch.tensor(stats["prevalence"], dtype=torch.float32)
        prevalence = prevalence.to(device)
        model.initialize_output_bias(prevalence)
        print("Initialized output bias from class prevalence")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss setup
    # In tiny_overfit_mode: no pos_weight, optional BCE-only mode, ignore_absent_classes=True
    # Otherwise: use pos_weight, combined loss, ignore_absent_classes=False (original paper)
    if config.tiny_overfit_mode:
        bce_criterion = nn.BCEWithLogitsLoss()  # No pos_weight
        if config.tiny_overfit_bce_only:
            soft_f1_criterion = None
            print("  Loss: BCE only (tiny_overfit_bce_only=True)")
        else:
            soft_f1_criterion = MacroSoftF1Loss(ignore_absent_classes=True)
            print(f"  Loss: BCE + soft F1 (lambda={config.lambda_soft_f1_tiny_overfit}, ignore_absent_classes=True)")
    else:
        bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        soft_f1_criterion = MacroSoftF1Loss(ignore_absent_classes=False)
        print(f"  Loss: BCE + soft F1 (lambda={config.lambda_soft_f1}, ignore_absent_classes=False)")
    
    # Optimizer
    if config.tiny_overfit_mode:
        # No weight decay in tiny_overfit_mode
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=0.0,
        )
        scheduler = None  # No scheduler in tiny_overfit_mode
        print("  Optimizer: Adam (weight_decay=0.0, no scheduler)")
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    best_metric = -1.0
    best_epoch = 0
    
    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        print("-" * 60)
        
        # Determine effective lambda_soft_f1
        if config.tiny_overfit_mode:
            effective_lambda_soft_f1 = config.lambda_soft_f1_tiny_overfit
        else:
            effective_lambda_soft_f1 = config.lambda_soft_f1
        
        # Train
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            bce_criterion=bce_criterion,
            soft_f1_criterion=soft_f1_criterion,
            optimizer=optimizer,
            device=device,
            use_nonzero_mask=config.use_nonzero_mask,
            grad_clip_norm=config.grad_clip_norm,
            max_batches=config.max_train_batches,
            lambda_soft_f1=effective_lambda_soft_f1,
        )
        
        # Compute train metrics (for debug/overfit modes)
        compute_train_metrics = config.tiny_overfit_mode or config.overfit_small_batch
        train_eval_metrics = None
        if compute_train_metrics:
            # Create a temporary dataloader for train metrics
            train_eval_loader = make_dataloader(
                train_ds,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
            )
            train_eval_metrics = evaluate(
                model=model,
                val_loader=train_eval_loader,
                criterion=bce_criterion,
                device=device,
                threshold=config.threshold,
                use_nonzero_mask=config.use_nonzero_mask,
                max_batches=config.max_train_batches,
                compute_train_metrics=True,
                do_threshold_sweep=True,
                return_attention=config.tiny_overfit_mode,  # Collect attention in tiny_overfit
            )
        
        # Validate
        val_metrics = evaluate(
            model=model,
            val_loader=val_loader,
            criterion=bce_criterion,
            device=device,
            threshold=config.threshold,
            use_nonzero_mask=config.use_nonzero_mask,
            max_batches=config.max_val_batches,
            compute_train_metrics=config.tiny_overfit_mode,  # Save logits only in tiny_overfit
            do_threshold_sweep=True,
            return_attention=config.tiny_overfit_mode,  # Collect attention in tiny_overfit
        )
        
        # Print metrics
        # Select metric for checkpoint selection
        if config.tiny_overfit_mode:
            # In tiny-overfit, use present-class macro F1
            current_metric = val_metrics.get('best_macro_f1', val_metrics.get('macro_f1_at_0.5', 0.0))
        else:
            # In normal training, use selection_metric
            if config.selection_metric == "weighted_f1":
                current_metric = val_metrics.get('weighted_f1', 0.0)
            elif config.selection_metric == "macro_f1":
                current_metric = val_metrics.get('best_macro_f1', 0.0)
            elif config.selection_metric == "micro_f1":
                current_metric = val_metrics.get('micro_f1', 0.0)
            else:
                current_metric = val_metrics.get('weighted_f1', 0.0)
        
        current_lr = optimizer.param_groups[0]["lr"]
        
        print(f"Train loss: {train_metrics['train_loss']:.4f}")
        
        # Log num_valid_softf1_classes if available
        if 'avg_num_valid_softf1_classes' in train_metrics:
            print(f"Train avg valid softF1 classes: {train_metrics['avg_num_valid_softf1_classes']:.1f}")
        
        if train_eval_metrics is not None:
            # In tiny-overfit mode, emphasize macro_f1 @ 0.5 as primary signal
            if config.tiny_overfit_mode:
                print(f"Train macro F1 @ 0.5: {train_eval_metrics['macro_f1_at_0.5']:.4f} (primary signal)")
                print(f"Train best macro F1: {train_eval_metrics['best_macro_f1']:.4f} (threshold={train_eval_metrics['best_threshold']:.3f})")
            else:
                print(f"Train macro F1 @ 0.5: {train_eval_metrics['macro_f1_at_0.5']:.4f}")
                print(f"Train best macro F1: {train_eval_metrics['best_macro_f1']:.4f} (threshold={train_eval_metrics['best_threshold']:.3f})")
            
            print(f"Train micro F1: {train_eval_metrics['micro_f1']:.4f}")
            print(f"Train macro Recall: {train_eval_metrics['macro_recall']:.4f}")
            
            # Early success criterion for tiny_overfit
            if config.tiny_overfit_mode:
                if train_eval_metrics['macro_f1_at_0.5'] > 0.90 or train_eval_metrics['best_macro_f1'] > 0.90 or train_eval_metrics['micro_f1'] > 0.95:
                    print("✓ Tiny overfit succeeded!")
        
        print(f"Val loss: {val_metrics['val_loss']:.4f}")
        
        # In tiny-overfit mode, emphasize macro_f1 @ 0.5 as primary signal
        if config.tiny_overfit_mode:
            print(f"Val macro F1 @ 0.5: {val_metrics['macro_f1_at_0.5']:.4f} (primary signal)")
            print(f"Val best macro F1: {val_metrics['best_macro_f1']:.4f} (threshold={val_metrics['best_threshold']:.3f})")
        else:
            print(f"Val micro F1: {val_metrics['micro_f1']:.4f}")
            print(f"Val macro F1 @ 0.5: {val_metrics['macro_f1_at_0.5']:.4f}")
            print(f"Val best macro F1: {val_metrics['best_macro_f1']:.4f} (threshold={val_metrics['best_threshold']:.3f})")
        
        if not config.tiny_overfit_mode:
            # Normal training: show comprehensive metrics
            print(f"Val weighted F1: {val_metrics.get('weighted_f1', 0.0):.4f}")
            print(f"Val exact match: {val_metrics.get('exact_match', 0.0):.4f}")
            print(f"Val macro Recall: {val_metrics['best_macro_recall']:.4f}")
            print(f"Val macro Precision: {val_metrics.get('best_macro_precision', 0.0):.4f}")
            print(f"Val balanced macro F1: {val_metrics.get('best_balanced_macro_f1', 0.0):.4f} (threshold={val_metrics.get('best_balanced_threshold', 0.5):.3f})")
        
        if config.tiny_overfit_mode or config.overfit_small_batch:
            print(f"\nDebug metrics:")
            print(f"  Avg logit (positive): {val_metrics['avg_logit_positive']:.4f}")
            print(f"  Avg logit (negative): {val_metrics['avg_logit_negative']:.4f}")
            print(f"  Avg prob (positive): {val_metrics['avg_prob_positive']:.4f}")
            print(f"  Avg prob (negative): {val_metrics['avg_prob_negative']:.4f}")
            
            # Attention collapse inspection
            if 'attention_entropy_mean' in val_metrics:
                print(f"  Attention entropy (mean): {val_metrics['attention_entropy_mean']:.4f}")
                print(f"  Attention max (mean): {val_metrics['attention_max_mean']:.4f}")
        
        print(f"LR: {current_lr:.6e}")
        print(f"\nPerformance:")
        print(f"  Train: {train_metrics['batches_per_sec']:.2f} batches/sec, "
              f"{train_metrics['samples_per_sec']:.1f} samples/sec")
        print(f"  Train data time: {train_metrics['avg_data_time']*1000:.2f}ms/batch, "
              f"compute time: {train_metrics['avg_compute_time']*1000:.2f}ms/batch")
        print(f"  Val: {val_metrics['batches_per_sec']:.2f} batches/sec, "
              f"{val_metrics['samples_per_sec']:.1f} samples/sec")
        print(f"  Val data time: {val_metrics['avg_data_time']*1000:.2f}ms/batch, "
              f"compute time: {val_metrics['avg_compute_time']*1000:.2f}ms/batch")
        
        # Step scheduler (disabled in tiny_overfit_mode)
        if scheduler is not None:
            scheduler.step(val_metrics["best_macro_f1"])
        
        # Save last checkpoint
        last_checkpoint_path = save_dir / "last_checkpoint.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=current_metric,
            config=config,
            save_path=last_checkpoint_path,
            best_threshold=val_metrics["best_threshold"],
        )
        
        # Save best checkpoint
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            best_checkpoint_path = save_dir / "best_checkpoint.pt"
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": best_metric,
                "best_threshold": val_metrics["best_threshold"],
                "config": asdict(config),
            }
            torch.save(checkpoint_data, best_checkpoint_path)
            metric_name = config.selection_metric if not config.tiny_overfit_mode else "macro_f1"
            print(f"✓ New best {metric_name}: {best_metric:.4f} (threshold={val_metrics['best_threshold']:.3f})")
        
        # Save debug logits in tiny_overfit mode
        if config.tiny_overfit_mode and "logits" in val_metrics:
            debug_path = save_dir / f"debug_logits_epoch_{epoch:03d}.npz"
            np.savez(
                debug_path,
                logits=val_metrics["logits"].cpu().numpy(),
                probs=val_metrics["probs"].cpu().numpy(),
                targets=val_metrics["targets"].cpu().numpy(),
            )
            print(f"  Saved debug logits to {debug_path}")
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Best macro F1: {best_metric:.4f} (epoch {best_epoch})")
    print(f"Checkpoint saved to: {save_dir / 'best_checkpoint.pt'}")


if __name__ == "__main__":
    # Tiny overfit mode for debugging
    # Note: For tiny_overfit, CPU is often sufficient and avoids GPU memory issues
    config_tiny = TrainConfig(
        root="EELS",
        batch_size=16,  # Reduced for GPU memory constraints
        num_workers=4,  # Reduced to avoid memory issues
        lr=1e-3,
        epochs=150,
        use_nonzero_mask=True,
        tiny_overfit_mode=True,
        tiny_overfit_num_samples=32,
        tiny_overfit_epochs=50,
        device="cpu",  # Use CPU for tiny_overfit to avoid GPU memory issues
        # device="cuda",  # Uncomment to use GPU if you have enough memory
    )
    
    main(config_tiny)
    
    # Uncomment for overfit_small_batch mode:
    # config_overfit = TrainConfig(
    #     root="EELS",
    #     batch_size=32,
    #     num_workers=4,
    #     lr=1e-3,
    #     epochs=10,
    #     use_nonzero_mask=True,
    #     overfit_small_batch=True,
    #     overfit_num_samples=128,
    # )
    # main(config_overfit)
    
    # Uncomment for normal training:
    # config = TrainConfig(
    #     root="EELS",
    #     batch_size=32,
    #     num_workers=4,
    #     lr=1e-3,
    #     epochs=10,
    #     use_nonzero_mask=True,
    # )
    # main(config)
