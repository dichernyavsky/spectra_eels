"""
Minimal training script for EELS multi-label classification.

Config: use --config path/to/config.yaml or --config paper_unet to load a saved
model + training protocol. Default (no --config) uses the same baseline as configs/paper_unet.yaml.
"""
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from dataset import EELSDataset, make_dataloader
from config import TrainConfig, load_config, get_model
from losses import build_loss
from metrics import compute_metrics, threshold_sweep


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    threshold: float = 0.5,
    max_batches: int | None = None,
) -> tuple[float, dict]:
    model.train()
    total_loss = 0.0
    n = 0
    all_logits = []
    all_targets = []
    total_batches = len(loader)
    # Show several progress updates per epoch (similar to single-label training).
    progress_step = max(1, total_batches // 10)
    epoch_start_time = time.time()
    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)
        out = model(x, mask)
        loss = criterion(out["logits"], y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        all_logits.append(out["logits"].detach())
        all_targets.append(y)
        if batch_idx % progress_step == 0 or batch_idx == total_batches:
            # In-epoch training progress with rough ETA.
            elapsed = time.time() - epoch_start_time
            done_frac = batch_idx / total_batches if total_batches else 0.0
            eta = (elapsed / done_frac - elapsed) if done_frac > 0 else 0.0
            elapsed_min = int(elapsed // 60)
            elapsed_sec = int(elapsed % 60)
            eta_min = int(eta // 60)
            eta_sec = int(eta % 60)
            print(
                f"  [train] batch {batch_idx}/{total_batches} - "
                f"loss={loss.item():.4f} - "
                f"elapsed={elapsed_min:02d}:{elapsed_sec:02d} - "
                f"eta={eta_min:02d}:{eta_sec:02d}"
            )
    avg_loss = total_loss / n if n else 0.0
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(logits, targets, threshold=threshold)
    return avg_loss, metrics, logits, targets


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    threshold: float,
    max_batches: int | None = None,
) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    n = 0
    all_logits = []
    all_targets = []
    total_batches = len(loader)
    # Validation progress updates (more sparse than training).
    progress_step = max(1, total_batches // 100)
    epoch_start_time = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            out = model(x, mask)
            loss = criterion(out["logits"], y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            all_logits.append(out["logits"])
            all_targets.append(y)
            if batch_idx % progress_step == 0 or batch_idx == total_batches:
                elapsed = time.time() - epoch_start_time
                done_frac = batch_idx / total_batches if total_batches else 0.0
                eta = (elapsed / done_frac - elapsed) if done_frac > 0 else 0.0
                elapsed_min = int(elapsed // 60)
                elapsed_sec = int(elapsed % 60)
                eta_min = int(eta // 60)
                eta_sec = int(eta % 60)
                print(
                    f"  [val]   batch {batch_idx}/{total_batches} - "
                    f"loss={loss.item():.4f} - "
                    f"elapsed={elapsed_min:02d}:{elapsed_sec:02d} - "
                    f"eta={eta_min:02d}:{eta_sec:02d}"
                )
    avg_loss = total_loss / n if n else 0.0
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(logits, targets, threshold=threshold)
    return avg_loss, metrics, logits, targets


def main(cfg: TrainConfig | None = None) -> None:
    if cfg is None:
        cfg = TrainConfig()
    # Use CPU if config says cuda but no GPU is available
    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"
        print("CUDA not available, using CPU")
    if cfg.smoke:
        # In smoke mode run fast to validate that the pipeline works.
        cfg.epochs = 1
        cfg.num_workers = 0
    device = torch.device(cfg.device)
    print(f"Using device: {device}")
    save_path = Path(cfg.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    train_ds = EELSDataset(cfg.root, "train")
    val_ds = EELSDataset(cfg.root, "val")
    train_loader = make_dataloader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = make_dataloader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    test_ds = EELSDataset(cfg.root, "test")
    test_loader = make_dataloader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    # Train class statistics for pos_weight and bias init
    all_y = []
    for batch in train_loader:
        all_y.append(batch["y"])
        if cfg.smoke and len(all_y) >= 2:
            break
    y_train = torch.cat(all_y, dim=0).float()
    prevalence = y_train.mean(dim=0)
    p = prevalence.clamp(1e-4, 1.0 - 1e-4)
    pos_weight = (1.0 - p) / p

    model = get_model(cfg.model_name, **cfg.model_kwargs).to(device)
    if hasattr(model, "init_bias_from_prevalence"):
        model.init_bias_from_prevalence(prevalence.to(device))
    if cfg.multi_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: total={n_params:,}, trainable={n_trainable:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        threshold=0.025,
        threshold_mode="abs",
    )
    criterion = build_loss(
        cfg.loss_mode,
        lambda_soft_f1=cfg.lambda_soft_f1,
        pos_weight=pos_weight.to(device),
    )

    # Recreate train_loader for training (previous pass was consumed for stats)
    train_loader = make_dataloader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )

    max_train = cfg.smoke_max_train_batches if cfg.smoke else cfg.steps_per_epoch
    max_val = cfg.smoke_max_val_batches if cfg.smoke else None

    # Selection uses weighted F1 at the article (weighted) threshold on validation set.
    best_val_weighted_f1_article = -1.0
    best_threshold_article_weighted = cfg.threshold
    best_threshold_article_micro = cfg.threshold
    best_threshold_best_wf1 = cfg.threshold

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_metrics, _, _ = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            threshold=cfg.threshold,
            max_batches=max_train,
        )
        # Monitor LR based on train weighted F1 at threshold cfg.threshold (paper uses 0.8).
        scheduler.step(train_metrics["weighted_f1"])

        val_loss, val_metrics_default, val_logits, val_targets = evaluate(
            model,
            val_loader,
            criterion,
            device,
            cfg.threshold,
            max_batches=max_val,
        )

        # Threshold sweep on validation set.
        sweep = threshold_sweep(val_logits, val_targets)
        th_article_weighted = sweep["threshold_article_weighted"]
        m_article_weighted = sweep["metrics_at_article_weighted"]
        th_article_micro = sweep["threshold_article_micro"]
        m_article_micro = sweep["metrics_at_article_micro"]
        th_best_wf1 = sweep["threshold_best_weighted_f1"]
        m_best_wf1 = sweep["metrics_at_best_weighted_f1"]

        print(f"epoch {epoch}")
        print(
            "  train  "
            f"loss={train_loss:.4f}  "
            f"micro_f1={train_metrics['micro_f1']:.4f}  "
            f"macro_f1={train_metrics['macro_f1']:.4f}  "
            f"weighted_f1={train_metrics['weighted_f1']:.4f}  "
            f"weighted_precision={train_metrics['weighted_precision']:.4f}  "
            f"weighted_recall={train_metrics['weighted_recall']:.4f}  "
            f"exact_match={train_metrics['exact_match']:.4f}  "
            f"rmse={train_metrics['rmse']:.4f}"
        )
        print(
            "  val@default  "
            f"loss={val_loss:.4f}  "
            f"micro_f1={val_metrics_default['micro_f1']:.4f}  "
            f"macro_f1={val_metrics_default['macro_f1']:.4f}  "
            f"weighted_f1={val_metrics_default['weighted_f1']:.4f}  "
            f"weighted_precision={val_metrics_default['weighted_precision']:.4f}  "
            f"weighted_recall={val_metrics_default['weighted_recall']:.4f}  "
            f"exact_match={val_metrics_default['exact_match']:.4f}  "
            f"rmse={val_metrics_default['rmse']:.4f}  "
            f"threshold={cfg.threshold:.2f}"
        )
        print(
            "  val@article(weighted)  "
            f"threshold={th_article_weighted:.2f}  "
            f"micro_precision={m_article_weighted['micro_precision']:.4f}  "
            f"micro_recall={m_article_weighted['micro_recall']:.4f}  "
            f"micro_f1={m_article_weighted['micro_f1']:.4f}  "
            f"macro_precision={m_article_weighted['macro_precision']:.4f}  "
            f"macro_recall={m_article_weighted['macro_recall']:.4f}  "
            f"macro_f1={m_article_weighted['macro_f1']:.4f}  "
            f"weighted_precision={m_article_weighted['weighted_precision']:.4f}  "
            f"weighted_recall={m_article_weighted['weighted_recall']:.4f}  "
            f"weighted_f1={m_article_weighted['weighted_f1']:.4f}  "
            f"exact_match={m_article_weighted['exact_match']:.4f}  "
            f"rmse={m_article_weighted['rmse']:.4f}"
        )
        print(
            "  val@article(micro)    "
            f"threshold={th_article_micro:.2f}  "
            f"micro_precision={m_article_micro['micro_precision']:.4f}  "
            f"micro_recall={m_article_micro['micro_recall']:.4f}  "
            f"micro_f1={m_article_micro['micro_f1']:.4f}  "
            f"macro_precision={m_article_micro['macro_precision']:.4f}  "
            f"macro_recall={m_article_micro['macro_recall']:.4f}  "
            f"macro_f1={m_article_micro['macro_f1']:.4f}  "
            f"weighted_precision={m_article_micro['weighted_precision']:.4f}  "
            f"weighted_recall={m_article_micro['weighted_recall']:.4f}  "
            f"weighted_f1={m_article_micro['weighted_f1']:.4f}  "
            f"exact_match={m_article_micro['exact_match']:.4f}  "
            f"rmse={m_article_micro['rmse']:.4f}"
        )
        print(
            "  val@best_wf1 "
            f"threshold_best_wf1={th_best_wf1:.2f}  "
            f"weighted_f1={m_best_wf1['weighted_f1']:.4f}  "
            f"weighted_precision={m_best_wf1['weighted_precision']:.4f}  "
            f"weighted_recall={m_best_wf1['weighted_recall']:.4f}  "
            f"exact_match={m_best_wf1['exact_match']:.4f}  "
            f"rmse={m_best_wf1['rmse']:.4f}"
        )

        # Test evaluation (monitoring only; threshold from val sweep).
        _, test_metrics, _, _ = evaluate(
            model,
            test_loader,
            criterion,
            device,
            threshold=th_article_weighted,
            max_batches=max_val,
        )
        print(
            "  test@article(weighted)  "
            f"weighted_precision={test_metrics['weighted_precision']:.4f}  "
            f"weighted_recall={test_metrics['weighted_recall']:.4f}  "
            f"weighted_f1={test_metrics['weighted_f1']:.4f}  "
            f"exact_match={test_metrics['exact_match']:.4f}  "
            f"rmse={test_metrics['rmse']:.4f}"
        )

        # Save best model by weighted F1 at article threshold.
        if m_article_weighted["weighted_f1"] > best_val_weighted_f1_article:
            best_val_weighted_f1_article = m_article_weighted["weighted_f1"]
            best_threshold_article_weighted = th_article_weighted
            best_threshold_article_micro = th_article_micro
            best_threshold_best_wf1 = th_best_wf1
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(
                {
                    "model": state_dict,
                    "epoch": epoch,
                    "best_val_weighted_f1_article": best_val_weighted_f1_article,
                    "threshold_article_weighted": best_threshold_article_weighted,
                    "threshold_article_micro": best_threshold_article_micro,
                    "threshold_best_weighted_f1": best_threshold_best_wf1,
                    "num_classes": 80,
                },
                save_path / "best.pt",
            )
            print(
                "  -> saved best checkpoint "
                f"(weighted_f1_article={best_val_weighted_f1_article:.4f}, "
                f"threshold_article_weighted={best_threshold_article_weighted:.2f}, "
                f"threshold_article_micro={best_threshold_article_micro:.2f}, "
                f"threshold_best_wf1={best_threshold_best_wf1:.2f})"
            )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config or preset name (e.g. paper_unet). Default: built-in baseline.",
    )
    p.add_argument("--smoke", action="store_true", help="Smoke run: 1 epoch, few batches")
    p.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    p.add_argument("--num_workers", type=int, default=None, help="Override DataLoader workers")
    p.add_argument("--multi_gpu", action="store_true", help="Use all visible GPUs (DataParallel)")
    args = p.parse_args()

    if args.config is not None:
        cfg = load_config(args.config)
        print(f"Loaded config: {args.config}")
    else:
        cfg = TrainConfig()

    if args.smoke:
        cfg.smoke = True
    if args.multi_gpu:
        cfg.multi_gpu = True
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    main(cfg)
