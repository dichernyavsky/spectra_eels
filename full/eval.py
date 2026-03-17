"""
Evaluation script for EELS multi-label model:
- loads checkpoint
- uses threshold selected on validation set (threshold_article)
- reports full set of multi-label metrics on the chosen split.
"""
from pathlib import Path

import torch

from dataset import EELSDataset, make_dataloader
from model import EELSModel
from metrics import compute_metrics


def main(
    root: str = "EELS",
    checkpoint: str = "checkpoints/best.pt",
    split: str = "test",
    batch_size: int = 32,
    num_workers: int = 4,
    threshold: float | None = None,
    device: str | None = None,
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    model = EELSModel()
    ckpt = torch.load(checkpoint, map_location=dev)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        # By default use threshold_article_weighted chosen on validation set.
        if threshold is None:
            ckpt_thr = ckpt.get("threshold_article_weighted", None)
            if ckpt_thr is None:
                threshold = 0.5
            else:
                threshold = float(ckpt_thr)
    else:
        model.load_state_dict(ckpt)
        if threshold is None:
            threshold = 0.5

    model = model.to(dev)
    model.eval()

    dataset = EELSDataset(root, split)
    loader = make_dataloader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    all_logits = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(dev)
            mask = batch["mask"].to(dev)
            out = model(x, mask)
            all_logits.append(out["logits"])
            all_targets.append(batch["y"])
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0).to(dev)

    metrics = compute_metrics(logits, targets, threshold=threshold)
    print(f"Split: {split}  threshold: {threshold:.2f}")
    print(
        "  micro  "
        f"precision={metrics['micro_precision']:.4f}  "
        f"recall={metrics['micro_recall']:.4f}  "
        f"f1={metrics['micro_f1']:.4f}"
    )
    print(
        "  macro  "
        f"precision={metrics['macro_precision']:.4f}  "
        f"recall={metrics['macro_recall']:.4f}  "
        f"f1={metrics['macro_f1']:.4f}"
    )
    print(
        "  weighted  "
        f"precision={metrics['weighted_precision']:.4f}  "
        f"recall={metrics['weighted_recall']:.4f}  "
        f"f1={metrics['weighted_f1']:.4f}"
    )
    print(
        "  other  "
        f"exact_match={metrics['exact_match']:.4f}  "
        f"rmse={metrics['rmse']:.4f}"
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--root", default="EELS")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    # If not provided, uses threshold_article from checkpoint.
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    main(
        root=args.root,
        checkpoint=args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
        device=args.device,
    )
