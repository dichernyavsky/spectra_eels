"""
Evaluation script for single-element EELS classifier.
Load checkpoint, run on val/test, print accuracy and top-3 accuracy.
Optional: confusion matrix and top error pairs.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_single import EELSSingleElementDataset, make_single_dataloader
from model_single import EELSSingleModel

from train_single import accuracy, top3_accuracy


def main(
    checkpoint: str = "checkpoints_single/best.pt",
    root: str = "EELS",
    split: str = "val",
    batch_size: int = 64,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    device: str | None = None,
    show_confusion: bool = False,
    show_top_errors: int = 0,
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    ckpt = torch.load(checkpoint, map_location=dev)
    num_classes = ckpt.get("num_classes", 80)
    model = EELSSingleModel(num_classes=num_classes)
    model.load_state_dict(ckpt["model"])
    model = model.to(dev)
    model.eval()

    dataset = EELSSingleElementDataset(
        root, split=split, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )
    loader = make_single_dataloader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    all_logits = []
    all_y = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(dev)
            mask = batch["mask"].to(dev)
            out = model(x, mask)
            all_logits.append(out["logits"])
            all_y.append(batch["y"])
    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0).to(dev)

    acc = accuracy(logits, y)
    acc3 = top3_accuracy(logits, y)
    print(f"Split: {split}")
    print(f"  accuracy:    {acc:.4f}")
    print(f"  top-3 acc:   {acc3:.4f}")

    if show_confusion or show_top_errors > 0:
        pred = logits.argmax(dim=1)
        y_np = y.cpu().numpy()
        pred_np = pred.cpu().numpy()
        idx_to_element = dataset.idx_to_element

        if show_confusion:
            # Confusion matrix: [num_classes, num_classes]
            cm = np.zeros((num_classes, num_classes), dtype=np.int64)
            for i in range(len(y_np)):
                cm[y_np[i], pred_np[i]] += 1
            print("\nConfusion matrix (rows=true, cols=pred):")
            print("  (showing counts; rows/cols = class index)")
            np.set_printoptions(threshold=num_classes * num_classes, linewidth=120)
            print(cm)

        if show_top_errors > 0:
            wrong = (y_np != pred_np)
            wrong_true = y_np[wrong]
            wrong_pred = pred_np[wrong]
            # Count (true, pred) pairs
            pair_counts: dict[tuple[int, int], int] = {}
            for t, p in zip(wrong_true.tolist(), wrong_pred.tolist()):
                pair_counts[(t, p)] = pair_counts.get((t, p), 0) + 1
            sorted_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])[:show_top_errors]
            print(f"\nTop {show_top_errors} error pairs (true -> predicted, count):")
            for (t, p), count in sorted_pairs:
                t_name = idx_to_element.get(t, str(t))
                p_name = idx_to_element.get(p, str(p))
                print(f"  {t_name} -> {p_name}  count={count}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints_single/best.pt")
    p.add_argument("--root", default="EELS")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--confusion", action="store_true", help="Print confusion matrix")
    p.add_argument("--top_errors", type=int, default=0, metavar="N", help="Print top N error pairs")
    args = p.parse_args()
    main(
        checkpoint=args.checkpoint,
        root=args.root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        device=args.device,
        show_confusion=args.confusion,
        show_top_errors=args.top_errors,
    )
