from pathlib import Path
import numpy as np
import torch
import tensorflow as tf

# Paper model was compiled with TF Addons F1Score; needed for load_model()
try:
    import tensorflow_addons as tfa
    _CUSTOM_OBJECTS = {"Addons>F1Score": tfa.metrics.F1Score}
except ImportError:
    _CUSTOM_OBJECTS = {}

from dataset import EELSDataset, make_dataloader
from metrics import compute_metrics, threshold_sweep


def probs_to_logits(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    probs = np.clip(probs, eps, 1.0 - eps)
    return np.log(probs / (1.0 - probs))


def main(
    model_dir: str = "models_from_paper/UNet",
    root: str = "full/EELS",
    split: str = "val",
    batch_size: int = 32,
    num_workers: int = 4,
    threshold: float | None = None,
):
    print(f"Loading TF model from: {model_dir}")
    model = tf.keras.models.load_model(
        model_dir,
        custom_objects=_CUSTOM_OBJECTS,
        compile=False,  # skip optimizer/metrics (we use our own metrics)
    )

    print("TF model input shape:", model.input_shape)
    print("TF model output shape:", model.output_shape)
    model.summary()

    dataset = EELSDataset(root, split)
    loader = make_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    all_logits = []
    all_targets = []

    for batch in loader:
        x = batch["x"]          # [B, 1, 3072]
        y = batch["y"]          # [B, 80]

        # TF Conv1D expects [B, N, C]
        x_tf = x.permute(0, 2, 1).cpu().numpy().astype(np.float32)  # [B, 3072, 1]

        # Model outputs sigmoid probabilities [B, 80]
        probs = model.predict(x_tf, verbose=0)
        logits = probs_to_logits(probs)

        all_logits.append(torch.from_numpy(logits).float())
        all_targets.append(y.float())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    if threshold is None:
        sweep = threshold_sweep(logits, targets)
        th = sweep["threshold_article_weighted"]
        metrics = sweep["metrics_at_article_weighted"]

        print(f"Split: {split}")
        print(f"threshold_article_weighted = {th:.2f}")
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
    else:
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
    p.add_argument("--model_dir", default="models_from_paper/UNet")
    p.add_argument("--root", default="full/EELS")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--threshold", type=float, default=None)
    args = p.parse_args()

    main(
        model_dir=args.model_dir,
        root=args.root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
    )
