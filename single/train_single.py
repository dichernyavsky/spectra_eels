"""
Training script for single-element EELS classifier.
CrossEntropyLoss, accuracy and top-3 accuracy, save best by val accuracy.
"""
from dataclasses import dataclass
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_single import EELSSingleElementDataset, make_single_dataloader
from model_single import EELSSingleModel


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Fraction of correct predictions."""
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def top3_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Fraction of samples where true class is in top-3 predictions."""
    top3 = logits.topk(3, dim=1).indices  # [B, 3]
    correct = (top3 == y[:, None]).any(dim=1).float()
    return correct.mean().item()


@dataclass
class Config:
    root: str = "EELS"
    batch_size: int = 64
    num_workers: int = 4
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "checkpoints_single"
    seed: int = 42
    val_ratio: float = 0.1
    test_ratio: float = 0.1


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    n = 0
    all_logits: list[torch.Tensor] = []
    all_y: list[torch.Tensor] = []
    total_batches = len(loader)
    # print progress several times per epoch (about 10 updates)
    progress_step = max(1, total_batches // 10) 
    epoch_start_time = time.time()
    for batch_idx, batch in enumerate(loader, start=1):
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
        all_y.append(y)
        if batch_idx % progress_step == 0 or batch_idx == total_batches:
            # show in-epoch training progress with ETA
            elapsed = time.time() - epoch_start_time
            done_frac = batch_idx / total_batches if total_batches else 0.0
            eta = (elapsed / done_frac - elapsed) if done_frac > 0 else 0.0
            elapsed_min = int(elapsed // 60)
            elapsed_sec = int(elapsed % 60)
            eta_min = int(eta // 60)
            eta_sec = int(eta % 60)
            print(
                f"  batch {batch_idx}/{total_batches} - "
                f"loss={loss.item():.4f} - "
                f"elapsed={elapsed_min:02d}:{elapsed_sec:02d} - "
                f"eta={eta_min:02d}:{eta_sec:02d}"
            )
    avg_loss = total_loss / n if n else 0.0
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_y, dim=0)
    acc = accuracy(logits, targets)
    acc3 = top3_accuracy(logits, targets)
    return avg_loss, acc, acc3


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    all_logits = []
    all_y = []
    total_batches = len(loader)
    # show validation progress similar to training
    progress_step = max(1, total_batches // 100)
    epoch_start_time = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            out = model(x, mask)
            loss = criterion(out["logits"], y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            all_logits.append(out["logits"])
            all_y.append(y)
            if batch_idx % progress_step == 0 or batch_idx == total_batches:
                elapsed = time.time() - epoch_start_time
                done_frac = batch_idx / total_batches if total_batches else 0.0
                eta = (elapsed / done_frac - elapsed) if done_frac > 0 else 0.0
                elapsed_min = int(elapsed // 60)
                elapsed_sec = int(elapsed % 60)
                eta_min = int(eta // 60)
                eta_sec = int(eta % 60)
                print(
                    f"  [val] batch {batch_idx}/{total_batches} - "
                    f"loss={loss.item():.4f} - "
                    f"elapsed={elapsed_min:02d}:{elapsed_sec:02d} - "
                    f"eta={eta_min:02d}:{eta_sec:02d}"
                )
    avg_loss = total_loss / n if n else 0.0
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_y, dim=0)
    acc = accuracy(logits, targets)
    acc3 = top3_accuracy(logits, targets)
    return avg_loss, acc, acc3


def main(cfg: Config | None = None) -> None:
    if cfg is None:
        cfg = Config()
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    save_path = Path(cfg.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    train_ds = EELSSingleElementDataset(
        cfg.root, split="train", val_ratio=cfg.val_ratio, test_ratio=cfg.test_ratio, seed=cfg.seed
    )
    val_ds = EELSSingleElementDataset(
        cfg.root, split="val", val_ratio=cfg.val_ratio, test_ratio=cfg.test_ratio, seed=cfg.seed
    )
    num_classes = train_ds.num_classes

    train_loader = make_single_dataloader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = make_single_dataloader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    model = EELSSingleModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_acc = -1.0
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc, train_acc3 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_acc3 = evaluate(model, val_loader, criterion, device)
        print(
            f"epoch {epoch}  train loss={train_loss:.4f}  train acc={train_acc:.4f}  train top3={train_acc3:.4f}"
        )
        print(
            f"         val   loss={val_loss:.4f}  val   acc={val_acc:.4f}  val   top3={val_acc3:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "num_classes": num_classes,
                },
                save_path / "best.pt",
            )
            print(f"  -> saved best (val_acc={best_val_acc:.4f})")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    p.add_argument("--root", default=None)
    p.add_argument("--save_dir", default=None)
    p.add_argument("--batch_size", type=int, default=None, help="Override config batch size")
    p.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override number of DataLoader workers",
    )
    args = p.parse_args()
    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.root is not None:
        cfg.root = args.root
    if args.save_dir is not None:
        cfg.save_dir = args.save_dir
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    main(cfg)
