"""
Minimal EELS dataset: reads HDF5 with spectra and labels_identification only.
"""
from pathlib import Path
from typing import Dict, List

import bisect
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

SPECTRA_KEY = "spectra"
LABELS_KEY = "labels_identification"
SPECTRUM_LENGTH = 3072
NUM_CLASSES = 80


def _split_dir(root: Path, split: str) -> Path:
    if split == "train":
        return root / "trainingset"
    if split == "val":
        return root / "validationset"
    if split == "test":
        return root / "testset"
    raise ValueError(f"split must be 'train', 'val', or 'test'; got {split!r}")


def _list_hdf5(directory: Path) -> List[Path]:
    directory = directory.resolve()
    files = sorted(directory.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(
            f"No .hdf5 files in {directory}. "
            "Create EELS/trainingset/ (and validationset/, testset/) with .hdf5 files, "
            "or pass root=/path/to/your/EELS to EELSDataset and Config."
        )
    return files


def _build_records(file_paths: List[Path]) -> List[tuple]:
    records = []
    for fp in file_paths:
        with h5py.File(fp, "r") as f:
            n = f[SPECTRA_KEY].shape[0]
            assert f[SPECTRA_KEY].shape[1] == SPECTRUM_LENGTH
            assert f[LABELS_KEY].shape == (n, NUM_CLASSES)
        records.append((fp, n))
    return records


def _cumulative_sizes(records: List[tuple]) -> List[int]:
    out = []
    total = 0
    for _, n in records:
        total += n
        out.append(total)
    return out


class EELSDataset(Dataset):
    """EELS dataset from HDF5: only spectra and labels_identification."""

    def __init__(self, root: str | Path, split: str):
        self._handles: Dict[int, h5py.File] = {}
        self.root = Path(root)
        self.split = split
        dir_path = _split_dir(self.root, split)
        self._file_paths = _list_hdf5(dir_path)
        self._records = _build_records(self._file_paths)
        self._cumulative = _cumulative_sizes(self._records)

    def __len__(self) -> int:
        return self._cumulative[-1] if self._cumulative else 0

    def _locate(self, index: int) -> tuple:
        file_idx = bisect.bisect_right(self._cumulative, index)
        prev = 0 if file_idx == 0 else self._cumulative[file_idx - 1]
        row = index - prev
        return file_idx, row

    def _get_handle(self, file_idx: int) -> h5py.File:
        if file_idx not in self._handles:
            self._handles[file_idx] = h5py.File(self._records[file_idx][0], "r")
        return self._handles[file_idx]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        file_idx, row = self._locate(index)
        h5f = self._get_handle(file_idx)
        x = np.asarray(h5f[SPECTRA_KEY][row], dtype=np.float32)
        y = np.asarray(h5f[LABELS_KEY][row], dtype=np.float32)
        mask = (x != 0).astype(np.float32)
        # Min-max normalize each spectrum to [0, 1]
        x = x / x.max() 

        x = x[np.newaxis, :]  # [1, 3072]
        mask = mask[np.newaxis, :]  # [1, 3072]
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "mask": torch.from_numpy(mask),
        }

    def __del__(self):
        for f in getattr(self, "_handles", {}).values():
            try:
                f.close()
            except Exception:
                pass
        if hasattr(self, "_handles"):
            self._handles.clear()


def _collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "y": torch.stack([b["y"] for b in batch], dim=0),
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
    }


def _worker_init(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None and hasattr(worker_info.dataset, "_handles"):
        worker_info.dataset._handles.clear()


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate,
        persistent_workers=(num_workers > 0),
        worker_init_fn=_worker_init if num_workers > 0 else None,
    )
