"""
Single-element EELS dataset: reads pre-built HDF5 from EELS_single/.
Expects root/trainingset/single_train.hdf5, root/validationset/single_val.hdf5,
root/testset/single_test.hdf5 (created by build_single_dataset.py).
No on-the-fly filtering. Does not use single_element_spectra/.
"""
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

SPECTRA_KEY = "spectra"
CLASS_ID_KEY = "class_id"
SPECTRUM_LENGTH = 3072
NUM_CLASSES = 80

SPLIT_FILES = {
    "train": ("trainingset", "single_train.hdf5"),
    "val": ("validationset", "single_val.hdf5"),
    "test": ("testset", "single_test.hdf5"),
}


def _split_path(root: Path, split: str) -> Path:
    dir_name, filename = SPLIT_FILES[split]
    return root / dir_name / filename


def print_class_distribution(dataset: "EELSSingleElementDataset") -> None:
    """Print total samples and per-class distribution."""
    n = len(dataset)
    print(f"Total samples: {n}")
    if n == 0:
        return
    counts = dataset._class_counts
    for c in sorted(counts.keys()):
        name = dataset.idx_to_element.get(c, str(c))
        print(f"  class {c} ({name}): {counts[c]}")


class EELSSingleElementDataset(Dataset):
    """
    Single-element EELS dataset from pre-built HDF5.
    Reads root/{trainingset,validationset,testset}/single_*.hdf5.
    Each file has: spectra (M, 3072), class_id (M,). No filtering on the fly.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.root = Path(root)
        self.split = split
        self._handle: h5py.File | None = None
        # val_ratio, test_ratio, seed ignored (split = single file per split)

        path = _split_path(self.root, split)
        if not path.is_file():
            raise FileNotFoundError(
                f"Single-label file not found: {path}. "
                "Run build_single_dataset.py to create EELS_single/ first, then use root='EELS_single'."
            )
        self._path = path
        with h5py.File(path, "r") as f:
            if SPECTRA_KEY not in f or CLASS_ID_KEY not in f:
                raise KeyError(f"Expected '{SPECTRA_KEY}' and '{CLASS_ID_KEY}' in {path}")
            spectra_ds = f[SPECTRA_KEY]
            class_ds = f[CLASS_ID_KEY]
            self._n = spectra_ds.shape[0]
            assert class_ds.shape[0] == self._n
            class_id = np.asarray(class_ds)
        self._handle = None  # open lazily in __getitem__ for multiprocessing safety
        self.num_classes = NUM_CLASSES
        self.idx_to_element = {}
        self.element_to_idx = {}
        self._class_counts = {}
        for c in range(NUM_CLASSES):
            cnt = int((class_id == c).sum())
            if cnt > 0:
                self._class_counts[c] = cnt

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self._handle is None:
            self._handle = h5py.File(self._path, "r")
        x = np.asarray(self._handle[SPECTRA_KEY][index], dtype=np.float32)
        class_index = int(self._handle[CLASS_ID_KEY][index])
        if x.ndim == 1:
            x = x[np.newaxis, :]
        else:
            x = x.reshape(1, -1)
        x = x.reshape(1, SPECTRUM_LENGTH)
        mask = (x != 0).astype(np.float32)
        return {
            "x": torch.from_numpy(x),
            "mask": torch.from_numpy(mask),
            "y": class_index,
        }

    def __del__(self):
        if getattr(self, "_handle", None) is not None:
            try:
                self._handle.close()
            except Exception:
                pass
            self._handle = None


def _collate_single(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "y": torch.tensor([b["y"] for b in batch], dtype=torch.long),
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
    }


def _worker_init_single(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None and hasattr(worker_info.dataset, "_handle"):
        if worker_info.dataset._handle is not None:
            try:
                worker_info.dataset._handle.close()
            except Exception:
                pass
            worker_info.dataset._handle = None


def make_single_dataloader(
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
        collate_fn=_collate_single,
        persistent_workers=(num_workers > 0),
        worker_init_fn=_worker_init_single if num_workers > 0 else None,
    )
