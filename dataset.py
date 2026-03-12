from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import bisect
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Constants
ELEMENTS = [
    "Ag", "Al", "Ar", "As", "Au", "B", "Ba", "Be", "Bi", "Br",
    "C", "Ca", "Cd", "Ce", "Cl", "Co", "Cr", "Cs", "Cu", "Dy",
    "Er", "Eu", "F", "Fe", "Ga", "Gd", "Ge", "Hf", "Hg", "Ho",
    "I", "In", "Ir", "K", "Kr", "La", "Lu", "Mg", "Mn", "Mo",
    "N", "Na", "Nb", "Nd", "Ne", "Ni", "O", "Os", "P", "Pb",
    "Pd", "Pm", "Pr", "Pt", "Rb", "Re", "Rh", "Ru", "S", "Sb",
    "Sc", "Se", "Si", "Sm", "Sn", "Sr", "Ta", "Tb", "Tc", "Te",
    "Ti", "Tl", "Tm", "V", "W", "Xe", "Y", "Yb", "Zn", "Zr"
]

ELEMENT_TO_IDX = {el: i for i, el in enumerate(ELEMENTS)}

HDF5_SPECTRA_KEY = "spectra"
HDF5_LABEL_ID_KEY = "labels_identification"
HDF5_LABEL_QUANT_KEY = "labels_quantification"
SPECTRUM_LENGTH = 3072
NUM_CLASSES = 80


@dataclass
class EELSPaths:
    root: Path

    @property
    def train_dir(self) -> Path:
        return self.root / "trainingset"

    @property
    def val_dir(self) -> Path:
        return self.root / "validationset"

    @property
    def test_dir(self) -> Path:
        return self.root / "testset"

    @property
    def single_element_dir(self) -> Path:
        return self.root / "single_element_spectra"


def list_hdf5_files(directory: Path) -> List[Path]:
    files = sorted(directory.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found in {directory}")
    return files


def extract_element_from_filename(path: Path) -> str:
    # TRAIN_Ag.hdf5 -> Ag
    stem = path.stem
    return stem.split("_")[-1]


def inspect_hdf5_file(file_path: Path) -> None:
    with h5py.File(file_path, "r") as f:
        print(f"\nFILE: {file_path.name}")
        print("Keys:", list(f.keys()))
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                print(
                    f"  {key}: shape={obj.shape}, dtype={obj.dtype}"
                )


@dataclass
class FileRecord:
    path: Path
    element_name: str
    n_samples: int


def build_file_records(file_paths: List[Path]) -> List[FileRecord]:
    records: List[FileRecord] = []

    for fp in file_paths:
        with h5py.File(fp, "r") as f:
            n_samples = f[HDF5_SPECTRA_KEY].shape[0]

            # sanity checks
            assert f[HDF5_SPECTRA_KEY].shape[1] == SPECTRUM_LENGTH
            assert f[HDF5_LABEL_ID_KEY].shape == (n_samples, NUM_CLASSES)
            assert f[HDF5_LABEL_QUANT_KEY].shape == (n_samples, NUM_CLASSES)

        records.append(
            FileRecord(
                path=fp,
                element_name=extract_element_from_filename(fp),
                n_samples=n_samples,
            )
        )

    return records


def build_cumulative_sizes(records: List[FileRecord]) -> List[int]:
    cumulative = []
    total = 0
    for rec in records:
        total += rec.n_samples
        cumulative.append(total)
    return cumulative


def nonzero_mask_1d(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return (np.abs(x) > eps).astype(np.float32)


def nonzero_bounds_1d(x: np.ndarray, eps: float = 1e-12) -> Tuple[int, int]:
    nz = np.flatnonzero(np.abs(x) > eps)
    if len(nz) == 0:
        return 0, 0
    return int(nz[0]), int(nz[-1] + 1)  # [left, right)


@dataclass
class SpectrumPreprocessConfig:
    add_channel_dim: bool = True
    return_nonzero_mask: bool = True
    return_nonzero_bounds: bool = False
    return_window_mask: bool = False
    window_mask_margin: int = 50  # Margin inside valid window for window_mask
    dtype: np.dtype = np.float32


def preprocess_spectrum(
    spectrum: np.ndarray,
    cfg: SpectrumPreprocessConfig
) -> Dict[str, np.ndarray]:
    x = spectrum.astype(cfg.dtype, copy=False)

    out: Dict[str, np.ndarray] = {
        "spectrum": x
    }

    if cfg.return_nonzero_mask:
        out["nonzero_mask"] = nonzero_mask_1d(x)

    if cfg.return_nonzero_bounds:
        left, right = nonzero_bounds_1d(x)
        out["nonzero_bounds"] = np.array([left, right], dtype=np.int64)

    if cfg.return_window_mask:
        # Window mask based on nonzero_bounds with margin
        # This creates a mask that is slightly smaller than the full nonzero region
        # to prevent learning from exact window boundaries
        left, right = nonzero_bounds_1d(x)
        
        if right > left:  # Valid window exists
            # Add margin inside the valid window
            window_left = min(left + cfg.window_mask_margin, right)
            window_right = max(right - cfg.window_mask_margin, window_left)
            
            # Create mask: 1 inside window, 0 outside
            window_mask = np.zeros_like(x, dtype=np.float32)
            window_mask[window_left:window_right] = 1.0
        else:
            # No valid window, return all zeros
            window_mask = np.zeros_like(x, dtype=np.float32)
        
        out["window_mask"] = window_mask

    if cfg.add_channel_dim:
        out["spectrum"] = out["spectrum"][None, :]  # (1, 3072)
        if "nonzero_mask" in out:
            out["nonzero_mask"] = out["nonzero_mask"][None, :]  # (1, 3072)
        if "window_mask" in out:
            out["window_mask"] = out["window_mask"][None, :]  # (1, 3072)

    return out


class EELSHDF5Dataset(Dataset):
    """
    Lazy dataset: читает только нужный sample из нужного HDF5 файла.
    """

    def __init__(
        self,
        file_paths: List[Path],
        task: str = "identification",   # "identification" or "quantification"
        preprocess_cfg: Optional[SpectrumPreprocessConfig] = None,
        return_metadata: bool = False,
        return_index: bool = False,
    ):
        assert task in {"identification", "quantification"}

        self.file_records = build_file_records(file_paths)
        self.cumulative_sizes = build_cumulative_sizes(self.file_records)
        self.task = task
        self.preprocess_cfg = preprocess_cfg or SpectrumPreprocessConfig()
        self.return_metadata = return_metadata
        self.return_index = return_index

        # handles opened lazily per worker / process
        self._file_handles: Dict[int, h5py.File] = {}

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def _locate_index(self, index: int) -> Tuple[int, int]:
        file_idx = bisect.bisect_right(self.cumulative_sizes, index)
        prev_cum = 0 if file_idx == 0 else self.cumulative_sizes[file_idx - 1]
        row_idx = index - prev_cum
        return file_idx, row_idx

    def _get_h5_handle(self, file_idx: int) -> h5py.File:
        if file_idx not in self._file_handles:
            path = self.file_records[file_idx].path
            self._file_handles[file_idx] = h5py.File(path, "r")
        return self._file_handles[file_idx]

    def __getitem__(self, index: int):
        file_idx, row_idx = self._locate_index(index)
        record = self.file_records[file_idx]
        h5f = self._get_h5_handle(file_idx)

        spectrum = h5f[HDF5_SPECTRA_KEY][row_idx]  # (3072,)

        if self.task == "identification":
            label = h5f[HDF5_LABEL_ID_KEY][row_idx]   # (80,)
        else:
            label = h5f[HDF5_LABEL_QUANT_KEY][row_idx]  # (80,)

        proc = preprocess_spectrum(spectrum, self.preprocess_cfg)

        sample = {
            "x": torch.from_numpy(proc["spectrum"]).float(),
            "y": torch.from_numpy(label.astype(np.float32, copy=False)).float(),
        }

        if "nonzero_mask" in proc:
            sample["nonzero_mask"] = torch.from_numpy(proc["nonzero_mask"]).float()

        if "nonzero_bounds" in proc:
            sample["nonzero_bounds"] = torch.from_numpy(proc["nonzero_bounds"])

        if "window_mask" in proc:
            sample["window_mask"] = torch.from_numpy(proc["window_mask"]).float()

        if self.return_index:
            sample["index"] = index

        if self.return_metadata:
            sample["meta"] = {
                "global_index": index,
                "file_idx": file_idx,
                "row_idx": row_idx,
                "file_name": record.path.name,
                "query_element": record.element_name,
                "query_element_idx": ELEMENT_TO_IDX.get(record.element_name, -1),
            }

        return sample

    def reset_file_handles(self) -> None:
        """Close and clear all file handles. Safe for worker initialization."""
        self.close()

    def close(self) -> None:
        for f in self._file_handles.values():
            try:
                f.close()
            except Exception:
                pass
        self._file_handles.clear()

    def __del__(self):
        self.close()


def get_split_file_paths(root: str | Path, split: str) -> List[Path]:
    """Get list of HDF5 file paths for a given split."""
    paths = EELSPaths(Path(root))

    if split == "train":
        return list_hdf5_files(paths.train_dir)
    elif split == "val":
        return list_hdf5_files(paths.val_dir)
    elif split == "test":
        return list_hdf5_files(paths.test_dir)
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'.")


def make_split_dataset(
    root: str | Path,
    split: str,
    task: str = "identification",
    preprocess_cfg: Optional[SpectrumPreprocessConfig] = None,
    return_metadata: bool = False,
    return_index: bool = False,
) -> EELSHDF5Dataset:
    file_paths = get_split_file_paths(root, split)
    return EELSHDF5Dataset(
        file_paths=file_paths,
        task=task,
        preprocess_cfg=preprocess_cfg,
        return_metadata=return_metadata,
        return_index=return_index,
    )


def eels_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    x = torch.stack([item["x"] for item in batch], dim=0)
    y = torch.stack([item["y"] for item in batch], dim=0)

    out = {
        "x": x,   # [B, 1, 3072]
        "y": y,   # [B, 80]
    }

    if "nonzero_mask" in batch[0]:
        out["nonzero_mask"] = torch.stack(
            [item["nonzero_mask"] for item in batch], dim=0
        )

    if "nonzero_bounds" in batch[0]:
        out["nonzero_bounds"] = torch.stack(
            [item["nonzero_bounds"] for item in batch], dim=0
        )

    if "window_mask" in batch[0]:
        out["window_mask"] = torch.stack(
            [item["window_mask"] for item in batch], dim=0
        )

    if "index" in batch[0]:
        out["index"] = torch.tensor([item["index"] for item in batch], dtype=torch.long)

    if "meta" in batch[0]:
        out["meta"] = [item["meta"] for item in batch]

    return out


def worker_init_fn(worker_id: int) -> None:
    """Worker initialization function to reset file handles safely."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        if hasattr(dataset, 'reset_file_handles'):
            dataset.reset_file_handles()


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
        collate_fn=eels_collate_fn,
        persistent_workers=(num_workers > 0),
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )


def validate_sample(sample: Dict[str, torch.Tensor], task: str = "identification") -> None:
    """Validate sample structure and data types."""
    # Check x
    assert "x" in sample, "Sample must contain 'x'"
    x = sample["x"]
    assert x.shape == (1, SPECTRUM_LENGTH), f"Expected x.shape == (1, {SPECTRUM_LENGTH}), got {x.shape}"
    assert x.dtype == torch.float32, f"Expected x.dtype == float32, got {x.dtype}"
    assert torch.isfinite(x).all(), "x must contain only finite values"

    # Check y
    assert "y" in sample, "Sample must contain 'y'"
    y = sample["y"]
    assert y.shape == (NUM_CLASSES,), f"Expected y.shape == ({NUM_CLASSES},), got {y.shape}"
    assert y.dtype == torch.float32, f"Expected y.dtype == float32, got {y.dtype}"

    # Check nonzero_bounds if present
    if "nonzero_bounds" in sample:
        bounds = sample["nonzero_bounds"]
        assert len(bounds) == 2, f"nonzero_bounds must have length 2, got {len(bounds)}"
        assert bounds[0] >= 0 and bounds[0] <= bounds[1] <= SPECTRUM_LENGTH, \
            f"Invalid nonzero_bounds: {bounds}, must satisfy 0 <= left <= right <= {SPECTRUM_LENGTH}"

    # Check labels_identification are binary
    if task == "identification":
        assert torch.all((y == 0) | (y == 1)), \
            f"Labels for identification task must be binary (0/1), found values: {torch.unique(y)}"


def inspect_dataset_samples(dataset: EELSHDF5Dataset, n: int = 10, seed: int = 42) -> None:
    """Inspect random samples from dataset."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    indices = random.sample(range(len(dataset)), min(n, len(dataset)))

    print(f"\n{'='*60}")
    print(f"Inspecting {len(indices)} random samples from dataset")
    print(f"{'='*60}\n")

    for i, idx in enumerate(indices, 1):
        sample = dataset[idx]
        print(f"Sample {i} (index {idx}):")
        print(f"  x shape: {sample['x'].shape}")
        print(f"  x min/max: {sample['x'].min():.6f} / {sample['x'].max():.6f}")

        if "nonzero_mask" in sample:
            nz_count = sample["nonzero_mask"].sum().item()
            print(f"  Non-zero points: {nz_count} / {SPECTRUM_LENGTH}")

        if "nonzero_bounds" in sample:
            bounds = sample["nonzero_bounds"]
            print(f"  Non-zero bounds: [{bounds[0].item()}, {bounds[1].item()})")

        y_sum = sample["y"].sum().item()
        y_pos = (sample["y"] > 0).sum().item()
        print(f"  Label sum: {y_sum:.2f}, Positive labels: {y_pos}")

        if "meta" in sample:
            meta = sample["meta"]
            query_elem = meta["query_element"]
            query_idx = meta["query_element_idx"]
            print(f"  Query element: {query_elem} (idx={query_idx})")
            if query_idx >= 0:
                label_value = sample["y"][query_idx].item()
                print(f"  Query element in labels: {label_value > 0} (value={label_value})")

        print()


def build_dataset_index_summary(records: List[FileRecord]) -> Dict:
    """Build summary statistics for dataset file records."""
    if not records:
        return {
            "num_files": 0,
            "total_samples": 0,
            "min_samples_per_file": 0,
            "max_samples_per_file": 0,
            "mean_samples_per_file": 0.0,
            "elements": [],
        }

    n_samples_list = [rec.n_samples for rec in records]
    elements = sorted(set(rec.element_name for rec in records))

    return {
        "num_files": len(records),
        "total_samples": sum(n_samples_list),
        "min_samples_per_file": min(n_samples_list),
        "max_samples_per_file": max(n_samples_list),
        "mean_samples_per_file": sum(n_samples_list) / len(n_samples_list),
        "elements": elements,
    }


def print_dataset_summary(records: List[FileRecord]) -> None:
    """Print formatted dataset summary."""
    summary = build_dataset_index_summary(records)

    print(f"\n{'='*60}")
    print("Dataset Summary")
    print(f"{'='*60}")
    print(f"Number of files: {summary['num_files']}")
    print(f"Total samples: {summary['total_samples']:,}")
    print(f"Samples per file:")
    print(f"  Min: {summary['min_samples_per_file']:,}")
    print(f"  Max: {summary['max_samples_per_file']:,}")
    print(f"  Mean: {summary['mean_samples_per_file']:.1f}")
    print(f"Number of unique elements: {len(summary['elements'])}")
    print(f"{'='*60}\n")


def demo(root: str | Path) -> None:
    """Demo function for testing dataset."""
    preprocess_cfg = SpectrumPreprocessConfig(
        add_channel_dim=True,
        return_nonzero_mask=True,
        return_nonzero_bounds=True,
    )

    train_ds = make_split_dataset(
        root=root,
        split="train",
        task="identification",
        preprocess_cfg=preprocess_cfg,
        return_metadata=True,
    )

    print("Train dataset size:", len(train_ds))
    print_dataset_summary(train_ds.file_records)

    sample = train_ds[0]
    validate_sample(sample, task="identification")
    print("Sample validation passed!")

    inspect_dataset_samples(train_ds, n=5)

    train_loader = make_dataloader(
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=4,
    )

    batch = next(iter(train_loader))
    print(f"Batch x shape: {batch['x'].shape}")  # [32, 1, 3072]
    print(f"Batch y shape: {batch['y'].shape}")  # [32, 80]
