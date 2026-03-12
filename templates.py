from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bisect

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import (
    ELEMENTS,
    ELEMENT_TO_IDX,
    EELSPaths,
    FileRecord,
    SpectrumPreprocessConfig,
    build_file_records,
    build_cumulative_sizes,
    list_hdf5_files,
    preprocess_spectrum,
    HDF5_SPECTRA_KEY,
    HDF5_LABEL_ID_KEY,
    NUM_CLASSES,
)


class EELSSingleElementTemplateDataset(Dataset):
    """
    Датасет single_element_spectra, полезен как банк шаблонов.
    """

    def __init__(
        self,
        file_paths: List[Path],
        preprocess_cfg: Optional[SpectrumPreprocessConfig] = None,
    ):
        self.file_records = build_file_records(file_paths)
        self.cumulative_sizes = build_cumulative_sizes(self.file_records)
        self.preprocess_cfg = preprocess_cfg or SpectrumPreprocessConfig()
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

        spectrum = h5f[HDF5_SPECTRA_KEY][row_idx]
        label_id = h5f[HDF5_LABEL_ID_KEY][row_idx]

        # Extract actual element ONLY from label (not from filename!)
        # Ground truth is the label, not the filename
        label_sum = label_id.sum()
        nonzero_indices = np.where(label_id > 0)[0]
        
        # Check if it's a valid single-element label
        is_valid = (label_sum == 1.0) and (len(nonzero_indices) == 1)
        
        if is_valid:
            actual_element_idx = int(nonzero_indices[0])
            actual_element_name = ELEMENTS[actual_element_idx] if actual_element_idx < len(ELEMENTS) else "UNKNOWN"
        else:
            # Invalid single-element label - set to -1
            # Do NOT use filename as ground truth
            actual_element_idx = -1
            actual_element_name = "INVALID"

        proc = preprocess_spectrum(spectrum, self.preprocess_cfg)

        return {
            "x": torch.from_numpy(proc["spectrum"]).float(),
            "y": torch.from_numpy(label_id.astype(np.float32, copy=False)).float(),
            "element_name": actual_element_name,  # From label only
            "element_idx": actual_element_idx,     # From label only (-1 if invalid)
            "row_idx": row_idx,
            "file_name": record.path.name,  # Metadata only, not ground truth
            "is_valid_single_element": is_valid,
        }

    def close(self) -> None:
        for f in self._file_handles.values():
            try:
                f.close()
            except Exception:
                pass
        self._file_handles.clear()

    def __del__(self):
        self.close()


def make_single_element_dataset(
    root: str | Path,
    preprocess_cfg: Optional[SpectrumPreprocessConfig] = None,
) -> EELSSingleElementTemplateDataset:
    paths = EELSPaths(Path(root))
    file_paths = list_hdf5_files(paths.single_element_dir)
    return EELSSingleElementTemplateDataset(
        file_paths=file_paths,
        preprocess_cfg=preprocess_cfg,
    )


def stack_templates_by_element(
    dataset: EELSSingleElementTemplateDataset,
    max_templates_per_element: Optional[int] = None,
    strict_single_label: bool = True,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Stack templates grouped by element index.
    
    Args:
        dataset: Single element template dataset
        max_templates_per_element: Maximum templates per element
        strict_single_label: If True, only use samples with label.sum() == 1
    
    Returns:
        Dictionary mapping element_idx to stacked tensors:
        {
            element_idx: {
                "spectra": Tensor[num_templates, 1, 3072],
                "labels": Tensor[num_templates, 80],
                "mask": Tensor[num_templates, 1, 3072] (if enabled),
                "bounds": Tensor[num_templates, 2] (if enabled),
                "element_name": str,
            }
        }
    """
    templates_by_element: Dict[int, List[Dict[str, torch.Tensor]]] = {}

    for idx in range(len(dataset)):
        sample = dataset[idx]
        element_idx = sample["element_idx"]

        # Skip invalid single-element templates
        if strict_single_label:
            # Only use samples with exactly one positive label
            if not sample.get("is_valid_single_element", False) or element_idx < 0:
                continue
        else:
            # In non-strict mode, use argmax(label) but still skip if invalid
            if element_idx < 0:
                continue

        if element_idx not in templates_by_element:
            templates_by_element[element_idx] = []

        if max_templates_per_element is None or len(templates_by_element[element_idx]) < max_templates_per_element:
            templates_by_element[element_idx].append(sample)

    # Stack templates for each element
    result: Dict[int, Dict[str, torch.Tensor]] = {}

    for element_idx, samples in templates_by_element.items():
        if not samples:
            continue

        element_name = samples[0]["element_name"]

        stacked = {
            "spectra": torch.stack([s["x"] for s in samples], dim=0),  # (N, 1, 3072)
            "labels": torch.stack([s["y"] for s in samples], dim=0),  # (N, 80)
            "element_name": element_name,
        }

        # Add optional fields if present
        if "nonzero_mask" in samples[0]:
            stacked["mask"] = torch.stack([s["nonzero_mask"] for s in samples], dim=0)

        if "nonzero_bounds" in samples[0]:
            stacked["bounds"] = torch.stack([s["nonzero_bounds"] for s in samples], dim=0)

        result[element_idx] = stacked

    return result


def build_template_bank(
    root: str | Path,
    preprocess_cfg: Optional[SpectrumPreprocessConfig] = None,
    max_templates_per_element: Optional[int] = None,
    strict_single_label: bool = True,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Build template bank from single-element spectra.
    
    Args:
        root: Root path to EELS dataset
        preprocess_cfg: Preprocessing configuration
        max_templates_per_element: Maximum number of templates per element (None = all)
        strict_single_label: If True, only use samples with label.sum() == 1 (default: True)
    
    Returns:
        Dictionary mapping element_idx to stacked templates:
        {
            element_idx: {
                "spectra": Tensor[num_templates, 1, 3072],
                "labels": Tensor[num_templates, 80],
                "mask": Tensor[num_templates, 1, 3072] (if enabled),
                "bounds": Tensor[num_templates, 2] (if enabled),
                "element_name": str,
            }
        }
    """
    dataset = make_single_element_dataset(root, preprocess_cfg=preprocess_cfg)
    template_bank = stack_templates_by_element(
        dataset,
        max_templates_per_element=max_templates_per_element,
        strict_single_label=strict_single_label,
    )
    dataset.close()
    return template_bank
