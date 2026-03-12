from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

from dataset import HDF5_LABEL_ID_KEY, NUM_CLASSES


def compute_class_stats_identification(file_paths: List[Path]) -> Dict[str, np.ndarray]:
    """Compute class statistics for identification task."""
    pos_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    total_samples = 0

    for fp in file_paths:
        with h5py.File(fp, "r") as f:
            y = f[HDF5_LABEL_ID_KEY][:]
            pos_counts += y.sum(axis=0)
            total_samples += y.shape[0]

    neg_counts = total_samples - pos_counts
    pos_weight = np.divide(
        neg_counts,
        np.maximum(pos_counts, 1.0),
        out=np.zeros_like(neg_counts),
        where=np.maximum(pos_counts, 1.0) > 0
    )

    prevalence = pos_counts / max(total_samples, 1)

    return {
        "pos_counts": pos_counts,
        "neg_counts": neg_counts,
        "prevalence": prevalence,
        "pos_weight": pos_weight,
        "total_samples": np.array([total_samples], dtype=np.int64),
    }


def compute_cooccurrence_matrix(file_paths: List[Path]) -> np.ndarray:
    """
    Compute co-occurrence matrix C = sum_n (y_n^T @ y_n) for multi-label data.
    
    Returns:
        Matrix of shape (NUM_CLASSES, NUM_CLASSES) where C[i, j] is the number
        of samples where both class i and class j are present.
    """
    cooccurrence = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)

    for fp in file_paths:
        with h5py.File(fp, "r") as f:
            y = f[HDF5_LABEL_ID_KEY][:]  # (N, 80)
            # For each sample, compute outer product and add to cooccurrence
            for i in range(y.shape[0]):
                y_vec = y[i, :]  # (80,)
                cooccurrence += np.outer(y_vec, y_vec)

    return cooccurrence


def compute_label_cardinality_stats(file_paths: List[Path]) -> Dict:
    """
    Compute statistics about label cardinality (number of positive labels per sample).
    
    Returns:
        Dictionary with:
        - mean_cardinality: average number of positive labels per sample
        - cardinality_distribution: Counter of cardinality values
        - min_cardinality: minimum number of positive labels
        - max_cardinality: maximum number of positive labels
    """
    cardinalities = []

    for fp in file_paths:
        with h5py.File(fp, "r") as f:
            y = f[HDF5_LABEL_ID_KEY][:]  # (N, 80)
            # Sum along class dimension to get number of positive labels per sample
            sample_sums = y.sum(axis=1)  # (N,)
            cardinalities.extend(sample_sums.tolist())

    cardinalities = np.array(cardinalities)
    cardinality_dist = Counter(cardinalities.astype(int))

    return {
        "mean_cardinality": float(cardinalities.mean()),
        "std_cardinality": float(cardinalities.std()),
        "min_cardinality": int(cardinalities.min()),
        "max_cardinality": int(cardinalities.max()),
        "cardinality_distribution": dict(cardinality_dist),
        "total_samples": len(cardinalities),
    }


def save_stats_json(stats: Dict, output_path: Path) -> None:
    """Save statistics to JSON file (converting numpy arrays to lists)."""
    # Convert numpy arrays to lists for JSON serialization
    json_stats = {}
    for key, value in stats.items():
        if isinstance(value, np.ndarray):
            json_stats[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            json_stats[key] = float(value)
        else:
            json_stats[key] = value

    with open(output_path, "w") as f:
        json.dump(json_stats, f, indent=2)


def save_stats_npz(stats: Dict, output_path: Path) -> None:
    """Save statistics to NPZ file (preserving numpy arrays)."""
    np.savez(output_path, **stats)


def load_stats_npz(input_path: Path) -> Dict[str, np.ndarray]:
    """Load statistics from NPZ file."""
    return dict(np.load(input_path))
