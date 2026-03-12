"""
Diagnostic script to detect shortcuts in EELS dataset.

Checks:
1. Whether query_element_idx is always present in y
2. Distribution of nonzero_bounds / window lengths per element
3. Whether element can be classified by window length alone (shortcut detection)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from dataset import (
    make_split_dataset,
    SpectrumPreprocessConfig,
    ELEMENT_TO_IDX,
    ELEMENTS,
    NUM_CLASSES,
)


def check_query_element_in_labels(dataset, n_samples: int = 5000) -> dict:
    """
    Check if query_element_idx is always (or almost always) present in y.
    
    Returns:
        Dictionary with statistics about query element presence
    """
    print("\n" + "="*60)
    print("Checking query_element_idx presence in labels")
    print("="*60)
    
    # Sample random indices
    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    
    query_present = 0
    query_absent = 0
    element_stats = defaultdict(lambda: {"present": 0, "absent": 0})
    
    for idx in indices:
        sample = dataset[idx]
        meta = sample.get("meta")
        if meta is None:
            continue
        
        query_idx = meta.get("query_element_idx", -1)
        if query_idx < 0:
            continue
        
        y = sample["y"]
        is_present = y[query_idx].item() > 0.5
        
        element_name = meta.get("query_element", "unknown")
        element_stats[element_name]["present" if is_present else "absent"] += 1
        
        if is_present:
            query_present += 1
        else:
            query_absent += 1
    
    total = query_present + query_absent
    presence_rate = query_present / total if total > 0 else 0.0
    
    print(f"Total samples checked: {total}")
    print(f"Query element present: {query_present} ({presence_rate*100:.2f}%)")
    print(f"Query element absent: {query_absent} ({(1-presence_rate)*100:.2f}%)")
    
    # Per-element breakdown
    print("\nPer-element breakdown (top 10):")
    sorted_elements = sorted(
        element_stats.items(),
        key=lambda x: x[1]["present"] + x[1]["absent"],
        reverse=True
    )[:10]
    
    for elem_name, stats in sorted_elements:
        total_elem = stats["present"] + stats["absent"]
        if total_elem > 0:
            rate = stats["present"] / total_elem
            print(f"  {elem_name:3s}: {stats['present']:4d}/{total_elem:4d} ({rate*100:5.2f}%)")
    
    return {
        "total_samples": total,
        "presence_rate": presence_rate,
        "query_present": query_present,
        "query_absent": query_absent,
        "element_stats": dict(element_stats),
    }


def analyze_nonzero_bounds_distribution(dataset, n_samples: int = 5000) -> dict:
    """
    Analyze distribution of nonzero_bounds / window lengths per element.
    
    Returns:
        Dictionary with statistics about window lengths
    """
    print("\n" + "="*60)
    print("Analyzing nonzero_bounds / window length distribution")
    print("="*60)
    
    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    
    element_window_lengths = defaultdict(list)
    all_window_lengths = []
    
    for idx in indices:
        sample = dataset[idx]
        
        if "nonzero_bounds" not in sample:
            continue
        
        meta = sample.get("meta")
        if meta is None:
            continue
        
        bounds = sample["nonzero_bounds"]
        left, right = bounds[0].item(), bounds[1].item()
        window_length = right - left
        
        element_name = meta.get("query_element", "unknown")
        element_window_lengths[element_name].append(window_length)
        all_window_lengths.append(window_length)
    
    # Overall statistics
    all_lengths = np.array(all_window_lengths)
    print(f"Total samples with bounds: {len(all_lengths)}")
    print(f"Window length statistics:")
    print(f"  Mean: {all_lengths.mean():.1f}")
    print(f"  Std: {all_lengths.std():.1f}")
    print(f"  Min: {all_lengths.min()}")
    print(f"  Max: {all_lengths.max()}")
    print(f"  Median: {np.median(all_lengths):.1f}")
    
    # Per-element statistics
    print("\nPer-element window length statistics (top 10 by count):")
    sorted_elements = sorted(
        element_window_lengths.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:10]
    
    for elem_name, lengths in sorted_elements:
        lengths_arr = np.array(lengths)
        print(f"  {elem_name:3s}: mean={lengths_arr.mean():6.1f}, "
              f"std={lengths_arr.std():5.1f}, "
              f"min={lengths_arr.min():4d}, "
              f"max={lengths_arr.max():4d}, "
              f"n={len(lengths):4d}")
    
    return {
        "all_window_lengths": all_lengths,
        "element_window_lengths": {k: np.array(v) for k, v in element_window_lengths.items()},
    }


def test_window_length_classification(dataset, n_samples: int = 5000) -> dict:
    """
    Test if element can be classified by window length alone (shortcut detection).
    
    Returns:
        Dictionary with classification results
    """
    print("\n" + "="*60)
    print("Testing window length as shortcut for element classification")
    print("="*60)
    
    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    
    # Collect data: (window_length, element_idx)
    X = []
    y = []
    
    for idx in indices:
        sample = dataset[idx]
        
        if "nonzero_bounds" not in sample:
            continue
        
        meta = sample.get("meta")
        if meta is None:
            continue
        
        bounds = sample["nonzero_bounds"]
        left, right = bounds[0].item(), bounds[1].item()
        window_length = right - left
        
        query_idx = meta.get("query_element_idx", -1)
        if query_idx < 0:
            continue
        
        # Check if query element is actually in labels
        y_labels = sample["y"]
        if y_labels[query_idx].item() < 0.5:
            continue  # Skip if query element not in labels
        
        X.append([window_length])
        y.append(query_idx)
    
    if len(X) < 100:
        print(f"Not enough samples ({len(X)}), need at least 100")
        return {"accuracy": 0.0, "n_samples": len(X), "is_shortcut": False}
    
    X = np.array(X)
    y = np.array(y)
    
    # Split train/test
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    # Train simple classifier
    clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Baseline: random guess
    n_classes = len(np.unique(y))
    baseline = 1.0 / n_classes
    
    print(f"Samples used: {len(X)}")
    print(f"Train samples: {n_train}, Test samples: {len(X_test)}")
    print(f"Unique elements: {n_classes}")
    print(f"Classification accuracy: {accuracy:.4f}")
    print(f"Baseline (random): {baseline:.4f}")
    print(f"Improvement over baseline: {accuracy - baseline:.4f}")
    
    is_shortcut = accuracy > baseline + 0.1  # Significant improvement
    
    if is_shortcut:
        print("⚠️  WARNING: Window length appears to be a shortcut!")
        print("   Model might learn to classify by window length alone.")
    else:
        print("✓ Window length does not appear to be a strong shortcut.")
    
    return {
        "accuracy": accuracy,
        "baseline": baseline,
        "n_samples": len(X),
        "n_train": n_train,
        "n_test": len(X_test),
        "n_classes": n_classes,
        "is_shortcut": is_shortcut,
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose shortcuts in EELS dataset")
    parser.add_argument("--root", type=str, default="EELS", help="Path to EELS dataset root")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"],
                        help="Dataset split to analyze")
    parser.add_argument("--n_samples", type=int, default=5000,
                        help="Number of random samples to analyze")
    args = parser.parse_args()
    
    print("="*60)
    print("EELS Dataset Shortcut Diagnosis")
    print("="*60)
    print(f"Dataset root: {args.root}")
    print(f"Split: {args.split}")
    print(f"Sample size: {args.n_samples}")
    
    # Create dataset with metadata and nonzero_bounds
    preprocess_cfg = SpectrumPreprocessConfig(
        add_channel_dim=True,
        return_nonzero_mask=True,
        return_nonzero_bounds=True,
    )
    
    dataset = make_split_dataset(
        root=args.root,
        split=args.split,
        task="identification",
        preprocess_cfg=preprocess_cfg,
        return_metadata=True,
        return_index=False,
    )
    
    print(f"\nDataset size: {len(dataset):,}")
    
    # Run diagnostics
    query_stats = check_query_element_in_labels(dataset, n_samples=args.n_samples)
    bounds_stats = analyze_nonzero_bounds_distribution(dataset, n_samples=args.n_samples)
    shortcut_test = test_window_length_classification(dataset, n_samples=args.n_samples)
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Query element presence rate: {query_stats['presence_rate']*100:.2f}%")
    print(f"Window length classification accuracy: {shortcut_test['accuracy']:.4f}")
    print(f"Is window length a shortcut? {shortcut_test['is_shortcut']}")
    
    if shortcut_test['is_shortcut']:
        print("\n⚠️  RECOMMENDATION: Consider using windowed mask to prevent shortcut learning.")


if __name__ == "__main__":
    main()
