"""
Example usage of dataset, stats, and templates modules.
"""

from pathlib import Path

from dataset import (
    SpectrumPreprocessConfig,
    make_split_dataset,
    make_dataloader,
    print_dataset_summary,
    inspect_dataset_samples,
    validate_sample,
)
from stats import (
    compute_class_stats_identification,
    compute_cooccurrence_matrix,
    compute_label_cardinality_stats,
    save_stats_npz,
    load_stats_npz,
)
from templates import build_template_bank


def main():
    # Path to EELS dataset
    root = Path("EELS")

    # 1. Create train dataset
    print("=" * 60)
    print("1. Creating train dataset")
    print("=" * 60)

    preprocess_cfg = SpectrumPreprocessConfig(
        add_channel_dim=True,
        return_nonzero_mask=True,
        return_nonzero_bounds=False,
    )

    train_ds = make_split_dataset(
        root=root,
        split="train",
        task="identification",
        preprocess_cfg=preprocess_cfg,
        return_metadata=False,  # Default is False for training
        return_index=False,
    )

    print(f"Train dataset size: {len(train_ds):,}")
    print_dataset_summary(train_ds.file_records)

    # 2. Create dataloader
    print("\n" + "=" * 60)
    print("2. Creating dataloader")
    print("=" * 60)

    train_loader = make_dataloader(
        dataset=train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Get a batch
    batch = next(iter(train_loader))
    print(f"Batch x shape: {batch['x'].shape}")  # [32, 1, 3072]
    print(f"Batch y shape: {batch['y'].shape}")  # [32, 80]

    # Validate a sample
    sample = train_ds[0]
    validate_sample(sample, task="identification")
    print("Sample validation passed!")

    # Inspect random samples
    inspect_dataset_samples(train_ds, n=5, seed=42)

    # 3. Compute statistics
    print("\n" + "=" * 60)
    print("3. Computing class statistics")
    print("=" * 60)

    from dataset import get_split_file_paths

    train_files = get_split_file_paths(root, split="train")
    stats = compute_class_stats_identification(train_files)

    print(f"Total samples: {stats['total_samples'][0]:,}")
    print(f"Positive counts (first 10): {stats['pos_counts'][:10]}")
    print(f"Prevalence (first 10): {stats['prevalence'][:10]}")
    print(f"Pos weight (first 10): {stats['pos_weight'][:10]}")

    # Compute co-occurrence matrix
    print("\nComputing co-occurrence matrix...")
    cooccurrence = compute_cooccurrence_matrix(train_files)
    print(f"Co-occurrence matrix shape: {cooccurrence.shape}")
    print(f"Diagonal (self-cooccurrence): {cooccurrence.diagonal()[:10]}")

    # Compute label cardinality stats
    print("\nComputing label cardinality statistics...")
    cardinality_stats = compute_label_cardinality_stats(train_files)
    print(f"Mean cardinality: {cardinality_stats['mean_cardinality']:.2f}")
    print(f"Min/Max cardinality: {cardinality_stats['min_cardinality']} / {cardinality_stats['max_cardinality']}")
    print(f"Cardinality distribution: {cardinality_stats['cardinality_distribution']}")

    # Save statistics
    print("\nSaving statistics to stats.npz...")
    save_stats_npz(stats, Path("stats.npz"))
    print("Statistics saved!")

    # 4. Build template bank
    print("\n" + "=" * 60)
    print("4. Building template bank")
    print("=" * 60)

    template_cfg = SpectrumPreprocessConfig(
        add_channel_dim=True,
        return_nonzero_mask=True,
        return_nonzero_bounds=True,
    )

    template_bank = build_template_bank(
        root=root,
        preprocess_cfg=template_cfg,
        max_templates_per_element=10,  # Limit for quick prototyping
    )

    print(f"Number of elements in template bank: {len(template_bank)}")
    for element_idx in sorted(template_bank.keys())[:5]:  # Show first 5
        template = template_bank[element_idx]
        print(f"  Element {element_idx} ({template['element_name']}): "
              f"{template['spectra'].shape[0]} templates, "
              f"spectra shape: {template['spectra'].shape}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
