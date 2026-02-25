#!/usr/bin/env python3
"""
Local test for multiview dataset loading (LLaVA and ChatML formats).

Loads a small dataset from config, runs a few __getitem__ calls, and prints
sample keys and content. Use after download_qa_multiview_local.py to test
ChatML format with local files.

Usage:
    # Test with default config (ChatML local small sample)
    python experiments/10_vqa/test_multiview_dataset.py

    # Test with specific config
    python experiments/10_vqa/test_multiview_dataset.py --config experiments/10_vqa/data-qa-multiview-chatml-local.yaml

    # Test LLaVA format (converted multiview)
    python experiments/10_vqa/test_multiview_dataset.py --config experiments/10_vqa/data-multiview-local.yaml
"""

import argparse
import sys
from pathlib import Path

# Run from EO1 repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eo.data.schema import DataConfig
from eo.data.multim_dataset import MultimodaDataset


def main():
    parser = argparse.ArgumentParser(description="Test multiview dataset loading")
    parser.add_argument(
        "--config", type=str,
        default=str(REPO_ROOT / "experiments" / "10_vqa" / "data-qa-multiview-chatml-local.yaml"),
        help="Path to data config YAML",
    )
    parser.add_argument(
        "--num-samples", type=int, default=3,
        help="Number of samples to fetch (default: 3)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        print("Run download_qa_multiview_local.py first to create the small dataset.")
        sys.exit(1)

    print(f"Loading config: {config_path}")
    data_configs = DataConfig.from_yaml(str(config_path))
    if not data_configs.mm_datasets:
        print("ERROR: No mm_datasets in config")
        sys.exit(1)

    print(f"Building MultimodaDataset (format={data_configs.mm_datasets[0].format})...")
    try:
        dataset = MultimodaDataset(
            data_configs.mm_datasets,
            max_seq_length=8192,
        )
    except FileNotFoundError as e:
        print(f"ERROR: Missing data file: {e}")
        print("Run first: python experiments/10_vqa/download_qa_multiview_local.py")
        sys.exit(1)
    n = len(dataset)
    print(f"  Dataset length: {n}")

    if n == 0:
        print("ERROR: Empty dataset. Check json_path and format.")
        sys.exit(1)

    num_samples = min(args.num_samples, n)
    for i in range(num_samples):
        sample = dataset[i]
        print(f"\n--- Sample {i} ---")
        print(f"  Keys: {list(sample.keys())}")
        conv = sample.get("conversations", [])
        print(f"  Conversations: {len(conv)} turns")
        for j, turn in enumerate(conv[:4]):
            role = turn.get("role", "?")
            content = (turn.get("content", "") or "")[:80]
            print(f"    [{j}] {role}: {content!r}...")
        images = sample.get("image", [])
        if not isinstance(images, list):
            images = [images]
        print(f"  Images: {len(images)} paths")
        for p in images[:2]:
            print(f"    {p!r}")
        if "vision_backend" in sample:
            print(f"  vision_backend: {sample['vision_backend']}")

    print("\n[OK] test_multiview_dataset passed.")


if __name__ == "__main__":
    main()
