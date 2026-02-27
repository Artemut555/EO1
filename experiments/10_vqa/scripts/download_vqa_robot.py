#!/usr/bin/env python3
"""
Download and convert EO-Data1.5M QA subsets for VQA fine-tuning.

Uses STREAMING mode to avoid downloading full parquet files to cache.
Only the images we actually need are saved to disk (~3 GB vs ~70 GB).

Targets ~155K total samples — balanced across 9 robotics reasoning skills
that align with EO-Bench, ERQA, and RoboVQA evaluation benchmarks.

Usage:
    cd EO1
    python experiments/10_vqa/download_data.py
"""

import json
import sys
import io
from pathlib import Path

# -----------------------------------------------
# Configuration: which QA subsets and how many
# -----------------------------------------------
SUBSET_CONFIG = {
    "qa-affordance_qa":         20_000,  # Can the robot perform action X?
    "qa-episode_caption":       20_000,  # Describe the robot episode
    "qa-failure_detection":     25_000,  # Did the action succeed or fail?
    "qa-object_referring_qa":   20_000,  # Which object is being referred to?
    "qa-physical_common_sense": 20_000,  # Physics / common sense reasoning
    "qa-process_verification":  15_000,  # Has step X been completed?
    "qa-relation_reasoning":    15_000,  # Spatial relations between objects
    "qa-subtask_qa":            10_000,  # Subtask decomposition
    "qa-task_planning":         10_000,  # What to do next?
}

DATASET_ID = "IPEC-COMMUNITY/EO-Data1.5M"
DATA_DIR = Path("../data_vqa_robot")
IMAGES_DIR = DATA_DIR / "images"
JSONL_DIR = DATA_DIR / "jsonl"


def save_image(img, path):
    """Save a PIL image or image dict to disk as JPEG."""
    from PIL import Image

    if isinstance(img, Image.Image):
        # Convert RGBA/P to RGB for JPEG
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        img.save(path, "JPEG", quality=90)
        return True
    elif isinstance(img, dict) and "bytes" in img:
        # HF streaming sometimes returns {"bytes": b"...", "path": ...}
        pil_img = Image.open(io.BytesIO(img["bytes"]))
        if pil_img.mode in ("RGBA", "P", "LA"):
            pil_img = pil_img.convert("RGB")
        pil_img.save(path, "JPEG", quality=90)
        return True
    elif isinstance(img, bytes):
        pil_img = Image.open(io.BytesIO(img))
        if pil_img.mode in ("RGBA", "P", "LA"):
            pil_img = pil_img.convert("RGB")
        pil_img.save(path, "JPEG", quality=90)
        return True
    return False


def download_and_convert():
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not found. Install with: pip install datasets")
        sys.exit(1)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    JSONL_DIR.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    all_jsonl_paths = []

    for subset_name, max_samples in SUBSET_CONFIG.items():
        jsonl_path = JSONL_DIR / f"{subset_name}.jsonl"
        subset_image_dir = IMAGES_DIR / subset_name
        subset_image_dir.mkdir(parents=True, exist_ok=True)

        # Skip if already done
        if jsonl_path.exists():
            existing_count = sum(1 for _ in open(jsonl_path))
            if existing_count >= max_samples:
                print(f"[SKIP] {subset_name}: {existing_count} samples already exist")
                total_samples += existing_count
                all_jsonl_paths.append(jsonl_path)
                continue

        print(f"\n[STREAM] {subset_name} (target: {max_samples} samples)...")

        # ---- Download only the slice we need — no full dataset download ----
        # Using split="train[:N]" tells HF to only fetch the parquet shards
        # needed to cover the first N rows, instead of all 140 shards.
        try:
            print(f"  Loading first {max_samples} rows...")
            ds_stream = load_dataset(
                DATASET_ID,
                subset_name,
                split=f"train[:{max_samples}]",
            )
        except Exception as e:
            print(f"  WARNING: Failed to load {subset_name}: {e}")
            continue

        print(f"  Loaded {len(ds_stream)} samples")

        count = 0
        with open(jsonl_path, "w") as out_f:
            for idx, sample in enumerate(ds_stream):
                try:
                    conversations = sample.get("conversation")
                    if not conversations:
                        continue

                    images_data = sample.get("image", [])

                    # Normalize to list — some samples have a single image, not a list
                    if images_data is not None and not isinstance(images_data, (list, tuple)):
                        images_data = [images_data]
                    elif images_data is None:
                        images_data = []

                    # Skip samples with actions (we want pure VQA only)
                    actions = sample.get("action")
                    if actions and len(actions) > 0:
                        continue

                    # Save images to disk
                    image_paths = []
                    if images_data:
                        for img_idx, img in enumerate(images_data):
                            if img is None:
                                continue
                            img_filename = f"{count:06d}_{img_idx}.jpg"
                            img_path = subset_image_dir / img_filename
                            if not img_path.exists():
                                if not save_image(img, img_path):
                                    continue
                            image_paths.append(f"{subset_name}/{img_filename}")

                    # Skip samples with no images
                    if not image_paths:
                        continue

                    # Build JSONL entry in LLaVA format
                    entry = {"conversations": conversations}
                    if len(image_paths) == 1:
                        entry["image"] = image_paths[0]
                    else:
                        entry["image"] = image_paths

                    out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    count += 1

                except Exception as e:
                    print(f"  WARNING: Skipping sample {idx} in {subset_name}: {e}")
                    continue

                if count % 5000 == 0:
                    print(f"  Streamed {count}/{max_samples}...")

        print(f"  [OK] Wrote {count} samples to {jsonl_path}")
        total_samples += count
        all_jsonl_paths.append(jsonl_path)

        # Free memory and clean HF cache for this subset to avoid disk bloat
        del ds_stream
        try:
            import shutil, os
            cache_dir = os.path.expanduser("~/.cache/huggingface/datasets/IPEC-COMMUNITY___eo-data1.5_m")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                print(f"  [CLEAN] Removed HF cache")
        except Exception:
            pass

    # -----------------------------------------------
    # Create a merged JSONL for convenience
    # -----------------------------------------------
    merged_path = JSONL_DIR / "all_robot_vqa.jsonl"
    print(f"\n[MERGE] Creating merged JSONL: {merged_path}")
    total_merged = 0
    with open(merged_path, "w") as out_f:
        for jsonl_path in all_jsonl_paths:
            with open(jsonl_path) as in_f:
                for line in in_f:
                    out_f.write(line)
                    total_merged += 1

    # -----------------------------------------------
    # Summary
    # -----------------------------------------------
    print("\n" + "=" * 50)
    print("  Download complete!")
    print("=" * 50)
    print(f"  Total samples  : {total_merged}")
    print(f"  JSONL files    : {JSONL_DIR}/")
    print(f"  Images         : {IMAGES_DIR}/")
    print(f"  Merged JSONL   : {merged_path}")
    print()
    print("  Subset breakdown:")
    for jsonl_path in sorted(all_jsonl_paths):
        if jsonl_path.name != "all_robot_vqa.jsonl":
            n = sum(1 for _ in open(jsonl_path))
            print(f"    {jsonl_path.stem:<30} {n:>7} samples")
    print()
    print("  To train, run:")
    print("    bash experiments/10_vqa/train.sh")
    print("=" * 50)


if __name__ == "__main__":
    download_and_convert()
