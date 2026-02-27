#!/usr/bin/env python3
"""
Upload a sample of VQA robot data to S3 for profiling.
Data is read from EXTRA_EO/data/vqa_robot (default ~/projects/extra/eo).
1. List s3://gigaeye-data/visiondata/users/avzavarzin to verify folder exists
2. Sample N rows from all_robot_vqa.jsonl (default 1000)
3. Upload JSONL subset and corresponding images to S3
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root for imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eo_experiment.data_loading.s3_client import S3Client, load_secrets

BUCKET = "gigaeye-data"
S3_PREFIX = "visiondata/users/avzavarzin"
S3_DATA_PREFIX = f"{S3_PREFIX}/data_vqa_robot"

EXTRA_EO = os.environ.get("EXTRA_EO") or str(Path.home() / "projects" / "extra" / "eo")


def main(num_samples: int = 1000):
    data_dir = Path(EXTRA_EO) / "data" / "vqa_robot"
    jsonl_path = data_dir / "jsonl" / "all_robot_vqa.jsonl"
    images_dir = data_dir / "images"

    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found. Run download_data.py first.")
        sys.exit(1)

    secrets = load_secrets(ROOT / "secrets.env")
    client = S3Client.create(secrets)

    # 1. List S3 folder to verify it exists
    print(f"Listing s3://{BUCKET}/{S3_PREFIX}/...")
    keys = client.list_objects(BUCKET, prefix=S3_PREFIX + "/", max_keys=20)
    print(f"  Found {len(keys)} objects (showing up to 20)")
    for k in keys[:5]:
        print(f"    {k}")
    if keys:
        print("  OK: folder exists and has content")
    else:
        print("  Note: prefix is empty or new - continuing to upload")

    # 2. Load and sample JSONL
    print(f"\nLoading {jsonl_path}...")
    samples = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            samples.append(json.loads(line.strip()))

    print(f"  Sampled {len(samples)} samples")

    # 3. Collect unique image paths
    image_paths = set()
    for s in samples:
        img = s.get("image")
        if isinstance(img, list):
            image_paths.update(img)
        elif img:
            image_paths.add(img)

    print(f"  {len(image_paths)} unique images to upload")

    # 4. Upload images
    uploaded = 0
    for rel_path in image_paths:
        local_path = images_dir / rel_path
        if not local_path.exists():
            print(f"  WARNING: {local_path} not found, skipping")
            continue
        s3_key = f"{S3_DATA_PREFIX}/images/{rel_path}"
        data = local_path.read_bytes()
        client.upload(data, BUCKET, s3_key, content_type="image/jpeg")
        uploaded += 1
        if uploaded % 100 == 0:
            print(f"  Uploaded {uploaded}/{len(image_paths)} images...")

    print(f"  Uploaded {uploaded} images")

    # 5. Create sample JSONL and save locally + upload to S3
    sample_jsonl_path = data_dir / "jsonl" / "all_robot_vqa_sample.jsonl"
    with open(sample_jsonl_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n  Saved local sample: {sample_jsonl_path}")

    sample_jsonl_bytes = sample_jsonl_path.read_bytes()
    s3_jsonl_key = f"{S3_DATA_PREFIX}/jsonl/all_robot_vqa_sample.jsonl"
    client.upload(sample_jsonl_bytes, BUCKET, s3_jsonl_key, content_type="application/jsonl")
    print(f"  Uploaded JSONL to s3://{BUCKET}/{s3_jsonl_key}")

    print("\nDone! Use data-vqa-s3.yaml with vision_base_path for S3 images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload VQA robot data sample from EXTRA_EO to S3")
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples to upload (default: 1000)",
    )
    args = parser.parse_args()
    main(num_samples=args.samples)
