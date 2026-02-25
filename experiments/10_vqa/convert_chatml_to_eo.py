#!/usr/bin/env python3
"""
Convert multiview ChatML data from YTSaurus to EO LLaVA JSONL format.

Reads the ChatML table from YTSaurus (//home/visiondata/users/gregorykogan/
eo-data/multiview_chatml), converts dialogue format to LLaVA-style
conversations, and optionally downloads images from S3 for local debug.

ChatML format (input):
  {
    "uuid": "...",
    "dialogue": [
      {"role": "system", "content": "", "trainable": false},
      {"role": "user", "content": "...", "files": [...], "trainable": false},
      {"role": "assistant", "content": "...", "trainable": true}
    ],
    "image_paths": ["visiondata/robotics/.../img1.png", ...],
    "s3_bucket": "gigaeye-data"
  }

EO LLaVA format (output):
  {
    "conversations": [
      {"from": "human", "value": "<image><image>..."},
      {"from": "gpt", "value": "..."}
    ],
    "image": ["path/to/img1.png", "path/to/img2.png"]
  }

Usage:
    # Convert all rows, S3 image paths (for cloud training)
    python experiments/10_vqa/convert_chatml_to_eo.py

    # Convert first 200 rows and download images locally (for debug)
    python experiments/10_vqa/convert_chatml_to_eo.py --download-images --limit 200

    # Dry run: just print converted samples
    python experiments/10_vqa/convert_chatml_to_eo.py --limit 5 --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Ensure EO1 package is importable when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eo.data.chatml import chatml_row_to_llava

# -----------------------------------------------
# Configuration
# -----------------------------------------------
YT_TABLE = "//home/visiondata/users/gregorykogan/eo-data/multiview_chatml"
DEFAULT_S3_BUCKET = "gigaeye-data"

DATA_DIR = Path("../../data_multiview_chatml")
IMAGES_DIR = DATA_DIR / "images"
JSONL_DIR = DATA_DIR / "jsonl"


def convert_chatml_row(row: dict) -> dict | None:
    """Convert a single ChatML row to EO LLaVA format."""
    return chatml_row_to_llava(row)


def read_yt_table(limit: int | None = None) -> list[dict]:
    """Read rows from YTSaurus table using yt CLI."""
    yt_proxy = os.environ.get("YT_PROXY")
    if not yt_proxy:
        print("ERROR: YT_PROXY environment variable not set")
        sys.exit(1)

    table_path = YT_TABLE
    if limit is not None:
        table_path = f"{YT_TABLE}[:#{limit}]"

    cmd = [
        "yt", "read",
        "--proxy", yt_proxy,
        "--format", "json",
        table_path,
    ]

    print(f"[READ] Reading from {table_path} ...")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        # Try with conda
        print("  yt not found directly, trying conda run...")
        cmd = ["conda", "run", "-n", "avzavarzin"] + cmd
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

    rows = []
    for line in result.stdout.strip().split("\n"):
        if line:
            rows.append(json.loads(line))

    print(f"  Read {len(rows)} rows")
    return rows


def download_image_from_s3(s3_client, bucket: str, key: str, local_path: Path) -> bool:
    """Download a single image from S3 to local disk."""
    if local_path.exists():
        return True
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        data = s3_client.download(bucket, key)
        with open(local_path, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"  WARNING: Failed to download {key}: {e}")
        return False


def download_images(jsonl_path: Path, images_dir: Path, bucket: str):
    """Download all images referenced in a JSONL file from S3."""
    # Lazy import to avoid requiring boto3 when not downloading
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from eo.data.s3_loader import S3Client

    # Load S3 secrets
    secrets_path = Path(__file__).resolve().parents[3] / "secrets.env"
    secrets = {}
    if secrets_path.exists():
        with open(secrets_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    secrets[k.strip()] = v.strip().strip('"').strip("'")

    s3_client = S3Client.create(secrets if secrets else None)

    # Collect all image paths
    all_image_keys = set()
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            images = entry.get("image", [])
            if isinstance(images, str):
                images = [images]
            all_image_keys.update(images)

    print(f"\n[DOWNLOAD] Downloading {len(all_image_keys)} images from s3://{bucket}/ ...")
    images_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0

    def _download_one(key):
        local_path = images_dir / key
        return key, download_image_from_s3(s3_client, bucket, key, local_path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_download_one, key): key for key in all_image_keys}
        for future in as_completed(futures):
            key, success = future.result()
            if success:
                local_path = images_dir / key
                if local_path.stat().st_size > 0:
                    downloaded += 1
                else:
                    skipped += 1
            else:
                failed += 1

            total = downloaded + skipped + failed
            if total % 500 == 0:
                print(f"  Progress: {total}/{len(all_image_keys)} "
                      f"(downloaded={downloaded}, cached={skipped}, failed={failed})")

    print(f"  [OK] Downloaded: {downloaded}, Already cached: {skipped}, Failed: {failed}")

    # Rewrite JSONL with relative paths (relative to images_dir)
    local_jsonl_path = jsonl_path.with_suffix(".local.jsonl")
    count = 0
    with open(jsonl_path) as f_in, open(local_jsonl_path, "w") as f_out:
        for line in f_in:
            entry = json.loads(line.strip())
            # image paths stay the same (they are relative S3 keys,
            # and images_dir mirrors the S3 key structure)
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    print(f"  [OK] Local JSONL: {local_jsonl_path} ({count} samples)")
    return local_jsonl_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert multiview ChatML data from YTSaurus to EO LLaVA format"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of rows to read from YT table (default: all)"
    )
    parser.add_argument(
        "--download-images", action="store_true",
        help="Download images from S3 to local disk (for local debug)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print converted samples to stdout instead of writing to file"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory (default: ../../data_multiview_chatml)"
    )
    args = parser.parse_args()

    if args.output_dir:
        data_dir = Path(args.output_dir)
    else:
        data_dir = DATA_DIR
    jsonl_dir = data_dir / "jsonl"
    images_dir = data_dir / "images"

    # Step 1: Read YT table
    rows = read_yt_table(args.limit)

    # Step 2: Convert each row
    converted = []
    skipped = 0
    for row in rows:
        entry = convert_chatml_row(row)
        if entry is None:
            skipped += 1
            continue
        converted.append(entry)

    print(f"\n[CONVERT] Converted {len(converted)} samples ({skipped} skipped)")

    if args.dry_run:
        for entry in converted[:10]:
            print(json.dumps(entry, indent=2, ensure_ascii=False))
            print("---")
        return

    # Step 3: Write JSONL
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = jsonl_dir / "multiview_chatml.jsonl"
    with open(jsonl_path, "w") as f:
        for entry in converted:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  [OK] Wrote {len(converted)} samples to {jsonl_path}")

    # Step 4: Optionally download images
    if args.download_images:
        # Determine bucket from first row
        bucket = DEFAULT_S3_BUCKET
        if rows and rows[0].get("s3_bucket"):
            bucket = rows[0]["s3_bucket"]
        download_images(jsonl_path, images_dir, bucket)

    # Summary
    print("\n" + "=" * 60)
    print("  Conversion complete!")
    print("=" * 60)
    print(f"  Total samples  : {len(converted)}")
    print(f"  JSONL output   : {jsonl_path}")
    if args.download_images:
        print(f"  Images dir     : {images_dir}")
        local_jsonl = jsonl_path.with_suffix(".local.jsonl")
        if local_jsonl.exists():
            print(f"  Local JSONL    : {local_jsonl}")
    print()
    print("  For S3-based training, use: data-multiview-s3.yaml")
    print("  For local debug, use:       data-multiview-local.yaml")
    print("=" * 60)


if __name__ == "__main__":
    main()
