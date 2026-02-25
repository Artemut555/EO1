#!/usr/bin/env python3
"""
Download a small sample of qa-multiview_qa from YTSaurus for local testing.

Source: //home/visiondata/users/gregorykogan/eo-data/chatml/qa-multiview_qa
Writes ChatML JSONL to data/qa-multiview_qa/jsonl/ so the dataset loader
(format=chatml) can be tested with local files. Optionally downloads images
from S3 into data/qa-multiview_qa/images/.

Usage:
    # Download 20 ChatML rows + images to experiments/10_vqa/data/qa-multiview_qa/
    python experiments/10_vqa/download_qa_multiview_local.py

    # Smaller sample, no images
    python experiments/10_vqa/download_qa_multiview_local.py --limit 5 --no-download-images

    # Custom output dir (e.g. EXTRA_EO)
    python experiments/10_vqa/download_qa_multiview_local.py --output-dir /path/to/extra/eo/data/qa-multiview_qa
"""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Default: small sample for local tests
DEFAULT_LIMIT = 20
YT_TABLE = "//home/visiondata/users/gregorykogan/eo-data/chatml/qa-multiview_qa"
DEFAULT_S3_BUCKET = "gigaeye-data"

# Output under experiments/10_vqa/data/qa-multiview_qa (relative to EO1 repo root)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data" / "qa-multiview_qa"  # EO1/experiments/10_vqa/data/qa-multiview_qa


def read_yt_table(limit: int | None, table: str = YT_TABLE) -> list[dict]:
    """Read rows from YTSaurus table using yt CLI."""
    yt_proxy = os.environ.get("YT_PROXY")
    if not yt_proxy:
        print("ERROR: YT_PROXY environment variable not set")
        sys.exit(1)

    table_path = f"{table}[:#{limit}]" if limit else table
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
        print("  yt not found, trying conda run...")
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


def download_images_for_chatml(jsonl_path: Path, images_dir: Path, bucket: str) -> None:
    """Download all image_paths referenced in a ChatML JSONL from S3."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from eo.data.s3_loader import S3Client

    secrets_path = Path(__file__).resolve().parents[3] / "secrets.env"
    secrets = {}
    if secrets_path.exists():
        with open(secrets_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    secrets[k.strip()] = v.strip().strip('"').strip("'")

    client = S3Client.create(secrets if secrets else None)
    all_keys = set()
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line.strip())
            all_keys.update(row.get("image_paths", []))
    print(f"[DOWNLOAD] Downloading {len(all_keys)} images from s3://{bucket}/ ...")
    images_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for key in all_keys:
        local_path = images_dir / key
        if download_image_from_s3(client, bucket, key, local_path):
            downloaded += 1
    print(f"  [OK] Downloaded/cached {downloaded}/{len(all_keys)} images")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Download small qa-multiview_qa sample from YTSaurus for local testing"
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"Max rows to read from YT (default: {DEFAULT_LIMIT})"
    )
    parser.add_argument(
        "--no-download-images", action="store_true",
        help="Do not download images (JSONL only, for S3-backed tests)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"Output directory (default: {DEFAULT_DATA_DIR})"
    )
    args = parser.parse_args()

    data_dir = Path(args.output_dir) if args.output_dir else DEFAULT_DATA_DIR
    jsonl_dir = data_dir / "jsonl"
    images_dir = data_dir / "images"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    chatml_path = jsonl_dir / "qa-multiview_qa_chatml.jsonl"

    rows = read_yt_table(args.limit)
    if not rows:
        print("No rows read. Exiting.")
        sys.exit(1)

    # Write ChatML rows as-is (dataset with format=chatml will convert on load)
    with open(chatml_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[OK] Wrote {len(rows)} ChatML samples to {chatml_path}")

    bucket = DEFAULT_S3_BUCKET
    if rows and rows[0].get("s3_bucket"):
        bucket = rows[0]["s3_bucket"]

    if not args.no_download_images:
        download_images_for_chatml(chatml_path, images_dir, bucket)

    print("\n" + "=" * 60)
    print("  Download complete!")
    print("=" * 60)
    print(f"  ChatML JSONL : {chatml_path}")
    print(f"  Images dir   : {images_dir}")
    print()
    print("  Local test: python experiments/10_vqa/test_multiview_dataset.py")
    print("  Config (local): data-qa-multiview-chatml-local.yaml")
    print("  Config (S3):   data-qa-multiview-chatml-s3.yaml (use full YT→JSONL first)")
    print("=" * 60)


if __name__ == "__main__":
    main()
