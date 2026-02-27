#!/usr/bin/env python3
"""
Download multiview QA data from YTSaurus and optionally download images from S3.

Supports two output formats:
  - chatml: raw ChatML JSONL (converted on-the-fly during training)
  - llava:  pre-converted LLaVA-style JSONL (conversations + image paths)

Usage:
    # ChatML format, small local test (20 samples + images)
    python experiments/10_vqa/scripts/download_multiview.py --format chatml --limit 20

    # LLaVA format, all rows, S3 image paths (for cloud training)
    python experiments/10_vqa/scripts/download_multiview.py --format llava --no-download-images

    # ChatML, 5000 rows, no images (S3-backed training)
    python experiments/10_vqa/scripts/download_multiview.py --format chatml --limit 5000 --no-download-images

    # Dry run: print converted samples
    python experiments/10_vqa/scripts/download_multiview.py --limit 5 --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from eo.data.chatml import chatml_row_to_llava

# -----------------------------------------------
# Configuration
# -----------------------------------------------
YT_TABLE = "//home/visiondata/users/gregorykogan/eo-data/chatml/qa-multiview_qa"
DEFAULT_S3_BUCKET = "gigaeye-data"
DEFAULT_LIMIT = 20

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHATML_DIR = SCRIPT_DIR / "data" / "qa-multiview_qa"
DEFAULT_LLAVA_DIR = REPO_ROOT.parent.parent / "data_multiview_chatml"


# -----------------------------------------------
# YTSaurus reader
# -----------------------------------------------
def read_yt_table(limit: int | None = None, table: str = YT_TABLE) -> list[dict]:
    """Read rows from YTSaurus table using yt CLI."""
    yt_proxy = os.environ.get("YT_PROXY")
    if not yt_proxy:
        print("ERROR: YT_PROXY environment variable not set")
        sys.exit(1)

    table_path = f"{table}[:#{limit}]" if limit else table
    cmd = ["yt", "read", "--proxy", yt_proxy, "--format", "json", table_path]

    print(f"[READ] Reading from {table_path} ...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        print("  yt not found, trying conda run...")
        cmd = ["conda", "run", "-n", "avzavarzin"] + cmd
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    rows = [json.loads(line) for line in result.stdout.strip().split("\n") if line]
    print(f"  Read {len(rows)} rows")
    return rows


# -----------------------------------------------
# S3 image download
# -----------------------------------------------
def _get_s3_client():
    """Lazy-init S3 client."""
    from eo.data.s3_loader import S3Client
    secrets_path = REPO_ROOT.parent / "secrets.env"
    secrets = {}
    if secrets_path.exists():
        with open(secrets_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    secrets[k.strip()] = v.strip().strip('"').strip("'")
    return S3Client.create(secrets if secrets else None)


def download_image(s3_client, bucket: str, key: str, local_path: Path) -> bool:
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


def download_images_for_jsonl(jsonl_path: Path, images_dir: Path, bucket: str,
                              key_field: str = "image_paths"):
    """Download all images referenced in a JSONL from S3."""
    client = _get_s3_client()
    all_keys = set()
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line.strip())
            paths = row.get(key_field, [])
            if isinstance(paths, str):
                paths = [paths]
            all_keys.update(paths)

    print(f"[DOWNLOAD] Downloading {len(all_keys)} images from s3://{bucket}/ ...")
    images_dir.mkdir(parents=True, exist_ok=True)
    downloaded, failed = 0, 0

    def _dl(key):
        return key, download_image(client, bucket, key, images_dir / key)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_dl, k): k for k in all_keys}
        for future in as_completed(futures):
            _, ok = future.result()
            if ok:
                downloaded += 1
            else:
                failed += 1
            total = downloaded + failed
            if total % 500 == 0:
                print(f"  Progress: {total}/{len(all_keys)} (ok={downloaded}, fail={failed})")

    print(f"  [OK] Downloaded/cached: {downloaded}, Failed: {failed}")


# -----------------------------------------------
# Main
# -----------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download multiview QA data from YTSaurus"
    )
    parser.add_argument("--format", choices=["chatml", "llava"], default="chatml",
                        help="Output JSONL format (default: chatml)")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"Max rows to read (default: {DEFAULT_LIMIT}, 0=all)")
    parser.add_argument("--no-download-images", action="store_true",
                        help="Skip downloading images from S3")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print samples instead of writing files")
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None

    # Determine output directory
    if args.output_dir:
        data_dir = Path(args.output_dir)
    elif args.format == "chatml":
        data_dir = DEFAULT_CHATML_DIR
    else:
        data_dir = DEFAULT_LLAVA_DIR
    jsonl_dir = data_dir / "jsonl"
    images_dir = data_dir / "images"

    rows = read_yt_table(limit)
    if not rows:
        print("No rows read. Exiting.")
        sys.exit(1)

    bucket = rows[0].get("s3_bucket", DEFAULT_S3_BUCKET) if rows else DEFAULT_S3_BUCKET

    if args.format == "chatml":
        # Write raw ChatML rows (dataset loader converts on-the-fly)
        jsonl_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = jsonl_dir / "qa-multiview_qa_chatml.jsonl"

        if args.dry_run:
            for row in rows[:10]:
                print(json.dumps(row, indent=2, ensure_ascii=False))
                print("---")
            return

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[OK] Wrote {len(rows)} ChatML samples to {jsonl_path}")

        if not args.no_download_images:
            download_images_for_jsonl(jsonl_path, images_dir, bucket, key_field="image_paths")

    else:
        # Convert to LLaVA format
        converted, skipped = [], 0
        for row in rows:
            entry = chatml_row_to_llava(row)
            if entry is None:
                skipped += 1
                continue
            converted.append(entry)
        print(f"[CONVERT] Converted {len(converted)} samples ({skipped} skipped)")

        if args.dry_run:
            for entry in converted[:10]:
                print(json.dumps(entry, indent=2, ensure_ascii=False))
                print("---")
            return

        jsonl_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = jsonl_dir / "multiview_chatml.jsonl"
        with open(jsonl_path, "w") as f:
            for entry in converted:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"[OK] Wrote {len(converted)} LLaVA samples to {jsonl_path}")

        if not args.no_download_images:
            download_images_for_jsonl(jsonl_path, images_dir, bucket, key_field="image")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Download complete!")
    print(f"{'=' * 60}")
    print(f"  Format       : {args.format}")
    print(f"  Samples      : {len(rows)}")
    print(f"  JSONL        : {jsonl_dir}")
    if not args.no_download_images:
        print(f"  Images       : {images_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
