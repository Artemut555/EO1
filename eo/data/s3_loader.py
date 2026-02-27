# Copyright 2025 EO-Robotics Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
S3 image loader for EO dataset.
Uses max_pool_connections=1 as required for our S3 backend.

SSL handling (for cloud.ru and similar restricted networks):

  S3_SSL_VERIFY=false  →  pass verify=False to boto3 AND downgrade endpoint
                          from https:// to http:// (avoids SSL record-layer
                          failures caused by proxies / firewalls).

The OBS endpoint (obs.ru-moscow-1.hc.sbercloud.ru) accepts both HTTP and HTTPS,
and HTTP avoids all SSL issues while being slightly faster on the internal network.
"""

import logging
import os
import ssl
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import urllib3
from PIL import Image

logger = logging.getLogger(__name__)

_SSL_DISABLED = os.environ.get("S3_SSL_VERIFY", "true").lower() in ("false", "0", "no")

if _SSL_DISABLED:
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_S3_CLIENT: Optional["S3Client"] = None

_DOWNLOAD_MAX_RETRIES = 5
_DOWNLOAD_RETRY_BACKOFF = 2.0


def _load_secrets() -> dict[str, str]:
    """Load S3 secrets from env or secrets.env file."""
    if os.environ.get("S3_ENDPOINT"):
        return {
            "S3_ENDPOINT": os.environ["S3_ENDPOINT"],
            "S3_ACCESS_KEY": os.environ.get("S3_ACCESS_KEY", ""),
            "S3_SECRET_KEY": os.environ.get("S3_SECRET_KEY", ""),
        }
    bases = [Path.cwd()]
    if os.environ.get("S3_SECRETS_PATH"):
        bases.insert(0, Path(os.environ["S3_SECRETS_PATH"]))
    bases.extend([Path.cwd().parent, Path(__file__).resolve().parents[4]])
    for base in bases:
        if not base or not base.exists():
            continue
        path = base / "secrets.env" if base.is_dir() else base
        if path.exists():
            secrets = {}
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        secrets[k.strip()] = v.strip().strip('"').strip("'")
            return secrets
    return {}


class S3Client:
    """S3 client with max_pool_connections=1 and optional SSL bypass."""

    def __init__(self, endpoint: str, access_key: str, secret_key: str):
        import boto3
        from botocore.client import Config as BotoConfig

        if _SSL_DISABLED and endpoint.startswith("https://"):
            endpoint = endpoint.replace("https://", "http://", 1)
            logger.info("S3_SSL_VERIFY=false → downgraded endpoint to HTTP: %s", endpoint)

        self.endpoint = endpoint
        self.client = boto3.client(
            service_name="s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint,
            verify=not _SSL_DISABLED,
            config=BotoConfig(
                s3={"addressing_style": "virtual"},
                retries={"max_attempts": 30, "mode": "standard"},
                read_timeout=360,
                max_pool_connections=1,
            ),
        )

    @classmethod
    def create(cls, secrets: Optional[dict[str, str]] = None) -> "S3Client":
        if secrets is None:
            secrets = _load_secrets()
        ak = secrets.get("S3_ACCESS_KEY") or secrets.get("S3_DOWNLOAD_ACCESS_KEY")
        sk = secrets.get("S3_SECRET_KEY") or secrets.get("S3_DOWNLOAD_SECRET_KEY")
        ep = secrets.get("S3_ENDPOINT")
        if not ep or not ak or not sk:
            raise ValueError("S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY required")
        return cls(endpoint=ep, access_key=ak, secret_key=sk)

    def download(self, bucket: str, key: str) -> bytes:
        last_exc = None
        for attempt in range(_DOWNLOAD_MAX_RETRIES):
            try:
                response = self.client.get_object(Bucket=bucket, Key=key)
                return response["Body"].read()
            except (ssl.SSLError, OSError, ConnectionError) as exc:
                last_exc = exc
                wait = _DOWNLOAD_RETRY_BACKOFF * (2 ** attempt)
                logger.warning(
                    "S3 download %s/%s attempt %d/%d failed (%s: %s), retrying in %.1fs",
                    bucket, key, attempt + 1, _DOWNLOAD_MAX_RETRIES,
                    type(exc).__name__, exc, wait,
                )
                time.sleep(wait)
        raise last_exc  # type: ignore[misc]


def get_s3_client(secrets: Optional[dict[str, str]] = None) -> S3Client:
    global _S3_CLIENT
    if _S3_CLIENT is None:
        _S3_CLIENT = S3Client.create(secrets)
    return _S3_CLIENT


def load_image_from_s3(bucket: str, key: str) -> Image.Image:
    """Download image from S3 and return a fully-decoded PIL Image.

    Forces .load() so corrupt data is caught here (with retries) rather than
    later in the pipeline when the BytesIO buffer context is gone.
    """
    last_exc: Optional[Exception] = None
    client = get_s3_client()
    for attempt in range(_DOWNLOAD_MAX_RETRIES):
        try:
            data = client.download(bucket, key)
            img = Image.open(BytesIO(data))
            img.load()
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except (OSError, IOError) as exc:
            last_exc = exc
            wait = _DOWNLOAD_RETRY_BACKOFF * (2 ** attempt)
            logger.warning(
                "S3 image decode %s/%s attempt %d/%d failed (%s: %s), retrying in %.1fs",
                bucket, key, attempt + 1, _DOWNLOAD_MAX_RETRIES,
                type(exc).__name__, exc, wait,
            )
            time.sleep(wait)
    raise last_exc  # type: ignore[misc]
