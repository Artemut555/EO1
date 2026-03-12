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
Streaming dataset for MDS format (mds_shard_writer) with embedded images.

Uses MosaicML StreamingDataset for remote/local streaming, caching, and shuffle.
The mds_shard_writer binary layout matches MosaicML's native MDS encoding,
so StreamingDataset decodes samples natively via column_names/column_encodings
defined in the shard index.
"""

from __future__ import annotations

import io
import logging
import tarfile
from typing import Any

import numpy as np
import torch
from PIL import Image
from streaming import StreamingDataset

from eo.train.pipeline_config import TrainPipelineConfig

from .rope2d import get_rope_index_25

log = logging.getLogger(__name__)

CODE2DTYPE = {0: np.uint16, 1: np.uint32, 2: np.int64, 3: np.int32}


def _bytes_to_tensors(
    tokens: bytes, labels: bytes, dtype: np.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert raw token/label bytes to int64 torch tensors."""
    if not tokens:
        raise ValueError("Empty tokens")
    input_ids = torch.from_numpy(np.frombuffer(tokens, dtype=dtype).copy().astype(np.int64))
    if not labels:
        labels_t = torch.full_like(input_ids, -100)
    else:
        labels_t = torch.from_numpy(np.frombuffer(labels, dtype=dtype).copy().astype(np.int64))
    return input_ids, labels_t


def _extract_images(payloads: bytes) -> list[bytes]:
    """Extract image bytes from a TAR payload."""
    if not payloads:
        return []
    images = []
    with tarfile.open(fileobj=io.BytesIO(payloads), mode="r:*") as tar:
        for m in tar.getmembers():
            if m.isfile():
                f = tar.extractfile(m)
                if f is not None:
                    images.append(f.read())
    return images


class EOMDSStreamingDataset(StreamingDataset):
    """MDS dataset with embedded images (mds_shard_writer format).

    Returns the same dict format as MultimodaLeRobotDataset:
    input_ids, labels, position_ids, pixel_values, image_grid_thw.
    """

    def __init__(
        self,
        args: TrainPipelineConfig,
        processor: Any,
        remote: str | None = None,
        local: str | None = None,
        shuffle: bool = False,
        shuffle_seed: int = 9176,
        cache_limit_shards: int = 32,
        default_seq_length: int = 196,
        predownload: int | None = 100_000,
        download_retry: int = 2,
        download_timeout: float = 60,
        **kwargs: Any,
    ):
        if not remote and not local:
            raise ValueError("Either remote or local must be set for MDS dataset")
        if remote and not local:
            raise ValueError("local cache path is required when using remote MDS")

        cache_limit_bytes = cache_limit_shards * 64 * 1024 * 1024 if cache_limit_shards else None

        super().__init__(
            remote=remote,
            local=local or "/tmp/mds-cache",
            split=None,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            predownload=predownload,
            cache_limit=cache_limit_bytes,
            download_retry=download_retry,
            download_timeout=download_timeout,
            **kwargs,
        )

        self.args = args
        self.processor = processor
        self.default_seq_length = default_seq_length
        self.merge_size = processor.image_processor.merge_size
        self.get_rope_index = get_rope_index_25
        self._lengths: list[int] | None = None

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(i)

        dtype_code = int(sample.get("dtype", 2))
        dtype = CODE2DTYPE.get(dtype_code, np.int64)

        input_ids, labels = _bytes_to_tensors(sample["tokens"], sample["labels"], dtype)

        raw_images = _extract_images(sample["payloads"])
        pil_images = [Image.open(io.BytesIO(img)).convert("RGB") for img in raw_images]

        pixel_values = None
        image_grid_thw = None
        if pil_images:
            img_kwargs = {
                "min_pixels": self.args.image_min_pixels,
                "max_pixels": self.args.image_max_pixels,
            }
            if self.args.image_resized_width and self.args.image_resized_height:
                img_kwargs["resized_width"] = self.args.image_resized_width
                img_kwargs["resized_height"] = self.args.image_resized_height
            out = self.processor.image_processor(images=pil_images, **img_kwargs)
            pv = out["pixel_values"]
            ig = out["image_grid_thw"]
            pixel_values = pv if isinstance(pv, torch.Tensor) else torch.from_numpy(pv)
            image_grid_thw = ig if isinstance(ig, torch.Tensor) else torch.from_numpy(ig)

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            input_ids.unsqueeze(0),
            image_grid_thw=image_grid_thw,
        )

        data_dict: dict[str, Any] = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
        }
        if pixel_values is not None:
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_grid_thw
        return data_dict

    @property
    def lengths(self) -> list[int]:
        """Sequence lengths for packing (cached after first call)."""
        if self._lengths is not None:
            return self._lengths

        n = len(self)
        lengths = []
        for i in range(n):
            try:
                sample = super().__getitem__(i)
                lengths.append(int(sample.get("seq_length", self.default_seq_length)))
            except Exception:
                lengths.append(self.default_seq_length)

        self._lengths = lengths
        return self._lengths
