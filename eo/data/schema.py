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

"""Schema for the dataset configuration."""

from dataclasses import dataclass, field
from typing import Literal

import yaml

# Supported multimodal text+image dataset formats (per config).
MMFormat = Literal["llava", "chatml"]


@dataclass
class MDSDatasetConfig:
    """MDS streaming dataset with embedded images (mds_shard_writer format)."""

    remote: str | None = None  # e.g. s3://bucket/prefix/mds
    local: str | None = None  # local cache path (required when remote is set)
    shuffle: bool = False
    shuffle_seed: int = 9176
    cache_limit_shards: int = 32


@dataclass
class MMDatasetConfig:
    json_path: str
    sampling_strategy: str = "all"
    vision_base_path: str | None = None
    vision_backend: str = "local"
    s3_bucket: str | None = None
    s3_prefix: str | None = None
    # "llava": JSONL with conversations (from/human,gpt) + image paths.
    # "chatml": JSONL with dialogue (role/user,assistant) + image_paths; converted to LLaVA-like internally.
    format: MMFormat = "llava"
    # Optional YT table path for docs/download scripts; dataset loads from json_path (e.g. after download).
    yt_table: str | None = None


@dataclass
class LerobotConfig:
    repo_id: str
    root: str | None = None
    episodes: list[int] | None = None
    delta_action: bool = False
    state_mode: str = "MEAN_STD"

    train_subtask: str | bool | None = False  # Optional[true, false, mix:0.5, cumulate]
    select_video_keys: list[str] = None
    select_action_keys: list[str] = None
    select_state_keys: list[str] = None
    effector_indices: list[int] = None
    weight: float | None = None


@dataclass
class DataConfig:
    mm_datasets: list[MMDatasetConfig] = field(default_factory=list)
    mds_datasets: list[MDSDatasetConfig] = field(default_factory=list)
    lerobot_datasets: list[LerobotConfig] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> "DataConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            mm_datasets=[MMDatasetConfig(**d) for d in raw.get("mm_datasets") or []],
            mds_datasets=[MDSDatasetConfig(**d) for d in raw.get("mds_datasets") or []],
            lerobot_datasets=[LerobotConfig(**d) for d in raw.get("lerobot_datasets") or []],
        )
