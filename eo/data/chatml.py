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
ChatML format support for EO datasets.

ChatML rows (e.g. from YTSaurus) use:
  - dialogue: list of {role, content, files?, trainable?}
  - image_paths: list of S3 or local paths
  - s3_bucket: optional, per-row

We convert to the same LLaVA-style shape used by default (e.g. refcoco.jsonl):
  - conversations: list of {from: "human"|"gpt", value: "..."}  with "<image>" in value
  - image: list of paths (or single path)
  - s3_bucket: preserved from row if present

MultimodaDataset.__getitem__ runs llava_to_openai() on conversations for both
native LLaVA and ChatML-converted rows, so "<image>" is replaced with
<|vision_start|><|image_pad|><|vision_end|> and position_ids are produced
identically via get_rope_index_25 in MultimodaLeRobotDataset.
"""

from __future__ import annotations

from typing import Any

CHATML_IMAGE_TOKEN = "<image>"


def chatml_row_to_llava(row: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a single ChatML row to internal LLaVA-style format.

    Args:
        row: ChatML dict with "dialogue", "image_paths", optionally "s3_bucket".

    Returns:
        Dict with "conversations", "image"; optional "s3_bucket". None if invalid.
    """
    dialogue = row.get("dialogue")
    if not dialogue:
        return None

    image_paths = row.get("image_paths", [])
    if not image_paths:
        return None

    conversations = []
    for msg in dialogue:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            continue
        if role == "user":
            files = msg.get("files", [])
            n_images = len([f for f in files if f.get("type") == "image"])
            if n_images == 0:
                n_images = len(image_paths)
            image_prefix = CHATML_IMAGE_TOKEN * n_images
            conversations.append({
                "from": "human",
                "value": f"{image_prefix}{content}",
            })
        elif role == "assistant":
            conversations.append({
                "from": "gpt",
                "value": content,
            })

    if len(conversations) < 2:
        return None

    entry: dict[str, Any] = {
        "conversations": conversations,
        "image": image_paths if len(image_paths) > 1 else image_paths[0],
    }
    if row.get("s3_bucket"):
        entry["s3_bucket"] = row["s3_bucket"]
    return entry
