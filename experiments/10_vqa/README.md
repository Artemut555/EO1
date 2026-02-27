# Experiment 10: Robotics VQA fine-tuning

Fine-tune EO-1 (Qwen2.5-VL-3B) on robotics VQA datasets:
multiview correspondence, object reasoning, failure detection, etc.

## File structure

```
10_vqa/
  train.sh                            # Single training script (env-var driven)
  configs/
    data/                             # Dataset configs ({name}_{backend}.yaml)
      multiview_local.yaml            #   Multiview QA, LLaVA format, local images
      multiview_s3.yaml               #   Multiview QA, LLaVA format, S3 images
      multiview_chatml_local.yaml     #   Multiview QA, ChatML format, local
      multiview_chatml_s3.yaml        #   Multiview QA, ChatML format, S3
      vqa_robot_local.yaml            #   EO-Data1.5M robot VQA, local
      vqa_robot_s3.yaml               #   EO-Data1.5M robot VQA, S3
      refcoco_demo.yaml               #   RefCOCO demo (bundled with EO1)
    profiles/                         # Training presets (source before train.sh)
      debug.env                       #   1 GPU, 20 steps
      local_2gpu.env                  #   2 GPUs, 20 steps
      prod.env                        #   8 GPUs, 3 epochs, flash_attention_2
      local_2gpu_pack.env             #   2 GPUs, packing smoke test (EO-1-like LRs)
      chatml_s3_32gpu.env             #   32 GPUs cloud run
  scripts/
    download_vqa_robot.py             # Download EO-Data1.5M QA from HuggingFace
    download_multiview.py             # Download multiview QA from YTSaurus + S3
    upload_to_s3.py                   # Upload local VQA data to S3
  tests/
    test_dataset.py                   # Dataset loading smoke test
  data/                               # Local test data (gitignored)
```

## Quick start

All commands run from the EO1 repo root (`cd eo_experiment/EO1`).

### 1. Local debug (1 GPU, 20 steps, built-in data)

```bash
PROFILE=debug DATASET=experiments/10_vqa/configs/data/refcoco_demo.yaml \
  bash experiments/10_vqa/train.sh
```

### 2. Local 2-GPU multiview training

```bash
PROFILE=local_2gpu bash experiments/10_vqa/train.sh
# Uses multiview_local.yaml by default
```

### 2b. Local 2-GPU test on EO ChatML cleaned (S3 images)

Use the **eo** conda env so multiple DataLoader workers work with S3 (avoids SSL errors in workers):

```bash
conda activate eo
cd eo_experiment/EO1  # from qwen_tune root

PROFILE=local_2gpu DATASET=experiments/10_vqa/configs/data/eo_chatml_cleaned_s3.yaml \
  bash experiments/10_vqa/train.sh
```
Default `NUM_WORKERS=4` is used. If you see S3 SSL errors in workers, use `NUM_WORKERS=0`.

### 3. Override any parameter

```bash
GPUS=4 MAX_STEPS=100 DATASET=experiments/10_vqa/configs/data/vqa_robot_local.yaml \
  bash experiments/10_vqa/train.sh
```

### 4. Cloud job (cloud.ru)

From `qwen_tune` root:

**Pipeline test (~5 min, 32 GPUs)** — uses in-repo refcoco demo; no S3/mounts required:
```bash
python eo_experiment/cloud_job/send_job.py --profile pipeline_test_32gpu
```

**Full ChatML S3 (32 GPUs)** — requires multiview ChatML JSONL and S3 credentials in image/mount:
```bash
python eo_experiment/cloud_job/send_job.py --profile chatml_s3_32gpu
```

**EO-1-like full run (32 GPUs)** — EO ChatML cleaned (no actions), packing, 5 epochs, paper LRs. Requires project/extra folder mounted on cloud.ru and S3 for images:
```bash
python eo_experiment/cloud_job/send_job.py --profile eo1_like_full_run
```

### 5. Local 2-GPU smoke test with packing (before cloud full run)

From EO1 repo root, with conda env `eo` (for S3):

```bash
conda activate eo
PROFILE=local_2gpu_pack DATASET=experiments/10_vqa/configs/data/eo_chatml_cleaned_s3_smoke.yaml \
  bash experiments/10_vqa/train.sh
```

This runs 8 steps with `PACK_DATASET=True`, `MAX_PACKED_LENGTH=16384`, and EO-1-like LRs on 400 samples for a fast packing/config check.

## Data configs

| Config | Format | Backend | Dataset |
|--------|--------|---------|---------|
| `multiview_local.yaml` | LLaVA | local | Multiview stereo correspondence QA |
| `multiview_s3.yaml` | LLaVA | S3 | Same, images from S3 |
| `multiview_chatml_local.yaml` | ChatML | local | Multiview QA (raw ChatML, converted on load) |
| `multiview_chatml_s3.yaml` | ChatML | S3 | Same, images from S3 |
| `vqa_robot_local.yaml` | LLaVA | local | EO-Data1.5M robotics VQA |
| `vqa_robot_s3.yaml` | LLaVA | S3 | Same, images from S3 |
| `refcoco_demo.yaml` | LLaVA | local | RefCOCO demo (bundled) |
| `eo_chatml_cleaned_s3.yaml` | ChatML | S3 | EO ChatML cleaned (state/action removed), first:300 from interleave-free_chat |
| `eo_chatml_cleaned_s3_smoke.yaml` | ChatML | S3 | Same, first:400 (for local packing smoke test) |
| `eo_chatml_cleaned_s3_micro.yaml` | ChatML | S3 | Same, first:10 (micro test; use with `local_2gpu_pack_micro`) |
| `eo_chatml_cleaned_full_s3.yaml` | ChatML | S3 | Full 16 files, random:100% (for EO-1-like full run; mount required) |

## Packing, attention, and base VLM

### How packing is organized

When `PACK_DATASET=True`, `PackedDataset` (in `eo/data/dataset.py`) does **greedy sample packing** before training:

1. Each sample has a `seq_length` (from the data or default 196). The dataset exposes `lengths` for every item.
2. A single pass over the dataset (shuffled) fills buffers up to `max_packed_length` (default 16384). Samples are concatenated into "packs" so that each training step sees one pack = one long sequence.
3. `MultimodaPackedDataCollator` takes a pack (list of examples), **concatenates** their `input_ids`, `position_ids`, and `labels` along the sequence dimension, and returns a batch with batch size 1 and sequence length = sum of lengths in the pack.

So each training step is one packed sequence of multiple original samples (vision+text dialogs) placed back-to-back.

**Token / seq_length warnings:** If you see `[Warning] seq_len=0, json_path=...` during dataset load, some rows have no or invalid `seq_length`. The loader uses default `seq_length=196` for ChatML/LLaVA rows that do not set it (`eo/data/multim_dataset.py`). Rows with `seq_length > max_seq_length` (default 8192) are filtered out. To avoid seq_len=0 warnings, ensure your JSONL has valid `seq_length` per row or rely on the default 196.

### Attention over packed (interleaved) data: are previous questions visible?

The packed collator does **not** add a custom attention mask that would hide later samples from earlier ones. The model uses its **default causal (decoder) attention**. So the full packed sequence is one causal chain: **later samples in the pack can attend to all previous tokens**, including tokens from earlier samples (earlier questions/answers and images). So **packed samples are not independent**: later questions in the same pack do "see" previous questions and answers. If you need independence (e.g. no cross-sample attention), you would have to add a custom attention mask (e.g. block-diagonal) and ensure the code path uses it; the current code does not.

### Which repository was used to finetune Qwen2.5-VL-Instruct?

The **base model** we use is **Qwen/Qwen2.5-VL-3B-Instruct** from Hugging Face (released by the Qwen team). It is a pretrained vision-language model; we do not train it from scratch in this repo.

For **multimodal data format and finetuning recipes**, the EO-1 codebase (see `getting_started/1_load_dataset.ipynb`) references:

- **[Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)** — official Qwen vision-language repo (cookbooks, spatial understanding, etc.).
- **[2U1/Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune)** — community finetune recipe and data format used by EO-1 for image/video/text/points/bboxes.

The **Qwen2.5-VL-Instruct** checkpoint itself is produced by the Qwen team; the **finetuning recipe** (how to finetune that VLM on custom data) is what EO-1 aligns with via 2U1/Qwen2-VL-Finetune and Qwen2.5-VL docs.

## Downloading data

```bash
# EO-Data1.5M robot VQA (from HuggingFace, ~3 GB)
python experiments/10_vqa/scripts/download_vqa_robot.py

# Multiview ChatML (from YTSaurus, 20 samples + images for local test)
python experiments/10_vqa/scripts/download_multiview.py --format chatml --limit 20

# Multiview LLaVA (all rows, no images — for S3-backed training)
python experiments/10_vqa/scripts/download_multiview.py --format llava --no-download-images
```

## Adding a new dataset

1. Create a YAML in `configs/data/` following the `{name}_{backend}.yaml` convention.
2. Run training: `DATASET=experiments/10_vqa/configs/data/your_new.yaml bash experiments/10_vqa/train.sh`

## Adding a new training profile

1. Create a `.env` file in `configs/profiles/` with the variables you want to set.
2. Use it: `PROFILE=your_profile bash experiments/10_vqa/train.sh`
