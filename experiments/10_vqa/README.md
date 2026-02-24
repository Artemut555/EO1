# Experiment 10: Robotics VQA fine-tuning

Fine-tune EO-1 (Qwen2.5-VL-3B) on robotics VQA from EO-Data1.5M QA subsets.

When data and checkpoints live under `EXTRA_EO` (~/projects/extra/eo), use the parent repo's helper to generate a data yaml, then pass it to train:

```bash
# From qwen_tune root
python eo_experiment/scripts/write_data_yaml.py -o eo_experiment/config/data-vqa-extra.yaml
cd eo_experiment/EO1
accelerate launch ... --vlm-name-or-path "${EXTRA_EO}/checkpoints/pretrained/Qwen2.5-VL-3B-Instruct" \
  --data-path ../../config/data-vqa-extra.yaml ...
```

See the parent repo README for full instructions.
