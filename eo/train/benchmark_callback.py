"""
Benchmark instrumentation for measuring data-loading vs compute time.

Activated by setting BENCHMARK=true in the environment. Produces a JSON
report at {output_dir}/benchmark_results.json with per-step timing and
aggregate statistics.

Measurement approach: the HuggingFace Trainer fetches the next batch
BEFORE calling on_step_begin. So the gap between on_step_end(n-1) and
on_step_begin(n) captures the data-loading time, while on_step_begin(n)
to on_step_end(n) captures compute (forward + backward + optimizer).
"""

import json
import os
import time
from pathlib import Path
from statistics import mean, median

from transformers import TrainerCallback


WARMUP_STEPS = 5


def _percentile(data, p):
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


class BenchmarkCallback(TrainerCallback):
    """Records per-step wall time and data-loading time, writes JSON report."""

    def __init__(self, metadata=None):
        self.metadata = metadata or {}
        self._records = []
        self._step_begin_t = 0.0
        self._prev_step_end_t = 0.0
        self._train_start = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        self._train_start = time.monotonic()
        self._prev_step_end_t = time.monotonic()

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_begin_t = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        now = time.monotonic()
        compute_time = now - self._step_begin_t
        data_load_time = self._step_begin_t - self._prev_step_end_t
        step_time = compute_time + data_load_time

        self._records.append({
            "step": state.global_step,
            "step_time": round(step_time, 6),
            "data_load_time": round(data_load_time, 6),
            "compute_time": round(compute_time, 6),
        })
        self._prev_step_end_t = now

    def on_train_end(self, args, state, control, **kwargs):
        total_wall = time.monotonic() - self._train_start
        self._write_report(args.output_dir, total_wall, state)

    def _write_report(self, output_dir, total_wall, state):
        warm = [r for r in self._records if r["step"] > WARMUP_STEPS]

        def _stats(values):
            if not values:
                return {}
            s = sorted(values)
            return {
                "mean": round(mean(s), 4),
                "median": round(median(s), 4),
                "p95": round(_percentile(s, 95), 4),
                "min": round(s[0], 4),
                "max": round(s[-1], 4),
            }

        warm_step = [r["step_time"] for r in warm]
        warm_data = [r["data_load_time"] for r in warm]
        warm_compute = [r["compute_time"] for r in warm]

        total_data_pct = 0.0
        if warm_step:
            total_data_pct = round(sum(warm_data) / sum(warm_step) * 100, 2)

        n_gpus = int(os.environ.get("WORLD_SIZE", os.environ.get("TOTAL_PROCS", "1")))
        batch_per_gpu = int(os.environ.get("PER_DEVICE_BATCH_SIZE", "1"))
        samples_per_step = n_gpus * batch_per_gpu

        summary = {
            "metadata": {
                **self.metadata,
                "n_gpus": n_gpus,
                "batch_per_gpu": batch_per_gpu,
                "samples_per_step": samples_per_step,
                "total_steps": len(self._records),
                "warmup_steps_excluded": WARMUP_STEPS,
                "measured_steps": len(warm),
            },
            "totals": {
                "total_wall_sec": round(total_wall, 2),
                "total_train_steps": state.global_step,
                "samples_per_sec": round(
                    len(warm) * samples_per_step / sum(warm_step), 2
                ) if warm_step else 0,
            },
            "step_time": _stats(warm_step),
            "data_load_time": _stats(warm_data),
            "compute_time": _stats(warm_compute),
            "data_load_pct_of_step": total_data_pct,
            "per_step": self._records,
        }

        out_path = Path(output_dir) / "benchmark_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        if rank == 0:
            out_path.write_text(json.dumps(summary, indent=2))
            _print_summary(summary)



def _print_summary(s):
    meta = s["metadata"]
    vb = meta.get("vision_backend", "?")
    sep = "=" * 60
    print()
    print(sep)
    print("  BENCHMARK RESULTS")
    print(sep)
    print("  Vision backend : " + str(vb))
    print("  GPUs           : " + str(meta["n_gpus"]))
    print("  Batch/GPU      : " + str(meta["batch_per_gpu"]))
    print("  Total steps    : " + str(meta["total_steps"]))
    print("  Measured steps : " + str(meta["measured_steps"]) + "  (first " + str(meta["warmup_steps_excluded"]) + " excluded)")
    print("-" * 60)
    fmt = "  {:<20s}  mean={:<8s} median={:<8s} p95={:<8s}"
    for label, key in [("Step time (s)", "step_time"),
                       ("Data load (s)", "data_load_time"),
                       ("Compute (s)", "compute_time")]:
        st = s[key]
        if st:
            print(fmt.format(label, str(st["mean"]), str(st["median"]), str(st["p95"])))
    print("  Data load pct of step : " + str(s["data_load_pct_of_step"]) + "%")
    print("  Throughput          : " + str(s["totals"]["samples_per_sec"]) + " samples/sec")
    print("  Total wall time     : " + str(s["totals"]["total_wall_sec"]) + "s")
    print(sep)
    print()
