"""Measure route() latency on the shipped pretrained hybrid model.

Loads N random task texts from data/traces.jsonl, warms up the encoder,
then times route(task, k=3) on each task. Reports mean, p50, p90, p95, p99.

Honest numbers, single-machine, no batching: this is the per-call latency
a user would see calling Router.from_pretrained(...).route(text) one task
at a time. Not the throughput a server would get with batching.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACES_PATH = ROOT / "data" / "traces.jsonl"
DEFAULT_MODEL_DIR = ROOT / "models" / "baseline-v1-desc-hybrid"


def load_random_tasks(n: int, seed: int = 42) -> list[str]:
    tasks: list[str] = []
    with TRACES_PATH.open(encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            text = (t.get("task_text") or "").strip()
            if text:
                tasks.append(text)
    rng = random.Random(seed)
    rng.shuffle(tasks)
    return tasks[:n]


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"[load] tasks from {TRACES_PATH}", file=sys.stderr)
    tasks = load_random_tasks(args.n + args.warmup, seed=args.seed)
    if len(tasks) < args.n + args.warmup:
        raise SystemExit(
            f"only {len(tasks)} tasks available, need {args.n + args.warmup}"
        )
    warmup_tasks = tasks[: args.warmup]
    bench_tasks = tasks[args.warmup : args.warmup + args.n]

    print(f"[load] router from {args.model_dir}", file=sys.stderr)
    from agent_tool_router import Router
    t0 = time.time()
    router = Router.from_pretrained(args.model_dir)
    print(f"[load] from_pretrained in {time.time()-t0:.2f}s (lazy encoder)",
          file=sys.stderr)

    print(f"[warmup] {args.warmup} calls (encoder lazy-init on first call)",
          file=sys.stderr)
    t0 = time.time()
    for q in warmup_tasks:
        router.route(q, k=args.k)
    print(f"[warmup] done in {time.time()-t0:.2f}s", file=sys.stderr)

    print(f"[bench] {args.n} calls, k={args.k}", file=sys.stderr)
    timings_ms: list[float] = []
    t_start = time.time()
    for q in bench_tasks:
        t = time.perf_counter()
        router.route(q, k=args.k)
        timings_ms.append((time.perf_counter() - t) * 1000.0)
    t_total = time.time() - t_start
    print(f"[bench] done in {t_total:.2f}s ({args.n / t_total:.1f} calls/s)",
          file=sys.stderr)

    timings_ms.sort()
    mean = sum(timings_ms) / len(timings_ms)
    print()
    print(f"=== route() latency on {args.n} random tasks (k={args.k}) ===")
    print(f"  mean : {mean:7.2f} ms")
    print(f"  p50  : {percentile(timings_ms, 50):7.2f} ms")
    print(f"  p90  : {percentile(timings_ms, 90):7.2f} ms")
    print(f"  p95  : {percentile(timings_ms, 95):7.2f} ms")
    print(f"  p99  : {percentile(timings_ms, 99):7.2f} ms")
    print(f"  min  : {timings_ms[0]:7.2f} ms")
    print(f"  max  : {timings_ms[-1]:7.2f} ms")
    print(f"  throughput (single thread): {args.n / t_total:.1f} calls/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
