#!/usr/bin/env bash
# Rebuild data/traces.jsonl from the public benchmark sources.
# Run from the repo root.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p data

# tau-bench (gold actions inside historical_trajectories/).
if [ ! -d "data/tau-bench" ]; then
    git clone --depth 1 https://github.com/sierra-research/tau-bench.git data/tau-bench
fi

# OSWorld (task definitions only — no realized trajectories).
if [ ! -d "data/osworld" ]; then
    git clone --depth 1 https://github.com/xlang-ai/OSWorld.git data/osworld
fi

# SWE-bench Verified, Hermes, ToolACE come from HuggingFace via the loaders.
python3 -c "import datasets" >/dev/null 2>&1 || pip install --user datasets

python3 router/index/aggregate.py
echo
echo "Done. data/traces.jsonl is ready."
