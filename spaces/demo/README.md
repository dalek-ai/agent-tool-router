---
title: agent-tool-router
emoji: 🛠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
short_description: Route an agent task (FR/EN) to top-k tools out of 18K
tags:
  - agents
  - tool-routing
  - function-calling
  - retrieval
  - sentence-similarity
  - multilingual
  - french
---

# agent-tool-router · live demo

Type a task in English or French. The router returns the top-k tools to
call from a catalog of 18 671 tools collected from public agent benchmarks
(tau-bench, Hermes function-calling-v1, ToolACE).

Two pretrained models are loaded:

- **`baseline-v1-desc-hybrid`** — English-first, MiniLM-L6, ~35 MB centroids.
- **`baseline-v1-desc-hybrid-multilingual`** — 50+ languages, FR-friendly,
  ~80 MB centroids. On a 50-query parallel EN/FR probe, FR top-3 jumps from
  26% (default) to 54%, with EN top-3 flat at 82%.

Code, dataset rebuild script and full numbers:
[github.com/dalek-ai/agent-tool-router](https://github.com/dalek-ai/agent-tool-router).
Models: [huggingface.co/dalek-ai](https://huggingface.co/dalek-ai).
Parallel EN/FR eval queries (n=50):
[dalek-ai/agent-tool-router-eval-fr](https://huggingface.co/datasets/dalek-ai/agent-tool-router-eval-fr).
MIT.
