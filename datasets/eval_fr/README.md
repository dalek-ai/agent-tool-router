---
license: mit
language:
  - fr
  - en
task_categories:
  - text-retrieval
  - sentence-similarity
size_categories:
  - n<1K
pretty_name: agent-tool-router · parallel EN/FR eval
tags:
  - agents
  - tool-routing
  - function-calling
  - retrieval
  - multilingual
  - french
  - evaluation
configs:
  - config_name: default
    data_files:
      - split: test
        path: fr_eval_queries.json
---

# agent-tool-router · parallel EN/FR evaluation

50 parallel English/French queries used to evaluate
[`dalek-ai/baseline-v1-desc-hybrid`](https://huggingface.co/dalek-ai/baseline-v1-desc-hybrid)
(EN-first) versus
[`dalek-ai/baseline-v1-desc-hybrid-multilingual`](https://huggingface.co/dalek-ai/baseline-v1-desc-hybrid-multilingual)
(50+ languages) on a catalog of 18 671 tools collected from public agent
benchmarks (tau-bench, Hermes function-calling-v1, ToolACE).

## Numbers (hybrid models, α=0.5, V=18 671)

| model | top-3 EN | top-3 FR |
|---|---:|---:|
| `baseline-v1-desc-hybrid` (default, MiniLM-L6) | 82% | 26% |
| `baseline-v1-desc-hybrid-multilingual` | **82%** | **54%** |

Same EN top-3, +28pp on French.

## Schema

Each query exposes:

- `id`: stable identifier of the underlying concept (e.g. `cancel_order`).
- `en`: query text in English.
- `fr`: query text in French (manually translated, parallel intent).
- `expected_concepts`: list of concepts. A returned tool name counts as a hit
  if its normalized subtokens contain every token of at least one concept
  (order-free, case-insensitive, separator-insensitive). Lenient on naming
  conventions (`send_email` / `sendEmail` / `send-email` all match the
  `send email` concept), strict on semantic content (`cancel order` will not
  match `modify_pending_order_items`).

## How to reproduce

```bash
git clone https://github.com/dalek-ai/agent-tool-router
cd agent-tool-router
pip install -e ".[encoder]"
python -m router.eval.eval_fr_pretrained
```

The script downloads both pretrained models from the Hub and re-runs the
exact same matching logic that produced the table above.

## Related

- Code & full benchmarks: [github.com/dalek-ai/agent-tool-router](https://github.com/dalek-ai/agent-tool-router)
- Models: [huggingface.co/dalek-ai](https://huggingface.co/dalek-ai)
- Live demo: [huggingface.co/spaces/dalek-ai/agent-tool-router-demo](https://huggingface.co/spaces/dalek-ai/agent-tool-router-demo)

MIT.
