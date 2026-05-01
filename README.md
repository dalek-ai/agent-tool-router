# agent-tool-router

> Pick the right tools for an agent task. Boring baseline. Open dataset.

```python
from agent_tool_router import Router

r = Router.from_pretrained("baseline-v0")
r.route("Cancel my pending order and refund the credit", k=3)
# ['cancel_pending_order', 'return_delivered_order_items', 'transfer_to_human_agents']
```

## What this is

Most agent stacks today wire up a fixed bag of tools and let the LLM figure out
when to call what. That works until the bag has more than ~30 tools, at which
point prompt-stuffed tool descriptions blow up the context, latency creeps,
and routing decisions start to get random.

`agent-tool-router` is a small library that takes a task description and
returns the top-k tools to use, ranked. The first model is a centroid retrieval
baseline trained on **14 000 traces** from public agent benchmarks. It's
intentionally dumb and intentionally fast — you should be able to beat it.

## What's in the box

- `agent_tool_router/` — the SDK (`Router.from_pretrained`, `route(task, k)`).
- `router/index/` — loaders that normalize public datasets (tau-bench,
  Hermes function-calling-v1, ToolACE, SWE-bench Verified, OSWorld) into a
  unified `Trace` schema.
- `router/eval/` — the evaluation scripts that produced the numbers below.
- `scripts/make_dataset.sh` — rebuild `data/traces.jsonl` from public sources.

`data/` and `models/` are gitignored — generate them locally.

## Numbers (baseline-v0)

Trained on 8 162 task→tool sequences (cross-corpus, after dedup). Test on
2 041 held-out tasks. Tool vocabulary filtered to names appearing ≥ 3 times in
the training set: **265 tools**.

| metric | model | random | ratio |
|---|---:|---:|---:|
| top-1 per-call accuracy | 33.0% | 0.38% | **87.5×** |
| top-3 per-call accuracy | 63.8% | 1.13% | **56.4×** |
| top-5 per-call accuracy | 83.0% | 1.89% | 44.0× |
| top-10 per-call accuracy | 91.5% | 3.77% | 24.3× |

Per-source top-3 (same model, evaluated by source):

| source | n_test tasks | calls evaluated | top-3 acc | ratio |
|---|---:|---:|---:|---:|
| Hermes function-calling-v1 | 218 | 13 | 92.3% | 81.5× |
| ToolACE | 1 792 | 60 | 63.3% | 55.9× |
| tau-bench | 31 | 151 | 61.6% | 54.4× |

### Caveats — read these before quoting the numbers

- **Hermes leaks the tool name into the task text 21.5% of the time.** A row
  like *"Get the camera live feed"* gold-calls `get_camera_live_feed`. The
  model isn't really learning routing on those — it's doing fuzzy substring
  matching. We measured: tau-bench 0%, SWE-bench 0%, ToolACE 2.8%, Hermes
  21.5%. The cross-corpus number above is the headline; the closest-to-clean
  number is **tau-bench's top-3 = 5.0× random** (vocab=23, separate baseline
  in `router/eval/baseline_tfidf.py`).
- **The vocab is filtered.** 95% of the union vocab (~10 000 tool names) only
  appears once or twice. The baseline can't learn anything about those —
  it's evaluated on the 265 tools that actually have training signal.
  Cold-start tool routing is an open problem (see [Roadmap](#roadmap)).
- **Centroid retrieval is the floor, not the ceiling.** This is what a
  TF-IDF model and a bit of arithmetic can do. Anything you build should
  beat it; if it doesn't, the problem is your model, not the dataset.

## Try it

```bash
git clone https://github.com/dalek-ai/agent-tool-router.git
cd agent-tool-router
pip install -e .

# Optional: rebuild the dataset and a fresh model.
bash scripts/make_dataset.sh
python -m agent_tool_router.train --out models/baseline-v0
```

CLI:

```bash
python -m agent_tool_router "Book me a flight to Tokyo and a hotel near Shibuya" -k 5 --scores
```

## The dataset (14K traces)

| source | rows | gold actions? | notes |
|---|---:|:---:|---|
| ToolACE (Team-ACE) | 8 971 | yes | Synthetic function calls; vast vocab. |
| Hermes function-calling-v1 (NousResearch) | 2 180 | yes | OpenAI-format `<tool_call>` blocks. |
| tau-bench (sierra-research) | 1 980 | yes | Real GPT-4o + Sonnet-3.5 trajectories on retail/airline. |
| SWE-bench Verified | 500 | partial | Patch-only, used for completeness. |
| OSWorld | 369 | no | Task definitions only — no realized rollouts. |

All loaders normalize to a single `Trace` schema (see
`router/index/trace_schema.py`). Adding a new source means writing a loader
that yields `Trace` instances; nothing else changes.

## Roadmap

The current baseline answers *"which tools, ranked"* on a closed vocabulary.
The interesting questions are downstream:

1. **Cold-start tool routing.** Given a tool you've never seen, can you route
   to it from its description alone? This is the actual hard problem and where
   most of the dataset (95% singleton long-tail) is currently dead weight.
2. **Sequence routing.** The current model returns a set, not an ordered plan.
   The data has the order; we're just not using it yet.
3. **Cross-source generalization.** Leave-one-source-out evaluation isn't
   wired up yet — we don't actually know if the model is one-router-per-source
   under the hood.
4. **Real traces.** Public benchmarks are great for bootstrapping but skewed
   toward synthetic prompts. Opt-in trace contribution is the long-game moat.

## License

MIT. Be kind.

## Status

Day 3 of a 14-day phase 0. We're still figuring out the right shape of this
thing. Issues / PRs welcome — break things early.
