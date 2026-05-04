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
intentionally dumb and intentionally fast. You should be able to beat it.

## What's in the box

- `agent_tool_router/`: the SDK (`Router.from_pretrained`, `route(task, k)`).
- `router/index/`: loaders that normalize public datasets (tau-bench,
  Hermes function-calling-v1, ToolACE, SWE-bench Verified, OSWorld) into a
  unified `Trace` schema.
- `router/eval/`: the evaluation scripts that produced the numbers below.
- `scripts/make_dataset.sh`: rebuild `data/traces.jsonl` from public sources.

`data/` and `models/` are gitignored. Generate them locally.

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

### Caveats (read these before quoting the numbers)

- **Hermes leaks the tool name into the task text 21.5% of the time.** A row
  like *"Get the camera live feed"* gold-calls `get_camera_live_feed`. The
  model isn't really learning routing on those, it's doing fuzzy substring
  matching. We measured: tau-bench 0%, SWE-bench 0%, ToolACE 2.8%, Hermes
  21.5%. The cross-corpus number above is the headline; the closest-to-clean
  number is **tau-bench's top-3 = 5.0× random** (vocab=23, separate baseline
  in `router/eval/baseline_tfidf.py`).
- **The vocab is filtered.** 95% of the union vocab (~10 000 tool names) only
  appears once or twice. The baseline can't learn anything about those, so
  it's evaluated on the 265 tools that actually have training signal.
  Cold-start tool routing is an open problem (see [Roadmap](#roadmap)).
- **Centroid retrieval is the floor, not the ceiling.** This is what a
  TF-IDF model and a bit of arithmetic can do. Anything you build should
  beat it; if it doesn't, the problem is your model, not the dataset.
- **The pretrained model does not transfer across tool ecosystems via
  names.** Each source brings its own private universe of tool names:
  `cancel_pending_order` (tau-bench retail) doesn't appear in ToolACE;
  `Get Stock Price` (ToolACE) doesn't appear in Hermes. Leave-one-source-out
  vocab overlap is **0.0%–0.1%** across the three sources, so name-based
  routing trained on N-1 sources scores ~0% on the held-out source. For
  your **own** tools, see [`Router.from_examples()`](#use-it-on-your-own-tools).

- **Even after stripping leaky rows, the cross-corpus baseline holds.**
  Filtering out every row where the gold tool name appears verbatim or as
  in-order subtokens within a 4-token window of the task text drops the
  dataset from 14K to 10.4K rows but only moves the cross-corpus headline
  from 56.4× to **30.6× random top-3**. See
  `router/eval/baseline_cross_corpus_clean.py`.

### Cross-source generalization via tool descriptions

The roadmap line "can a model bridge ecosystems via tool *descriptions*
rather than tool *names*?" is now answered. Yes, mostly.

We extracted the natural-language description of every tool we could find
(2.6K from Hermes, 16K from ToolACE, 29 from tau-bench, in
`data/tool_descriptions.jsonl`) and re-ran leave-one-source-out, but
scoring tools by `cosine(task, description)` instead of by training tool
centroids on tool names. Description text only, no name subtokens:

| held out | catalog size | top-1 | top-3 | top-3 vs random |
|---|---:|---:|---:|---:|
| Hermes function-calling-v1 | 1 911 | 41.6% | 73.5% | **468× random** |
| ToolACE | 10 065 | 22.5% | 34.6% | **1 162× random** |
| tau-bench | 23 | 8.7% | 19.8% | 1.5× random |

For comparison, the same setup with **names** scored 0% top-3 across all
three sources. So descriptions transfer, names don't. tau-bench is the
weak case because its 23 tools are domain-specific customer-service flows
that have no analog in the training corpus, but it still beats random.
Source: `router/eval/baseline_loso_descriptions.py`.

A pre-trained sentence encoder (sentence-transformers/all-MiniLM-L6-v2)
gives a different shape of result. Same protocol, descriptions only:

| held out | catalog | TF-IDF top-3 | bi-encoder top-3 | hybrid α=0.5 top-3 | hybrid (best α) top-3 |
|---|---:|---:|---:|---:|---:|
| Hermes | 1 911 | 73.5% | 67.9% | 77.2% | **78.5% (α=0.7)** |
| ToolACE | 10 065 | 34.6% | 58.5% | 61.8% | **62.1% (α=0.4)** |
| tau-bench | 23 | 19.8% | 31.1% | 29.2% | **31.3% (α=0.1)** |

The two backends are complementary: TF-IDF wins when task and description
share lexical surface (Hermes), the bi-encoder wins when the description
paraphrases the task without sharing words (ToolACE, tau-bench). A flat
hybrid `0.5·cos_tfidf + 0.5·cos_encoder` Pareto-improves on Hermes and
ToolACE, with a small regression on tau-bench. Per-source-tuned α improves
all three. Source: `router/eval/baseline_loso_descriptions_hybrid.py`.

The bi-encoder pulls in `sentence-transformers` and torch (~250 Mo of
deps), so we don't ship it in the default install. The TF-IDF path is the
SDK default; the encoder and hybrid backends are available behind an
optional extras: `pip install agent-tool-router[encoder]`.

## Use it on your own tools

If your agent has 5 custom tools (`web_search`, `internal_kb`,
`run_sql`, ...) the pretrained model has never seen them. Build a router
in memory from a small seed list of your own tasks:

```python
from agent_tool_router import Router

r = Router.from_examples([
    ("search the web for recent papers on X",     ["web_search"]),
    ("look up the customer's order history",      ["internal_kb"]),
    ("run a SQL query to count active users",     ["run_sql"]),
    # ... ~10-30 examples per tool is enough to start
])
r.route("find the top sellers from last quarter", k=2)
# ['run_sql', 'internal_kb']
```

A full ~70-line example, with five mock tools and 68 seed examples, is in
[`examples/research_helper/`](examples/research_helper/). Run it:

```bash
python -m examples.research_helper.agent
```

If you already have OpenAI-style function specs (each tool comes with a
short natural-language description) and don't want to seed example tasks,
you can build the same kind of router directly from `(name, description)`
pairs. This is the same scoring rule that gave 73% top-3 cross-source in
our LOSO eval:

```python
r = Router.from_descriptions([
    ("web_search", "Search the web and return the top results."),
    ("run_sql",    "Execute a SQL query against the warehouse and return rows."),
    ("internal_kb","Look up a record in the internal customer knowledge base."),
    # ... ideally 50+ tools; with <50 short descriptions TF-IDF is too thin
])
r.route("count how many customers churned last quarter", k=2)
```

For small tool sets, `from_examples()` works better because example tasks
fit the vectorizer on richer text.

If your tool descriptions paraphrase tasks more than they share vocabulary
with them (a common case for OpenAPI / OpenAI specs), switch to the
encoder or hybrid backend:

```python
r = Router.from_descriptions(specs, backend="hybrid", alpha=0.5)
# requires: pip install agent-tool-router[encoder]
```

`backend="tfidf"` is the default and pulls no extra deps.
`backend="encoder"` runs sentence-transformers/all-MiniLM-L6-v2 and
scores by cosine on its embeddings. `backend="hybrid"` linearly combines
TF-IDF and encoder cosines: in our LOSO eval the hybrid Pareto-dominates
both solo backends with `alpha=0.5` on 2/3 held-out sources (see the
table above).

## Try the pretrained model

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
| OSWorld | 369 | no | Task definitions only, no realized rollouts. |

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
3. **Cross-source generalization.** Names don't transfer (LOSO ≈ 0%);
   descriptions do (LOSO ≈ 35–74% top-3 on the two held-out sources with
   broad catalogs; see Caveats). Next step: make `Router` first-class on
   tool descriptions, not just names. Today the SDK assumes you pass
   `(task, [tool_names])`; tomorrow it should accept
   `(task, [(name, description)])` and pick the path automatically based
   on what you give it.
4. **Real traces.** Public benchmarks are great for bootstrapping but skewed
   toward synthetic prompts. Opt-in trace contribution is the long-game moat.

## License

MIT. Be kind.

## Status

Phase 0, still figuring out the right shape of this thing. Issues / PRs
welcome. Break things early.
