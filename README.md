# agent-tool-router

> Pick the right tools for an agent task. Boring baseline. Open dataset.

```python
from agent_tool_router import Router

# First call downloads ~6 MB from huggingface.co/dalek-ai and caches it.
r = Router.from_pretrained("baseline-v1-desc")
r.route("cancel my pending order and refund the credit", k=3)
# ['refundOrder', 'modify_pending_order_items', 'cancel_pending_order']
```

Install: `pip install agent-tool-router` (Python SDK) or `npm install @dalek-ai/router` (TypeScript SDK, wraps the hosted API). No GPU, no torch, no API key.

**En français :** ce router open-source choisit les outils à appeler pour une tâche, parmi un catalogue de 18 000. Le pretrained multilingue sort 54% top-3 sur un panel de 50 tâches en français (`baseline-v1-desc-hybrid-multilingual`), sans coût mesurable côté anglais. Tout est téléchargeable depuis [huggingface.co/dalek-ai](https://huggingface.co/dalek-ai), licence MIT.

## Try it without installing

A hosted instance runs the `baseline-v1-desc-hybrid-multilingual-next-v1` model at
[dalek-ai-router-api.hf.space](https://dalek-ai-router-api.hf.space). One curl, no
signup, no API key:

```bash
curl -X POST https://dalek-ai-router-api.hf.space/route \
  -H 'Content-Type: application/json' \
  -d '{"task": "annule ma commande et rembourse-moi", "k": 3}'
```

Returns top-3 tools + scores + latency. Interactive Swagger UI at
[/docs](https://dalek-ai-router-api.hf.space/docs). Median latency ~200 ms on a
shared free CPU (vs ~9 ms locally on CPU). Free tier, rate-limited only by HF
Spaces quotas. Code: [`router/api/`](router/api/).

From a Node / TypeScript project:

```ts
import { route } from "@dalek-ai/router";

const result = await route({ task: "annule ma commande", k: 3 });
console.log(result.tools.map(t => t.name));
```

`npm install @dalek-ai/router` — zero runtime deps, uses built-in `fetch` (Node 18+).
Source: [`router/sdk/typescript/`](router/sdk/typescript/).

## Waitlist & feedback

If you build agents and want the API beyond the public demo (private models,
higher rate limits, eval datasets), drop your handle in the
[Waitlist discussion](https://github.com/dalek-ai/agent-tool-router/discussions).
Bug reports and feature requests also go there.

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

### Next-tool prediction (history-aware)

The retrieval router scores tools against the user query. That works when
the query describes the tool (`"cancel my order" → cancelOrder`). It
breaks when the agent is mid-trajectory and the *next* tool depends on
what was already called.

We mined 10 480 `(query, history, next_tool)` triplets from the 7 184
multi-turn traces in the dataset (ToolACE + Hermes + tau-bench, after
collapsing consecutive duplicates) and ran a Markov-1 rerank on top of
the shipped retrieval (top-K candidates, rerank by
`α · retrieval + (1-α) · P(next | last_history_tool)`, 80/20 split by
trace_id, add-one smoothing, Markov-1 fit on train only).

| Setup                                         | top-1 | top-3 | top-5 | top-10 |
|---                                            |---:|---:|---:|---:|
| Retrieval-only                                | 13.8% | 32.7% | 38.8% | 45.4% |
| Markov-1 rerank top-50 (α=0.4)                | 34.6% | 48.0% | 50.5% | 53.6% |
| **Markov-1 rerank top-200 (α=0.1)** ⬅ default | **39.0%** | **54.9%** | **57.7%** | **60.6%** |

That is **+22.2pp top-3** over retrieval, and **+6.7pp** over the top-50
rerank baseline, from widening the retrieval bucket without any new
training. Stratified by position, the gap is largest deep in the
trajectory: on Hermes at t≥3 the rerank pulls top-3 from ~31% to 100%;
on tau-bench at t=1 it goes 7.0% → 57.7%.

Retrieval recall@K is the mechanical ceiling on any rerank that lives on
top-K candidates: recall@50 = 58.9%, recall@200 = 69.6%. Markov-1 reaches
~99% of that ceiling at both bucket sizes — the rerank is essentially
optimal, the bottleneck is whether the gold tool is in the candidate set
at all.

Reproduce:
```
python router/eval/build_next_tool_dataset.py
python router/eval/build_next_tool_cache.py
python router/eval/eval_next_tool_markov.py     # K=50 sweep
python router/eval/eval_next_tool_widen.py      # K=50/100/150/200 sweep
```

A small learned MLP rerank (concat of query / prev-tool / candidate
MiniLM embeddings + retrieval score, 1 hidden 128, trained on 5K
positives) was also tested in `router/eval/train_next_tool_mlp.py` and
`eval_next_tool_mlp.py`. On top-50, it lifts top-3 from retrieval 32.7%
to 40.3% (+7.6pp), but loses -7.7pp versus Markov-1 — the counts-based
transition prior is hard to outscore with dense features on this much
training data. To actually move past the retrieval-recall ceiling, train
the retriever directly on the next-tool objective rather than stacking
more reranks on top-K.

**That direct-fine-tune is what
[`baseline-v1-desc-hybrid-next-v1`](https://huggingface.co/dalek-ai/baseline-v1-desc-hybrid-next-v1)
does.** Fine-tuning MiniLM-L6 on 8 386 (task, gold_description,
hard_negative) triples reshapes the retrieval space itself: recall@200
on the same held-out triplets jumps from 69.6% to 93.1%, and the
Markov-1 top-3 rerank K=200 jumps from 54.9% to **75.5%** (+20.6pp).
The fine-tuned encoder also Pareto-dominates the default English-only
encoder on the full LOSO refit benchmark (Hermes +1.3pp, ToolACE +6.1pp,
tau-bench +27.7pp top-3) — supervised next-tool signal generalizes to
single-task routing too. On the parallel EN/FR n=50 panel, the
fine-tune does **not** cost any French (26% → 28% top-3, within noise)
and **adds +4pp English top-3** (82% → 86%) over the default hybrid,
so it is a free upgrade for English-or-mixed catalogs; the multilingual
model still owns French (54% top-3). Reproduce:
`python router/eval/finetune_retriever_next_tool.py` (re-creates the
encoder) + `python -m agent_tool_router.train_descriptions --backend
hybrid --alpha 0.5 --encoder-model models/_finetune/minilm-next-v1
--out models/baseline-v1-desc-hybrid-next-v1`.

The rerank ships with `baseline-v1-desc-hybrid` (≥ 0.3.0). Pass the
tool names already called in the trace as `history=` and the top-200
candidates are reranked with the Markov-1 prior at α=0.1 (the sweep-best
on the held-out test):

```python
r = Router.from_pretrained("baseline-v1-desc-hybrid")

# Without history: pure retrieval.
r.route("I want to add a checked bag to my reservation", k=3)
# ['update_reservation_baggages', 'completeReservation', 'book_reservation']

# With history: the prior pulls in tools that usually follow the last one.
r.route(
    "I want to add a checked bag to my reservation",
    k=3,
    history=["update_reservation_flights"],
)
# ['update_reservation_baggages', 'update_reservation_passengers', 'cancel_reservation']
```

Override the mix with `markov_alpha=0.0` (Markov-only) or `1.0`
(retrieval-only). On `baseline-v1-desc-hybrid` the table adds ~21 KB to
the download; it's a no-op when `history` is omitted.

**History bigram (Markov-2, ≥ 0.4.0)** — passing two or more previous
tools triggers stupid backoff to a `(prev2, prev1) → next` bigram table
shipped in the same model dir (`markov2_counts.npz` + `markov2_keys.npy`,
~25 KB), falling back to Markov-1 when the bigram is unseen. On the
fine-tuned `baseline-v1-desc-hybrid-next-v1`, the bigram lifts held-out
next-tool **top-1 from 52.9% to 57.3% (+4.4pp)** and **top-3 from 75.5%
to 77.4% (+1.9pp)**, with the biggest single-bucket gain on tau-bench
t≥3 (long-horizon agents): 84.8% → 87.9% top-3. On the multilingual
fine-tune the gain is larger: top-1 51.7% → 56.9% (**+5.2pp**), top-3
73.7% → 76.2% (+2.5pp), and tau-bench t=2 reaches 100% top-3. Reproduce
with `python router/eval/eval_next_tool_markov2.py`.

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

## On HuggingFace

Everything is mirrored on [huggingface.co/dalek-ai](https://huggingface.co/dalek-ai):

| Surface | Link | What it's for |
|---|---|---|
| **Space (live demo)** | [agent-tool-router-demo](https://huggingface.co/spaces/dalek-ai/agent-tool-router-demo) | One-click gradio app. Type a task in EN or FR, get the top-3 tools out of 18 000. No install. |
| **Models** (6) | [dalek-ai](https://huggingface.co/dalek-ai) | Pretrained routers downloadable via `Router.from_pretrained(...)`. See [models](#pretrained-models) below. |
| **Dataset** | [agent-tool-router-eval-fr](https://huggingface.co/datasets/dalek-ai/agent-tool-router-eval-fr) | 50 parallel EN/FR evaluation queries used to measure the multilingual gap. MIT. |

### Pretrained models

`Router.from_pretrained("<name>")` downloads from huggingface.co/dalek-ai on
first call and caches locally. Six pretrained models are published:

- **`baseline-v0`** — 265 tool names that appear ≥ 3 times in the training
  corpus, centroids built from task TF-IDF. The smallest model, no extra
  dependencies, no long-tail coverage. ~100 MB.
- **`baseline-v1-desc`** — 18 671-tool long-tail catalog, centroids built
  from each tool's *description*. TF-IDF only, ~6 MB, no extra dependencies.
- **`baseline-v1-desc-hybrid`** — same catalog, hybrid TF-IDF + bi-encoder
  centroids scored as `0.5 * cos_tfidf + 0.5 * cos_encoder`. ~35 MB.
  Requires `pip install agent-tool-router[encoder]` at runtime
  (sentence-transformers + torch).
- **`baseline-v1-desc-hybrid-multilingual`** — same hybrid pipeline, but the
  encoder is `paraphrase-multilingual-MiniLM-L12-v2` (50+ languages). On a
  parallel EN/FR probe (n=50 hand-written queries) routed through the shipped
  model, FR top-3 jumps from 26% (default hybrid) to 54% with EN top-3 flat
  at 82%. On the full English LOSO refit benchmark the multilingual encoder
  trails the default by ~3.9pp weighted overall, so use the default if all
  your queries are English. ~80 MB.
- **`baseline-v1-desc-hybrid-next-v1`** — same hybrid pipeline, but the
  encoder has been fine-tuned on 8 386 next-tool prediction triplets with a
  contrastive (task, gold_description, hard_negative) loss. On held-out
  next-tool prediction (n=2 094), Markov-1 top-3 jumps from 54.9% (default
  hybrid, top-200 rerank) to **75.5%** (+20.6pp), and recall@200 from
  69.6% to 93.1%. On the full English LOSO refit benchmark (n=30 425),
  the fine-tuned encoder Pareto-dominates: Hermes +1.3pp, ToolACE +6.1pp,
  tau-bench +27.7pp top-3. On the parallel EN/FR n=50 panel, no French
  degradation (26%→28% top-3, within noise) and +4pp English top-3
  (82%→86%) over the default hybrid. ~30 MB. Encoder weights live
  separately at
  [`dalek-ai/minilm-next-v1`](https://huggingface.co/dalek-ai/minilm-next-v1).
- **`baseline-v1-desc-hybrid-multilingual-next-v1`** — same fine-tune
  recipe applied to the multilingual L12 encoder. Pareto-dominates the
  plain multilingual on every LOSO refit source: Hermes +7.3pp, ToolACE
  +4.3pp, **tau-bench +33.9pp** top-3. Versus the EN-only next-v1: gives
  up ~3pp on Hermes/ToolACE in exchange for **+4.7pp tau-bench** and
  multilingual coverage. FR/EN n=50: **54% FR top-3 preserved** (no drift
  from the EN-only training triples), +2pp EN top-3 over plain
  multilingual. ~33 MB. Encoder weights at
  [`dalek-ai/multilingual-next-v1`](https://huggingface.co/dalek-ai/multilingual-next-v1).
  Use this model for mixed FR/EN catalogs that also benefit from
  history-aware rerank.

```python
from agent_tool_router import Router
r = Router.from_pretrained("baseline-v1-desc-hybrid")
r.route("cancel my order and refund the credit", k=3)
# -> ['refundOrder', 'cancel_order', 'cancel_pending_order']
```

The encoder model is lazy-loaded on the first `route()` call, so import
cost is paid only when actually used.

### Rebuild from source

If you prefer to retrain locally instead of downloading:

```bash
git clone https://github.com/dalek-ai/agent-tool-router.git
cd agent-tool-router
pip install -e .

bash scripts/make_dataset.sh
python -m agent_tool_router.train --out models/baseline-v0
python -m agent_tool_router.train_descriptions --out models/baseline-v1-desc
python -m agent_tool_router.train_descriptions --out models/baseline-v1-desc-hybrid \
    --backend hybrid --alpha 0.5  # requires the [encoder] extras
```

Local `models/<name>/` directories take precedence over HuggingFace lookups.

Per-call top-3 accuracy of `baseline-v1-desc` against the full 18 671-tool
catalog, on 30 425 calls drawn from the corpus, by backend (random baseline
= 3/V = 0.016%):

| source | n calls | tfidf | encoder | hybrid α=0.5 |
|---|---:|---:|---:|---:|
| Hermes function-calling-v1 | 4 376 | 74.3% | 60.7% | **74.9%** |
| ToolACE | 17 169 | 52.4% | 54.8% | **62.8%** |
| tau-bench | 8 880 | 3.2% | 6.1% | **9.9%** |
| overall | 30 425 | 41.2% | 41.4% | **49.1%** |

The two backends look tied on overall top-3 (41.2% vs 41.4%) but they get
different things right: TF-IDF wins on Hermes (lexical surface overlap),
the bi-encoder wins on ToolACE and tau-bench (semantic paraphrase). Their
linear combination Pareto-dominates both on every source and every k, +7.9pp
overall. The encoder backend is opt-in behind `pip install agent-tool-router[encoder]`
(adds torch + sentence-transformers, ~250 MB).

Read the tau-bench row carefully even at 9.9% hybrid. The same 23
customer-service tools score 19.8% top-3 against a restricted catalog (LOSO
eval, see below) and only 9.9% against the full v1-desc catalog because the
18 000 ToolACE and Hermes confounders win against domain-specific descriptions.
The takeaway: **`baseline-v1-desc` is a discoverability layer for long-tail
public tools, not a substitute for routing on your own narrow catalog**.
For domain-specific tool sets, use `Router.from_descriptions(your_own)`.

Reproduce: `python -m router.eval.eval_baseline_v1_desc` (tfidf shipped
model), `python -m router.eval.eval_v1_desc_encoder [--hybrid]` (encoder /
hybrid, rebuilt from `data/tool_descriptions.jsonl`).

The same eval, but with the TF-IDF half **refit per held-out source** on
N-1 sources only (encoder is pretrained, source-agnostic). This is the
realistic cross-source number: how much does the shipped pipeline lose on
a brand-new source it has never seen at training time?

| held out | n calls | tfidf | encoder | hybrid α=0.5 |
|---|---:|---:|---:|---:|
| Hermes function-calling-v1 | 4 376 | 70.3% | 60.7% | **72.3%** |
| ToolACE | 17 169 | 36.5% | 54.8% | **58.7%** |
| tau-bench | 8 880 | 10.1% | 6.1% | **11.1%** |

Hybrid Pareto-dominates both solo backends on all three held-out sources.
ToolACE TF-IDF drops sharply (52.4% → 36.5%, -15.9pp) when the vocabulary
has not seen ToolACE descriptions; the encoder catches most of the fall.
The shipped hybrid, by contrast, only loses 2.6pp on Hermes and 4.1pp on
ToolACE vs the in-distribution number, and matches on tau-bench. Reproduce:
`python -m router.eval.eval_v1_desc_loso_hybrid`.

A bigger encoder (BAAI/bge-small-en-v1.5, 33M params, ~38 MB centroids)
was tested as an alternative to MiniLM-L6 (22M params, ~29 MB). LOSO refit
top-3: Hermes 72.3% → 75.0% (+2.7pp), tau-bench 11.1% → 14.3% (+3.2pp),
ToolACE 58.7% → 54.2% (-4.5pp). Weighted overall by n_calls drops 1.3pp.
MiniLM-L6 stays the default; pass `--encoder-model BAAI/bge-small-en-v1.5`
to the train script if Hermes/tau-bench is your weight class.

A multilingual encoder (`paraphrase-multilingual-MiniLM-L12-v2`, 117M
params) was also tested. On a 50-query parallel EN/FR probe routed through
the shipped pretrained, top-3 accuracy goes from 82%/26% (default hybrid,
EN/FR) to 82%/54% — same EN coverage, +28pp on FR. On the full English
LOSO refit, the multilingual encoder costs Hermes -8.8pp, ToolACE -3.9pp
and tau-bench -1.5pp on top-3 (weighted overall -3.9pp). Shipped as
`baseline-v1-desc-hybrid-multilingual` for users whose queries are not all
in English. Reproduce: `python -m router.eval.eval_fr_pretrained` (shipped
models on the EN/FR probe) and
`python -m router.eval.eval_v1_desc_loso_hybrid --encoder-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

Per-task min-max / z-score / rank normalization of `cos_tfidf` and
`cos_enc` before the linear combo was also tested (`router/eval/eval_v1_desc_loso_calibration.py`).
None of them pareto-dominate the unnormalized baseline at α=0.5: minmax
and zscore each gain a couple of pp on Hermes and tau-bench but lose
7-11pp on ToolACE, because ToolACE is 86% of the catalog and per-task
rescaling amplifies same-source distractors when ToolACE is held out.
The shipped pipeline keeps the unnormalized linear combo.

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
2. **Sequence routing.** History-aware reranks shipped in
   `baseline-v1-desc-hybrid` (Markov-1 ≥ 0.3.0, **Markov-2 stupid
   backoff ≥ 0.4.0**): top-3 next-tool accuracy 32.7% → 54.9% (Markov-1)
   → 55.1% (Markov-2) on a held-out test set (n=2094), ~99% of the
   recall@200 ceiling. Past 54.9% requires improving the retriever
   itself — training a retriever directly on the next-tool objective
   (`baseline-v1-desc-hybrid-next-v1`) lifts that ceiling to 75.5%
   top-3, and Markov-2 takes it to **77.4% (+1.9pp over Markov-1 on
   next-v1, +5.2pp top-1 on multilingual-next-v1)**. A learned MLP rerank
   was tested and archived (loses to Markov-1 by ~8pp top-3 on top-50;
   counts dominate dense features at this data scale).
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
