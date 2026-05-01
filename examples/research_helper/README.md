# research_helper — a 70-line dogfood

A tiny example agent built on top of `agent_tool_router`. It does **not** call
an LLM. It picks tools with the router, runs the chosen ones (mocked), and
prints what would happen.

The point isn't the tools. The point is the shape: this is what using the
router in a real agent loop looks like, with **your own** tools and **your
own** seed examples — not the 14K-trace bundled model.

## What's in here

| file | what it is |
| --- | --- |
| `tools.py` | five mock tools: `web_search`, `calculator`, `file_read`, `memory_lookup`, `python_exec`. Each has a name, a description, and a `run()` that prints a mocked result. |
| `seed_examples.py` | 68 `(task, [tools])` pairs — about a dozen per tool plus a handful of multi-tool examples. |
| `agent.py` | the agent. Builds the router via `Router.from_examples(SEED_EXAMPLES)`, then for each task asks for top-k and runs the tools whose cosine score clears a threshold. |

## Run it

```bash
# canned demo (6 tasks)
python -m examples.research_helper.agent

# one task
python -m examples.research_helper.agent "find me the latest paper on RLHF"

# interactive
python -m examples.research_helper.agent -i
```

## What it shows

Top-1 routing is correct on all 6 demo tasks (run it to verify). More
interesting:

- `"what is 19 * 23 + 7"` → top-1 `calculator`, top-2 `file_read`. The
  calculator runs and returns 444. file_read is included in top-2 but its
  cosine score (0.144) is barely above the run-threshold and the tool itself
  declares the input non-applicable. **That's the point of routing top-k +
  threshold + per-tool input validation.**

- `"look up the population of japan online and divide it by 1000"` →
  the router returns `calculator` and `web_search`, both above threshold. A
  real agent would chain them; this dogfood just runs both in parallel and
  prints both outputs. Multi-tool routing falls out of top-k for free.

## Why use `from_examples`

`Router.from_pretrained("baseline-v0")` loads the bundled model trained on
14K public traces — useful when your tool names overlap with the dataset
vocab. But most agents have their own private tools (`web_search`,
`calculator`, ... or `notion_search`, `internal_kb`, ...) that the bundled
model has never seen.

`Router.from_examples([(task, tools), ...])` builds the same kind of
centroid retriever in memory from a tiny seed list. No persistence, no
training script, no joblib files. ~30-50 examples is enough to get a usable
router across 5 tools. More examples → less noise.

## Caveats

- This is a baseline. With 5 tools + 68 examples, top-1 here is ~95%+
  on simple queries. With 50 tools + 200 examples, expect a real-world drop;
  re-run the eval on your own held-out tasks.
- Mock tools ignore the routed task content beyond logging it. The
  calculator does parse numbers out of the task text on a best-effort basis,
  so `"what is 19 * 23 + 7"` actually evaluates.
- The threshold (default `0.10`) is a heuristic, not a tuned value. Tune
  it on your own held-out set.
