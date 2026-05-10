"""Gradio Space demo for agent-tool-router.

Routes a free-form task (English or French) to the top-k tools from a
catalog of 18 671 entries pulled from public agent benchmarks
(tau-bench, Hermes function-calling-v1, ToolACE).

Two pretrained models are loaded:
  - default      = baseline-v1-desc-hybrid                (English-first, MiniLM-L6)
  - multilingual = baseline-v1-desc-hybrid-multilingual   (50+ langues, FR-friendly)
"""
from __future__ import annotations

import time

import gradio as gr
from agent_tool_router import Router

print("[boot] loading default hybrid (baseline-v1-desc-hybrid)...")
_t = time.time()
ROUTER_DEFAULT = Router.from_pretrained("baseline-v1-desc-hybrid")
ROUTER_DEFAULT.route("warmup", k=1)
print(f"[boot] default ready in {time.time() - _t:.1f}s")

print("[boot] loading multilingual hybrid (baseline-v1-desc-hybrid-multilingual)...")
_t = time.time()
ROUTER_MULTI = Router.from_pretrained("baseline-v1-desc-hybrid-multilingual")
ROUTER_MULTI.route("warmup", k=1)
print(f"[boot] multilingual ready in {time.time() - _t:.1f}s")


EXAMPLES = [
    ["cancel my pending order and refund the credit", "default", 5],
    ["envoie un message slack à l'équipe data sur le pipeline cassé", "multilingual", 5],
    ["transcribe this voice memo to english text", "default", 5],
    ["traduis ce document en anglais en gardant la mise en page", "multilingual", 5],
    ["book a hotel in Paris for two nights starting tomorrow", "default", 5],
    ["réserve une voiture de location à Lyon pour vendredi", "multilingual", 5],
    ["get the current weather forecast in Berlin", "default", 5],
    ["récupère le statut de mon dernier paiement Stripe", "multilingual", 5],
]


def route(task: str, model_choice: str, k: int):
    if not task or not task.strip():
        return [["—", "type a task above to see the top-k tools", 0.0]]
    router = ROUTER_MULTI if model_choice == "multilingual" else ROUTER_DEFAULT
    started = time.time()
    results = router.route(task.strip(), k=int(k), return_scores=True)
    elapsed_ms = (time.time() - started) * 1000.0
    rows = [[i + 1, r.tool, round(r.score, 4)] for i, r in enumerate(results)]
    rows.append(["—", f"latency: {elapsed_ms:.1f} ms", 0.0])
    return rows


HEADER = """\
# agent-tool-router

Pick the right tools for an agent task. Boring TF-IDF + bi-encoder baseline,
trained on 14 000 traces from public agent benchmarks. Catalog = 18 671 tools.
MIT, open dataset, no API key.

**En français :** ce router open-source choisit les outils à appeler pour une
tâche, parmi un catalogue de 18 000. Le pretrained multilingue sort 54%
top-3 sur 50 tâches en français, sans coût mesurable côté anglais.

[github.com/dalek-ai/agent-tool-router](https://github.com/dalek-ai/agent-tool-router)
· [huggingface.co/dalek-ai](https://huggingface.co/dalek-ai)
"""


with gr.Blocks(title="agent-tool-router demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(HEADER)

    with gr.Row():
        with gr.Column(scale=3):
            task_in = gr.Textbox(
                label="Task / tâche",
                placeholder="cancel my pending order, envoie un message slack, ...",
                lines=3,
            )
        with gr.Column(scale=2):
            model_in = gr.Radio(
                choices=["default", "multilingual"],
                value="default",
                label="Model",
                info=(
                    "default = baseline-v1-desc-hybrid (English-first, MiniLM-L6, ~35 MB).\n"
                    "multilingual = baseline-v1-desc-hybrid-multilingual (50+ langues, FR-friendly, ~80 MB)."
                ),
            )
            k_in = gr.Slider(1, 20, value=5, step=1, label="top-k")

    btn = gr.Button("Route", variant="primary")

    out = gr.Dataframe(
        headers=["rank", "tool", "score"],
        label="Top-k tools (last row = latency)",
        wrap=True,
    )

    btn.click(route, inputs=[task_in, model_in, k_in], outputs=out)
    task_in.submit(route, inputs=[task_in, model_in, k_in], outputs=out)

    gr.Examples(
        examples=EXAMPLES,
        inputs=[task_in, model_in, k_in],
        outputs=out,
        fn=route,
        cache_examples=False,
        label="Examples (EN + FR)",
    )

    gr.Markdown(
        """
---

### How it works

1. The task is encoded into both a TF-IDF sparse vector and a bi-encoder
   dense vector.
2. Each of the 18 671 tools has a centroid in both spaces, built from its
   natural-language description.
3. The score is `0.5 · cos_tfidf + 0.5 · cos_encoder`.
4. The top-k tool names are returned.

The "default" model uses `sentence-transformers/all-MiniLM-L6-v2` (English).
The "multilingual" model uses `paraphrase-multilingual-MiniLM-L12-v2` (50+
languages). Numbers and reproduction in the
[repo README](https://github.com/dalek-ai/agent-tool-router#numbers-baseline-v0).
"""
    )


if __name__ == "__main__":
    demo.launch()
