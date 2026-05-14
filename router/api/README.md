---
title: Agent Tool Router API
emoji: 🛤️
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: REST API for the open-source Agent Tool Router.
---

# Agent Tool Router — REST API

A minimal FastAPI service wrapping the open-source [Agent Tool Router](https://github.com/dalek-ai/agent-tool-router).

## Quick start

```bash
curl -X POST https://dalek-ai-router-api.hf.space/route \
  -H 'Content-Type: application/json' \
  -d '{"task": "annule ma commande et rembourse-moi", "k": 3}'
```

Returns the top-3 tool names from a catalog of 18 671 indexed tools, with
optional history-aware Markov rerank when `history` is passed.

## Endpoints

| Method | Path      | Purpose                                    |
| ------ | --------- | ------------------------------------------ |
| GET    | `/`       | Service info and example                   |
| GET    | `/health` | Liveness probe                             |
| GET    | `/models` | List loaded models                         |
| POST   | `/route`  | Route a task to top-k tools                |
| GET    | `/docs`   | OpenAPI / Swagger UI                       |

## Request schema

```json
{
  "task": "cancel my order",
  "history": ["get_order_details"],
  "k": 3,
  "model": "baseline-v1-desc-hybrid-multilingual-next-v1"
}
```

`history` and `model` are optional. Default model is
`baseline-v1-desc-hybrid-multilingual-next-v1` — multilingual (EN + FR),
Pareto-dominant on three held-out source benchmarks (Hermes, ToolACE,
tau-bench).

## Self-host

```bash
git clone https://github.com/dalek-ai/agent-tool-router
cd agent-tool-router/router/api
docker build -t router-api .
docker run -p 7860:7860 router-api
```

Or with `pip`:

```bash
pip install "agent-tool-router[encoder]@git+https://github.com/dalek-ai/agent-tool-router.git"
pip install fastapi "uvicorn[standard]"
uvicorn router.api.main:app --host 0.0.0.0 --port 7860
```

## Numbers

Held-out next-tool retrieval (n=2094, multilingual-next-v1 with Markov-2 backoff):

| Metric                          | Value  |
| ------------------------------- | -----: |
| top-1                           | 56.9%  |
| top-3                           | 76.2%  |
| tau-bench position 2, top-3     |  100%  |
| tau-bench position ≥ 3, top-3   | 92.9%  |

Latency on a 1-vCPU container: median ~15 ms per request.
