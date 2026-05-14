"""
Agent Tool Router — REST API.

A minimal FastAPI wrapper around the open-source `agent_tool_router.Router`,
designed to be deployed on a HuggingFace Spaces Docker template. One model
is loaded into memory at startup; routing is a single sparse-times-dense
matmul plus an optional Markov rerank, so request latency stays in the
~10-50ms band on free-tier CPU.

Run locally:
    pip install agent-tool-router[encoder]
    pip install fastapi uvicorn
    uvicorn router.api.main:app --reload

Environment variables:
    ATR_DEFAULT_MODEL     short name of the model returned when `model` is
                          omitted from the request. Defaults to
                          `baseline-v1-desc-hybrid-multilingual-next-v1`.
    ATR_PRELOAD_MODELS    comma-separated list of short names to eager-load
                          at startup. Defaults to ATR_DEFAULT_MODEL only.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent_tool_router.router import Router

log = logging.getLogger("router.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

DEFAULT_MODEL = os.getenv("ATR_DEFAULT_MODEL", "baseline-v1-desc-hybrid-multilingual-next-v1")
PRELOAD_MODELS = [
    m.strip() for m in os.getenv("ATR_PRELOAD_MODELS", DEFAULT_MODEL).split(",") if m.strip()
]

API_VERSION = "0.4.0"

app = FastAPI(
    title="Agent Tool Router API",
    description=(
        "Pick the right tools for an agent task. Open-source router trained on "
        "14K public agent traces. Top-3 = 77.4% on next-tool retrieval (n=2094 "
        "held-out). github.com/dalek-ai/agent-tool-router"
    ),
    version=API_VERSION,
    docs_url="/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

ROUTERS: dict[str, Router] = {}


@app.on_event("startup")
def load_models() -> None:
    for name in PRELOAD_MODELS:
        log.info("Loading model %s ...", name)
        t0 = time.time()
        ROUTERS[name] = Router.from_pretrained(name)
        log.info("  %s loaded in %.1fs", name, time.time() - t0)


class RouteRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=2000, description="The user task to route.")
    history: Optional[list[str]] = Field(
        default=None,
        description="Previously called tool names in this trace, oldest first. Enables Markov rerank.",
    )
    k: int = Field(default=3, ge=1, le=50, description="Number of tools to return.")
    model: Optional[str] = Field(
        default=None,
        description=f"Pretrained model name. Defaults to '{DEFAULT_MODEL}'.",
    )


class RoutedTool(BaseModel):
    name: str
    score: float


class RouteResponse(BaseModel):
    tools: list[RoutedTool]
    model: str
    latency_ms: float


@app.get("/")
def index() -> dict:
    return {
        "service": "agent-tool-router",
        "version": API_VERSION,
        "default_model": DEFAULT_MODEL,
        "loaded_models": list(ROUTERS.keys()),
        "endpoints": {
            "POST /route": "Route a task to top-k tools",
            "GET /models": "List loaded models",
            "GET /health": "Liveness probe",
            "GET /docs": "OpenAPI / Swagger UI",
        },
        "links": {
            "github": "https://github.com/dalek-ai/agent-tool-router",
            "huggingface": "https://huggingface.co/dalek-ai",
        },
        "example": {
            "curl": (
                "curl -X POST $URL/route -H 'Content-Type: application/json' "
                "-d '{\"task\": \"annule ma commande et rembourse-moi\", \"k\": 3}'"
            )
        },
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": len(ROUTERS)}


@app.get("/models")
def models() -> dict:
    return {"loaded": list(ROUTERS.keys()), "default": DEFAULT_MODEL}


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest) -> RouteResponse:
    model_name = req.model or DEFAULT_MODEL
    router = ROUTERS.get(model_name)
    if router is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not loaded. Available: {list(ROUTERS.keys())}",
        )
    t0 = time.time()
    results = router.route(
        req.task,
        k=req.k,
        return_scores=True,
        history=req.history,
    )
    latency_ms = (time.time() - t0) * 1000.0
    return RouteResponse(
        tools=[RoutedTool(name=r.tool, score=float(r.score)) for r in results],
        model=model_name,
        latency_ms=round(latency_ms, 2),
    )
