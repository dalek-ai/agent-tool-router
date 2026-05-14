FROM python:3.11-slim

WORKDIR /code

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

# HuggingFace Spaces caches under /code; honor that for the model snapshot.
ENV HF_HOME=/code/.hf_cache
ENV TRANSFORMERS_CACHE=/code/.hf_cache
ENV ATR_DEFAULT_MODEL=baseline-v1-desc-hybrid-multilingual-next-v1
ENV ATR_PRELOAD_MODELS=baseline-v1-desc-hybrid-multilingual-next-v1
ENV PORT=7860

# Install the SDK from GitHub until PyPI publication (GREG-ACTION-020).
RUN pip install --no-cache-dir \
    "agent-tool-router[encoder] @ git+https://github.com/dalek-ai/agent-tool-router.git@main" \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.27"

# Pull the default model at build time so the cold start is fast.
RUN python -c "from agent_tool_router.router import Router; Router.from_pretrained('${ATR_DEFAULT_MODEL}')"

# Copy the API entrypoint into a package-like path so uvicorn can import it.
COPY main.py /code/router/api/main.py
RUN touch /code/router/__init__.py /code/router/api/__init__.py

EXPOSE 7860

CMD ["uvicorn", "router.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
