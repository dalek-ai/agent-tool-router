/**
 * @dalek-ai/router — TypeScript SDK for the agent-tool-router hosted API.
 * https://github.com/dalek-ai/agent-tool-router
 */

export const DEFAULT_API_URL = "https://dalek-ai-router-api.hf.space";

export interface RouteOptions {
  task: string;
  k?: number;
  history?: string[];
  model?: string;
  apiUrl?: string;
  signal?: AbortSignal;
}

export interface RouteResultTool {
  name: string;
  score: number;
}

export interface RouteResult {
  tools: RouteResultTool[];
  model: string;
  latency_ms: number;
}

export class RouterError extends Error {
  status?: number;
  body?: string;
  constructor(message: string, status?: number, body?: string) {
    super(message);
    this.name = "RouterError";
    this.status = status;
    this.body = body;
  }
}

export async function route(opts: RouteOptions): Promise<RouteResult> {
  if (!opts.task || typeof opts.task !== "string") {
    throw new RouterError("`task` must be a non-empty string");
  }
  const url = (opts.apiUrl ?? DEFAULT_API_URL).replace(/\/$/, "") + "/route";
  const payload: Record<string, unknown> = {
    task: opts.task,
    k: opts.k ?? 3,
  };
  if (opts.history && opts.history.length > 0) payload.history = opts.history;
  if (opts.model) payload.model = opts.model;

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: opts.signal,
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new RouterError(
      `Router API ${res.status} ${res.statusText}`,
      res.status,
      body
    );
  }
  return (await res.json()) as RouteResult;
}

export class Router {
  private readonly apiUrl: string;
  private readonly defaultModel?: string;

  constructor(apiUrl: string = DEFAULT_API_URL, defaultModel?: string) {
    this.apiUrl = apiUrl;
    this.defaultModel = defaultModel;
  }

  async route(
    task: string,
    opts?: { k?: number; history?: string[]; model?: string; signal?: AbortSignal }
  ): Promise<RouteResult> {
    return route({
      task,
      k: opts?.k,
      history: opts?.history,
      model: opts?.model ?? this.defaultModel,
      apiUrl: this.apiUrl,
      signal: opts?.signal,
    });
  }

  async health(signal?: AbortSignal): Promise<{ status: string; models_loaded: number }> {
    const res = await fetch(this.apiUrl.replace(/\/$/, "") + "/health", { signal });
    if (!res.ok) throw new RouterError(`health ${res.status}`, res.status);
    return (await res.json()) as { status: string; models_loaded: number };
  }
}
