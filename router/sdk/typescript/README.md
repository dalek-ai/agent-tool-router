# @dalek-ai/router

TypeScript SDK for [agent-tool-router](https://github.com/dalek-ai/agent-tool-router) — picks the right tools for an agent task from a catalog of 18 000.

The default endpoint is a hosted API at `dalek-ai-router-api.hf.space`. No API key, no signup.

## Install

```bash
npm install @dalek-ai/router
```

Requires Node 18+ (uses the built-in `fetch`).

## Use

```ts
import { route } from "@dalek-ai/router";

const result = await route({
  task: "annule ma commande et rembourse-moi",
  k: 3,
});

console.log(result.tools);
// [
//   { name: "get_order_details", score: 0.26 },
//   { name: "get_user_details", score: 0.25 },
//   { name: "return_delivered_order_items", score: 0.24 }
// ]
console.log(result.model);      // "baseline-v1-desc-hybrid-multilingual-next-v1"
console.log(result.latency_ms); // ~200
```

### History-aware routing

Pass the previous tools called to bias the next-tool prediction (uses a Markov-2 history bigram with stupid-backoff under the hood):

```ts
const result = await route({
  task: "rembourse-moi maintenant",
  history: ["get_order_details", "verify_refund_eligibility"],
  k: 3,
});
```

### Reusable client

```ts
import { Router } from "@dalek-ai/router";

const r = new Router(); // defaults to https://dalek-ai-router-api.hf.space
const top3 = await r.route("send slack to my team");

// or point at a self-hosted instance
const self = new Router("https://router.mycompany.internal");
```

### Errors

Failed requests throw `RouterError` with `.status` and `.body`:

```ts
import { RouterError } from "@dalek-ai/router";

try {
  await route({ task: "..." });
} catch (e) {
  if (e instanceof RouterError) console.error(e.status, e.body);
}
```

## Hosted API specs

- URL: `https://dalek-ai-router-api.hf.space`
- Median latency: ~200 ms (shared free CPU on Hugging Face Spaces)
- Default model: `baseline-v1-desc-hybrid-multilingual-next-v1` (FR + EN, Pareto-dominates 3 LOSO benchmarks)
- Rate limits: HF Spaces public quotas (generous, not for production scale)
- Higher limits / private deployments: drop your handle on the [Waitlist discussion](https://github.com/dalek-ai/agent-tool-router/discussions/1)

## Numbers

- 18 671 tools indexed (Hermes function-calling, ToolACE, tau-bench)
- 14 000 agent traces in the training corpus
- 77.4% top-3 next-tool accuracy (Markov-2 backoff on the multilingual encoder)
- 100% top-3 on tau-bench position 2 (228 test calls, zero errors at this horizon)
- 9 ms p50 latency in-process (local MacBook MPS) with the Python SDK

Full benchmark numbers, eval scripts, and the dataset rebuild script:
[github.com/dalek-ai/agent-tool-router](https://github.com/dalek-ai/agent-tool-router)

## License

MIT.
