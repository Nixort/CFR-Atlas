# CFR-Atlas documentation

CFR-Atlas is an exact-attention memory-virtualization layer. This directory separates user-facing integration guidance from internal implementation details and gives each claim a clear validation boundary.

## Choose a reading path

| If you need to… | Read | Outcome |
|---|---|---|
| Decide whether CFR fits the runtime | [`../README.md`](../README.md) and [`CLAIMS.md`](CLAIMS.md) | Understand the exactness contract, non-goals and production boundary |
| Understand page execution and resident memory | [`ARCHITECTURE.md`](ARCHITECTURE.md) | Map the runtime, cache, scratch lifecycle and folded attention data flow |
| Write a backend adapter | [`ADAPTERS.md`](ADAPTERS.md) | Implement exact K/V replay and complete conformance before enabling CFR |
| Connect to public Ollama APIs | [`OLLAMA.md`](OLLAMA.md) | Discover models and use supported wrappers while preserving the exact-K/V fail-closed boundary |
| Review numerical behavior | [`MATH.md`](MATH.md) | Follow online-softmax folding, exactness assumptions and memory accounting |
| Run or interpret measurements | [`BENCHMARKS.md`](BENCHMARKS.md) | Reproduce the reference workload and avoid unsupported throughput claims |
| Review release readiness | [`STABILIZATION.md`](STABILIZATION.md) | Check API, MSRV, schema, supply-chain and hardening posture |
| Evaluate `no_std` work | [`NO_STD.md`](NO_STD.md) | Understand the current blockers and what is intentionally out of scope |
| Track planned work | [`ROADMAP.md`](ROADMAP.md) | Distinguish completed foundation work from integration and measurement next steps |

## Documentation conventions

The guides use the following vocabulary consistently:

- **baseline** means the conventional path that stores K/V rows for the same model computation;
- **exact** means regenerated K/V rows, head mapping, position policy and rounding match that baseline within the stated validation policy;
- **resident memory** means K/V pages retained by CFR-Atlas, not whole-process RSS;
- **reference backend** means the deterministic in-repository adapter, not a production language model;
- **benchmark** means a measured, reproducible workload with stated environment and scope, not an end-to-end LLM performance guarantee.

## Claim discipline

CFR-Atlas makes conditional claims. Do not promote a deterministic reference-backend result to a real-model guarantee without running adapter conformance and long-context validation for that model family. The exact conditions and falsification criteria are maintained in [`CLAIMS.md`](CLAIMS.md).
