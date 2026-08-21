# Backend adapters

A CFR-Atlas adapter connects virtual K/V pages to one concrete model runtime. The core can make memory residency explicit; only the adapter can establish that a regenerated page is the same page the runtime would otherwise have retained in its full K/V cache.

## Responsibility split

| CFR-Atlas provides | A backend adapter must provide |
|---|---|
| Page identity, ranges, scratch storage and cache residency | Access to token history and absolute positions |
| Folded exact-attention reduction | Model-specific K/V replay for `(layer, K/V head, token range)` |
| MHA/MQA/GQA, dtype and position policy types | Correct topology, RoPE/ALiBi and storage-rounding semantics |
| Page-level conformance utilities | Comparison against the runtime's conventional stored-KV path |
| Output/logit validation harness | A model-head projection for production logit validation |

The core does not own weights, tokenization, transformer execution, graph compilation or optimized matrix kernels.

## Required contract

Implement `KvRegenerator` and fill the requested row-major page buffers:

```rust
use cfr_atlas::prelude::*;
use std::ops::Range;

struct MyBackend;

impl KvRegenerator for MyBackend {
    fn regenerate_page(
        &self,
        key: PageKey,
        token_range: Range<usize>,
        head_dim: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<()> {
        // Replay the normal backend path for `key.layer`, `key.head` and
        // `token_range`; write K[token][dim] and V[token][dim].
        let _ = (key, token_range, head_dim, k_out, v_out);
        Ok(())
    }
}
```

The adapter must treat `PageKey.head` as a **K/V head**. For MHA, MQA and GQA, derive that head through `AttentionTopology` before requesting or validating a page. A final page may be shorter than the configured page size and must still replay exactly.

## Enablement gate: page conformance

Do not enable CFR for a model family after only compiling an adapter. First demonstrate that the backend's stored and regenerated K/V values agree under the exact production policy.

1. Run the normal stored-KV path for a short, deterministic context.
2. Request the same range through `KvRegenerator`.
3. Compare K and V with `compare_regenerated_page` or `assert_regenerated_page`.
4. Repeat across layers, K/V heads, initial/middle/final pages, MHA/MQA/GQA topology, position policy and storage dtype.
5. Treat any mismatch as an adapter bug or an unsupported mode, not as a cache-policy choice.

The in-repository `cfr-atlas-backend-ref` crate is a deterministic conformance fixture. It is not a language model or a performance proxy for a production backend.

## Preserve these semantics

A typical real backend must replay more than token ids. Its adapter should make the following inputs explicit and testable:

| Concern | Required behavior |
|---|---|
| Token history | Replay the same token ids and absolute positions used by the baseline |
| Layer path | Reproduce the forward computation needed to reach the requested layer |
| Head mapping | Preserve MHA/MQA/GQA query-to-K/V mapping |
| Position policy | Apply RoPE, ALiBi or equivalent logic at the same point as the baseline |
| Storage policy | Match `f32`, `bf16`, `f16` or other rounding behavior before attention consumes rows |
| Causality | Return only the requested half-open causal range in its original order |
| Numeric policy | State tolerance and accumulation assumptions when bit-exact ordering is not possible |

## From conformance to long-context validation

After page-level conformance passes, validate actual attention behavior before enabling a serving path.

1. Build `PromptCase` inputs that represent the model's relevant workloads.
2. Use `validate_decode_step` for targeted output/logit mismatches.
3. Use `validate_decode_loop` across selected layers, heads, positions and long-context shapes.
4. Record `MemoryTelemetry` with model identity, context length, page size, hot-cache budget, dtype and hardware information.
5. Keep a regression corpus for every model/runtime combination that is released.

For a real model, provide an actual `LogitProjector`. `DeterministicLogitProjector` exists only for the reference fixture.

## Optional runtime controls

After correctness is established, adapters may select `DotProductKernel`, tune page size with `PageSizeTuner`, reuse two cold-page buffers with `DoubleBufferedPipeline`, bound independent work with `ThreadPoolExecutor`, set per-layer cache budgets, or implement `TelemetryResidencyPolicy`. These controls may affect compute placement, locality, latency and resident bytes; they may not alter K/V content or output semantics.

## Review checklist

Before shipping an adapter, confirm all of the following:

- stored-KV versus regenerated-KV conformance passes under every supported topology/dtype/position configuration;
- exactness or stated numerical tolerance is checked at attention output and, where possible, model logits;
- resident-memory accounting is reported separately from process RSS;
- unsupported execution modes fail closed instead of silently replaying a different policy;
- cache admission and eviction are tested as performance controls only;
- published benchmark reports name the model backend, runtime configuration, workload and validation status.
