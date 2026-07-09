<!--
Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.

License: MIT
You can find the license file in the project root.

CFR-Atlas
The documentation was written for CFR-Atlas.
9 july 2026

Backend adapter guide for CFR-Atlas Phase 2.
-->

# Backend Adapters

Phase 2 defines the production boundary for crates named
`cfr-atlas-backend-*`. The first implementation is
`crates/cfr-atlas-backend-ref`, a deterministic reference backend used for
conformance tests and examples.

## Boundary

A backend adapter owns model-specific replay:

- token ledger access;
- layer and head validation;
- MHA, MQA or GQA mapping;
- RoPE, ALiBi or backend-specific positional behavior;
- dtype storage policy for `f32`, `bf16` or `f16` paths;
- comparison against the backend's classic stored-KV path.

The core crate still owns only virtual pages, hot-cache residency and folded
attention. It does not own model weights, tokenizer state or matmul kernels.

## Required contract

A backend adapter implements:

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
        let _ = (key, token_range, head_dim, k_out, v_out);
        Ok(())
    }
}
```

The output layout is always row-major:

```text
K[token][dim]
V[token][dim]
```

## Phase 2 components

| Component | API | Purpose |
|---|---|---|
| Token ledger | `TokenLedger`, `TokenRecord` | Replay token ids and absolute positions |
| Head mapping | `AttentionTopology` | Map query heads to K/V heads for MHA, MQA and GQA |
| Position policy | `PositionEncoding` | Preserve RoPE or ALiBi behavior at adapter boundary |
| Dtype policy | `DTypePolicy` | Deterministic `f32`, `bf16` or `f16` storage rounding |
| Conformance | `compare_regenerated_page` | Compare stored K/V with regenerated K/V |
| Reference adapter | `cfr-atlas-backend-ref` | Small deterministic adapter crate |

## Reference backend

The reference backend is not a real language model. It is a deterministic adapter
that exercises the same integration seams a real backend must implement.

Run it with:

```sh
cargo test --workspace --release
cargo run --release --example reference_backend
```

Expected behavior:

- GQA query heads map to the correct K/V head;
- RoPE modifies regenerated key rows deterministically;
- dtype policy rounds K/V rows deterministically;
- stored K/V and regenerated K/V pass conformance with `max_abs_diff = 0`;
- CFR folded attention consumes the adapter through `KvRegenerator`.

## Real backend checklist

A real adapter should provide one conformance test per model family:

1. Run the normal stored-KV path for a short context.
2. Request the same page through `KvRegenerator`.
3. Compare K and V with `assert_regenerated_page`.
4. Repeat across layers, heads, final partial pages and positional settings.
5. Only then run CFR folded attention against full-KV baseline logits.

## Phase 3 integration hooks

Backend crates can optionally use the performance helpers from the core crate:

- `DotProductKernel` for selecting the folded-attention dot-product boundary;
- `PageSizeTuner` for choosing page tokens from cache and scratch constraints;
- `DoubleBufferedPipeline` for reusing two cold-page buffers during regeneration;
- `ThreadPoolExecutor` for bounded batches of independent cold-page work;
- `TelemetryResidencyPolicy` for cache-counter-aware admissions;
- `HotCache::set_layer_budget` through `CfrAtlas::set_layer_hot_cache_bytes` for
  per-layer residency caps.

These hooks remain exactness-neutral. They affect scheduling, locality and
resident memory only.

## Phase 4 validation hooks

After stored-KV conformance passes, adapters should run the logit-level
validation harness before enabling CFR mode for a model family:

1. Build a `PromptCase` or reuse one from `regression_corpus`.
2. Convert it to a `TokenLedger` for the backend adapter.
3. Provide a model-head implementation of `LogitProjector`.
4. Run `validate_decode_step` for targeted debug checks.
5. Run `validate_decode_loop` across sampled layers, GQA query heads and prompt
   positions.
6. Record `MemoryTelemetry` together with model name, context length, page size
   and hot-cache budget.

The in-repository example uses `DeterministicLogitProjector` because the
reference backend is not a real model. Production adapters should replace it
with their actual language-model head.
