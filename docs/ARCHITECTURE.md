<!--
Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.

License: MIT
You can find the license file in the project root.

CFR-Atlas
The documentation was written for CFR-Atlas.
8 july 2026

CFR-Atlas architecture notes.
-->

# CFR-Atlas Architecture

This document describes how CFR-Atlas works as a backend-neutral memory layer for
exact causal attention. It is the canonical companion to the Rust source; every
major module maps back to a section here.



## 0. Design principles

1. **Exactness before compression.** CFR-Atlas does not reduce model quality by
   deleting, merging, quantizing, or approximating K/V rows.
2. **Memory as residency, not truth.** A cold page is not lost. It is represented
   by a deterministic recipe and can be regenerated when needed.
3. **Policy cannot change semantics.** Residency policy controls speed and RAM
   pressure only. Output quality depends only on the backend's exactness.
4. **CPU locality over large residency.** Prefer small scratch pages that fit
   cache-friendly execution over a giant long-lived KV allocation.
5. **Backend neutrality.** The crate owns attention virtualization, not model
   weights, tokenization, graph execution, or matmul kernels.
6. **Safe Rust core.** The reference crate forbids `unsafe_code`; backend
   integrations may use optimized kernels behind a separate boundary.

## 1. Atlas — runtime coordinator

`CfrAtlas` owns the mutable state for one attention core:

- `HotCache`: bounded resident KV page storage;
- `scratch_k` and `scratch_v`: temporary buffers for regenerated cold pages;
- `FoldedAttention`: online softmax reducer;
- `CfrCounters`: hit, regeneration, eviction and token counters.

It deliberately does **not** own model weights, token history, tokenizer state,
RoPE tables, or the transformer graph. Those belong to the embedding runtime or
the backend implementing `KvRegenerator`.

The public execution surface is:

- `AttentionRequest`: compact call descriptor for layer, head, query and causal context length;
- `attend_exact`: lowest-residency path, regenerating cold pages and not admitting
  them unless configured;
- `attend_exact_with_policy`: same exact computation with a caller-provided
  `ResidencyPolicy`.

## 2. Chart — virtual page identity

`PageKey` is the stable identity of a causal KV page:

```text
(layer, head, start_token)
```

`PageRange` attaches the concrete half-open token range `[start, end)` to that
key. The page range is allowed to be shorter than `page_tokens` at the causal
frontier, which keeps the final page exact for arbitrary context lengths.


## 3. Forge — regeneration contract

`KvRegenerator` is the seam between CFR-Atlas and a real model backend.

A production implementation must regenerate the exact K/V rows for:

```text
(layer, head, token_range, head_dim)
```

and write them in row-major layout:

```text
K[token][dim]
V[token][dim]
```

The regenerator is the source of truth. If it returns rows equal to the rows a
classic full-KV inference path would have stored, CFR-Atlas is lossless for that
attention computation.

### Production integration notes

A real LLM backend must usually preserve or replay:

- token ledger and positions;
- embedding output needed to reach the target layer;
- RoPE / ALiBi / positional state;
- layernorm and projection math;
- grouped-query or multi-query head mapping;
- dtype and rounding policy;
- deterministic matmul order if bit-exactness is required.

CFR-Atlas does not hide this cost. It makes the cost explicit and tradeable
against resident memory.

## 4. Harbor — hot-cache residency

`HotCache` is a byte-bounded LRU cache for resident K/V pages. It stores complete
pages as owned `Vec<f32>` buffers and accounts memory as:

```text
page_bytes = tokens_in_page * head_dim * sizeof(f32) * 2
```

A page larger than the configured budget is rejected. Otherwise, insertion may
trigger eviction of least-recently-used pages until the cache is within budget.

Hot-cache residency is an optimization, not a correctness requirement. A cache
miss falls back to exact regeneration.

## 5. Fold — online-softmax attention



Classic attention may conceptually materialize all K/V rows, compute logits for
all tokens, softmax them, and then multiply by V. CFR-Atlas folds this stream.
For each K/V page, it updates:

- running maximum logit;
- running softmax denominator;
- running output accumulator.

The reducer never needs the full attention score vector and never needs all K/V
pages resident at once.

Baseline:

```text
Attention(Q, K_all, V_all)
```

CFR-Atlas:

```text
Fold(Q, page_0)
Fold(Q, page_1)
...
Finish()
```

The reference reducer accumulates softmax bookkeeping in `f64` while accepting
`f32` K/V and query slices.

## 6. Compass — residency policy

`ResidencyPolicy` decides whether a regenerated page should be admitted to the
hot cache:

```text
Drop  -> use scratch page once, then discard
Admit -> insert into hot cache if budget allows
```

Provided policies:

- `NeverAdmit`: minimum resident memory;
- `KeepRecent`: keeps pages near the causal frontier hot.

Custom policies can use recency, layer, head, prompt structure, latency budget,
or external telemetry. They must never alter token content or generated K/V.

## 7. Gauge — observability

`CfrStatsSnapshot` exposes:

- hot hits;
- cold regenerations;
- cache admissions;
- admission rejections;
- evictions;
- consumed tokens;
- current hot-cache bytes;
- current hot-cache page count.

These counters make the memory/compute tradeoff visible during integration.

## 8. Quality invariant

For a query `Q`, full-KV baseline computes:

```text
O_baseline = Attention(Q, K_all, V_all)
```

CFR-Atlas computes:

```text
O_cfr = FoldedAttention(Q, Regenerate(page_0), ..., Regenerate(page_n))
```

If every regenerated page equals the corresponding baseline page, then the two
outputs are equivalent up to ordinary floating-point ordering behavior.

The integration test `tests/exactness.rs` verifies this invariant with a
deterministic backend and reports `max_abs_diff=0` in the benchmark path.

## 9. Memory model

Classic KV memory for one attention scope is approximately:

```text
2 * layers * tokens * kv_heads * head_dim * dtype_bytes
```

CFR-Atlas resident memory is bounded by:

```text
hot_cache_budget + scratch_page + token_ledger + metadata
```

The crate directly controls `hot_cache_budget` and `scratch_page`. The host LLM
runtime controls token ledger and model state.

## 10. Non-goals

CFR-Atlas is not:

- a tokenizer;
- a transformer graph executor;
- a matmul kernel library;
- a quantizer;
- a lossy KV-compression method;
- a replacement for backend-specific CPU optimizations.

It is a memory-virtualization layer for exact K/V attention.

## 11. Phase 2 adapter boundary

Phase 2 adds a concrete backend boundary without making the CFR core depend on a
real model runtime. Backend crates use the naming pattern:

```text
cfr-atlas-backend-*
```

The repository includes `crates/cfr-atlas-backend-ref`, a deterministic reference
adapter. It is not a language model. It is a conformance harness that proves the
shape expected from real CPU inference integrations.

## 12. Token ledger

`TokenLedger` records token ids and absolute model positions. A real backend may
store richer internal state, but the CFR adapter boundary needs at least:

```text
TokenRecord { token_id, position }
```

The ledger lets a regenerator replay exactly the token span requested by a
`PageKey` and a half-open token range.

## 13. Head topology

`AttentionTopology` defines how query heads map to K/V heads:

- `mha(heads)` maps query head `h` to K/V head `h`;
- `mqa(query_heads)` maps every query head to K/V head `0`;
- `gqa(query_heads, kv_heads)` maps groups of query heads to one K/V head.

This keeps `PageKey.head` meaningful for MHA, MQA and GQA adapters.

## 14. Positional and dtype policy

`PositionEncoding` documents how the adapter preserves position-dependent model
behavior:

- `None` leaves regenerated rows unchanged;
- `Rope(RopeConfig)` applies a deterministic key-side RoPE transform;
- `Alibi(AlibiConfig)` stores causal logit-bias slopes for adapters that need
  ALiBi replay.

`DTypePolicy` lets adapters emulate deterministic storage rounding for `f32`,
`bf16` and `f16` paths before rows are consumed by folded attention.

## 15. Backend conformance

A backend should prove this invariant before enabling CFR mode:

```text
stored_kv(page) == regenerate_kv(page)
```

The core helpers `compare_regenerated_page` and `assert_regenerated_page` compare
stored K/V buffers against regenerated K/V buffers and return a
`PageConformance` report. The reference backend runs this test with RoPE, GQA and
dtype rounding enabled.

## 16. Phase 3 performance layer

Phase 3 keeps the exactness invariant and adds performance controls around it:

- `DotProductKernel` provides a safe CPU kernel boundary for scalar and
  compiler-auto-vectorized dot products.
- `PageSizeTuner` chooses page sizes from head dimension, context length, cache
  estimates and scratch memory limits.
- `DoubleBufferedPipeline` provides two reusable cold-page buffers so runtimes
  can prepare one page while another page is being consumed.
- `ThreadPoolExecutor` gives adapters a dependency-free way to run independent
  cold-page work in bounded batches.
- `HotCache` supports optional per-layer byte budgets in addition to the global
  resident budget.
- `TelemetryResidencyPolicy` can use cache counters and utilization to decide
  whether a recently regenerated page should become hot.

These features are still policy and scheduling tools. They do not change model
weights, token content, K/V values or the folded-attention math.

## 17. Phase 4 validation layer

Phase 4 adds a backend-agnostic validation harness in `src/validation.rs`. It
compares a full-KV baseline and the CFR paging path at two levels:

```text
attention output drift
logit drift after LogitProjector
```

The harness is intentionally generic. A real integration can provide its own
`KvRegenerator` and `LogitProjector`; tests and examples use the deterministic
reference backend plus `DeterministicLogitProjector`.

The key public types are:

- `PromptCase` and `PromptShape` for regression prompt classes;
- `regression_corpus` for short, long, repeated, code and dialogue shapes;
- `StepValidationRequest` for one logit-level comparison;
- `ValidationPlan` and `validate_decode_loop` for sampled decode-loop checks;
- `MemoryTelemetry` for baseline KV bytes, CFR resident bytes and estimated
  memory reduction.

The validation tests cover multi-layer replay, GQA head mapping, long-context
memory telemetry and logit-level equality against the full-KV baseline.

## 18. Phase 1 hygiene boundary

The production-hygiene layer is part of the crate contract, not a separate
experiment. The workspace denies missing public documentation, keeps CI on
`cargo fmt`, strict Clippy and release tests, and contains deterministic
property-style tests for page ranges, checked row layout and hot-cache budget
invariants.

The fuzz target under `fuzz/targets/config_page_validation.rs` exercises
configuration, page-range and hot-cache validation paths without pulling
`cargo-fuzz` into the normal workspace build. Fuzzing is intentionally a
nightly-only path because `cargo-fuzz` uses sanitizer instrumentation; the
stable workspace commands remain unchanged.

## Phase 5 stabilization boundary

Phase 5 adds release-facing modules without changing the exact attention path:

- `src/schema.rs` stores and restores `Config` through a versioned text schema;
- `src/bench.rs` provides deterministic memory estimates for benchmark matrices;
- `src/stabilization.rs` exposes API-review, MSRV, no_std and supply-chain
  readiness metadata;
- `docs/CLAIMS.md` separates exactness and memory-accounting claims from code;
- release scripts generate checksum manifests and optional detached signatures.

This layer is intentionally observational. It must not change residency policy,
regeneration, folded attention or cache semantics. Hardening changes below this
boundary are limited to preserving invariants on error paths: transactional
cache accounting, finite-value admission checks and wiping temporary validation
buffers before returning errors.
