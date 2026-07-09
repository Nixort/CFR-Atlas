<!--
Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.

License: MIT
You can find the license file in the project root.

CFR-Atlas
The documentation was written for CFR-Atlas.
9 july 2026

CFR-Atlas project overview and user-facing build guide.
-->

# CFR-Atlas — Causal Folded Regeneration Atlas

> Exact, CPU-first memory virtualization for long-context transformer attention.

CFR-Atlas is a Rust core for running causal attention with a bounded resident KV
footprint. Its design goal is an *ideally balanced* inference path: keep the
model mathematically unchanged, keep the attention exact, and replace a large
always-resident KV cache with deterministic regeneration, hot pages, and a small
folded-attention working set.

This repository is an **early-stage production skeleton**, but the CFR core is
already testable. The central mechanisms — virtual KV pages, deterministic
`KvRegenerator`, bounded hot-cache residency, LRU eviction, scratch-page reuse,
and folded online-softmax attention — exist as safe Rust modules with integration
tests proving equality against a baseline full-KV computation for a deterministic
backend. Read `docs/ARCHITECTURE.md` for the full design, `docs/MATH.md` for the
folded-attention invariant, `docs/ROADMAP.md` for the phased delivery plan,
`docs/ADAPTERS.md` for the Phase 2 backend boundary and
`docs/STABILIZATION.md` for the Phase 5 release-readiness packet.

## Naming (atlas codenames)

| Codename        | Module / file             | Role |
|-----------------|---------------------------|------|
| **Atlas**       | `src/atlas.rs`            | Runtime coordinator: pages, cache, scratch buffers, counters |
| **Fold**        | `src/attention.rs`        | Streaming online-softmax reducer over K/V pages |
| **Forge**       | `src/regenerator.rs`      | Backend contract for exact K/V page regeneration |
| **Harbor**      | `src/cache.rs`            | Bounded resident hot-page cache with LRU eviction |
| **Chart**       | `src/page.rs`             | Stable page identity and token ranges |
| **Compass**     | `src/policy.rs`           | Residency policy: keep, admit, or drop regenerated pages |
| **Gauge**       | `src/stats.rs`            | Runtime counters and memory observability |
| **Ruler**       | `src/layout.rs`           | Checked layout math, row ranges and buffer wiping helpers |
| **Seal**        | `tests/exactness.rs`      | Exactness tests against baseline attention |
| **Sounding**    | `examples/bench_cfr.rs`   | Small benchmark for memory reduction and output equality |
| **Bridge**      | `crates/cfr-atlas-backend-ref` | Reference backend adapter |
| **Kernel**      | `src/kernel.rs`           | Safe CPU dot-product kernel boundary |
| **Tuner**       | `src/tuning.rs`           | Page-size autotuning for cache locality |
| **Pipeline**    | `src/pipeline.rs`         | Double-buffered cold-page regeneration buffers |
| **Workers**     | `src/executor.rs`         | Optional batch thread executor for cold-page work |
| **Schema**      | `src/schema.rs`           | Versioned config schema for stable embeddings |
| **Stability**   | `src/stabilization.rs`    | Phase 5 API, MSRV, no_std and supply-chain report |

## How it runs (one-paragraph mental model)

Classic inference stores K and V rows for every token, layer and head. CFR-Atlas
turns that cache into a virtual structure. `Atlas` walks causal-context pages.
If a page is hot, `Harbor` serves it from RAM. If it is cold, `Forge` regenerates
exact K/V rows into a scratch buffer. `Fold` immediately consumes those rows with
online softmax and discards the scratch page. `Compass` may admit recently useful
pages into `Harbor`, but this policy only changes speed and RAM pressure; it does
not change model quality.

## Current status

The repository now includes the backend-neutral CFR core, a reference adapter
crate, the Phase 3 performance layer, the Phase 4 validation harness and the
Phase 5 stabilization layer. Phase 4 adds logit-level full-KV comparison, sampled decode-loop validation,
multi-layer and multi-head/GQA checks, long-context memory telemetry and a
small regression corpus for short, long, repeated, code and dialogue prompt
shapes. Phase 5 adds a versioned configuration schema, public API review
metadata, MSRV policy, no_std feasibility report, supply-chain/release scripts
and an external-review claims packet. The reference adapter is not a real LLM,
but it implements the
production boundary for token-ledger replay, MHA/MQA/GQA head mapping,
RoPE/ALiBi positional policy, deterministic dtype policy and stored-KV
conformance checks.

Known-good local result from the deterministic benchmark:

```text
context_tokens=65536
head_dim=64
page_tokens=512
baseline_kv_bytes=33554432
cfr_scratch_bytes=262144
estimated_memory_reduction=128.00x
max_abs_diff=0e0
```

## Building

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
cargo run --release --example toy_cpu
cargo run --release --example bench_cfr -- 65536 64 512
cargo run --release --example reference_backend
cargo run --release --example bench_matrix
cargo run --release --example long_context_validation
cargo run --release --example stabilization_report
```

The crate is intentionally dependency-light, safe Rust only, forbids
`unsafe_code`, and denies missing public documentation at the lint level.

Optional fuzz target. The main workspace is stable Rust, but `cargo-fuzz`
requires nightly because it enables libFuzzer sanitizer instrumentation:

```sh
rustup toolchain install nightly
cargo +nightly fuzz run config_page_validation
```

A helper script is also provided:

```sh
./scripts/run_config_fuzz.sh
```

Phase 1 crate hygiene is complete: public API documentation is enforced with
`missing_docs = "deny"`, deterministic property-style tests cover page/cache
invariants, and a nightly `cargo-fuzz` target exists for config and page
validation.

The current hardening baseline additionally rejects non-finite hot-cache pages,
uses transactional cache accounting on insertion, rejects duplicate versioned
schema fields, wipes validation temporaries and generated validation queries on
error paths, and keeps benchmark and tuning inputs explicit about invalid
zero-sized shapes.


## Minimal integration

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
        // Replay the backend's exact forward path for:
        //   (key.layer, key.head, token_range)
        // and write row-major [token][head_dim] K and V.
        let _ = (key, token_range, head_dim, k_out, v_out);
        Ok(())
    }
}

fn run(query: &[f32], context_tokens: usize) -> Result<Vec<f32>> {
    let config = Config::builder(512, query.len())
        .hot_cache_bytes(256 << 20)
        .admit_regenerated_pages(true)
        .build()?;

    let backend = MyBackend;
    let mut atlas = CfrAtlas::new(config)?;
    let mut output = vec![0.0; query.len()];

    atlas.attend_exact_with_policy(
        &backend,
        &KeepRecent { recent_tokens: 2048 },
        AttentionRequest::new(0, 0, query, context_tokens),
        &mut output,
    )?;

    Ok(output)
}
```


## Phase 2 backend adapter MVP

Phase 2 is implemented as a real crate boundary:

```text
crates/
└── cfr-atlas-backend-ref/
    ├── Cargo.toml
    ├── src/lib.rs
    └── tests/conformance.rs
```

The new core APIs are:

```text
TokenLedger / TokenRecord
AttentionTopology / HeadMapping
PositionEncoding / RopeConfig / AlibiConfig
DTypePolicy / StorageDType / AccumulatorDType
compare_regenerated_page / assert_regenerated_page
```

The reference adapter proves the integration shape before wiring a real model
backend. See `docs/ADAPTERS.md` for the complete adapter checklist.

## Phase 3 performance layer

Phase 3 is implemented in the core crate without adding dependencies or unsafe
code:

```text
src/kernel.rs     safe scalar / auto-vectorized dot-product boundary
src/tuning.rs     page-size autotuning from cache and scratch constraints
src/pipeline.rs   double-buffered regenerated-page buffers
src/executor.rs   optional batch thread executor
src/policy.rs     telemetry-aware residency policy
src/cache.rs      global and per-layer hot-cache byte budgets
```

The benchmark matrix prints CSV-style rows for context length, head dimension,
page size, resident baseline bytes, scratch bytes and tuned page size:

```sh
cargo run --release --example bench_matrix
cargo run --release --example long_context_validation
cargo run --release --example stabilization_report
```

## Phase 4 validation layer

Phase 4 is implemented in `src/validation.rs` and `tests/long_context_validation.rs`.
It compares one sampled decode step through two paths:

```text
full-KV baseline -> attention output -> logits
CFR pages        -> attention output -> logits
```

The validation API exposes `PromptCase`, `PromptShape`,
`regression_corpus`, `DeterministicLogitProjector`,
`StepValidationRequest`, `ValidationPlan`, `validate_decode_step` and
`validate_decode_loop`. The helper is backend-agnostic: production integrations
can provide their own `KvRegenerator` and `LogitProjector` while keeping the
same drift and memory reports.

Run the validation example with:

```sh
cargo run --release --example long_context_validation
cargo run --release --example stabilization_report
```

## Guarantees

CFR-Atlas does not make semantic model-quality decisions. It never prunes tokens,
never quantizes K/V, never merges context, and never approximates attention. The
only contract is this:

```text
if regenerate(page_i) == baseline_kv(page_i) for every page_i,
then folded_attention(query, regenerated_pages) == baseline_attention(query, full_kv)
```

In real CPU code, exact bit identity depends on deterministic math order and the
backend's floating-point behavior. The reference folded reducer is stable and
uses `f64` accumulation for softmax bookkeeping.

## License

Licensed under the MIT License (see `LICENSE`).


## Phase 5 stabilization layer

Phase 5 is implemented without adding runtime dependencies:

```text
src/schema.rs          versioned config schema
src/bench.rs           deterministic benchmark memory estimates
src/stabilization.rs   API/MSRV/no_std/supply-chain readiness report
docs/STABILIZATION.md  release and API review packet
docs/CLAIMS.md         exactness and memory-accounting claims packet
```

The stabilization example prints the schema version, MSRV, dependency posture,
no_std feasibility result and deterministic memory estimate:

```sh
cargo run --release --example stabilization_report
```

The current `no_std` result is explicit rather than hand-waved: the crate is not
`no_std` today because `HotCache`, validation, examples and worker execution use
`std`/allocation, but an `alloc`-only core split is feasible after isolating
threading and validation behind features.
# CFR-Atlas
