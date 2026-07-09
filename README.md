# CFR-Atlas

CFR-Atlas is a safe Rust core for CPU-first long-context attention with a bounded resident KV-cache footprint.

Instead of keeping every K/V row resident in memory, CFR-Atlas treats the KV cache as a virtual structure. Hot pages can stay in a bounded cache; cold pages are regenerated deterministically into scratch buffers, consumed by an exact folded online-softmax attention pass, then wiped or reused. The algorithm trades CPU recomputation for lower resident memory while preserving the same attention result when the regenerator reproduces the baseline K/V rows exactly.

## What this repository contains

- A backend-neutral CFR runtime: page scheduling, hot-cache residency, scratch buffers, folded attention, and runtime counters.
- A deterministic `KvRegenerator` trait for exact K/V page replay.
- A reference backend adapter crate that demonstrates token replay, MHA/MQA/GQA head mapping, RoPE/ALiBi positional policy, dtype policy, and stored-KV conformance checks.
- Validation utilities for comparing CFR attention against a full-KV baseline at output and logit level.
- Benchmark helpers for memory estimates, tuned page sizes, and reproducible example runs.
- Release-readiness helpers for schema versioning, MSRV policy, dependency posture, and release metadata.

The codebase uses safe Rust only, forbids `unsafe_code`, denies missing public documentation, and currently has zero runtime dependencies in the main crate.

## Core guarantee

CFR-Atlas does not prune tokens, quantize K/V, merge context, or approximate attention. Its core contract is:

```text
if regenerate(page_i) == baseline_kv(page_i) for every causal page,
then CFR folded attention == full-KV baseline attention.
```

The resident memory reduction comes from not storing every historical K/V page at once. Correctness depends on deterministic regeneration and a backend that can replay the exact K/V rows required for each page.

## Build and test

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
```

Run the examples:

```sh
cargo run --release --example toy_cpu
cargo run --release --example bench_cfr -- 65536 64 512
cargo run --release --example reference_backend
cargo run --release --example bench_matrix
cargo run --release --example long_context_validation
cargo run --release --example stabilization_report
```

A known-good deterministic benchmark run reports exact output equality and a 128x resident scratch-vs-baseline KV estimate for a 65,536-token context with `head_dim=64` and `page_tokens=512`:

```text
context_tokens=65536
head_dim=64
page_tokens=512
baseline_kv_bytes=33554432
cfr_scratch_bytes=262144
estimated_memory_reduction=128.00x
max_abs_diff=0e0
```

## Optional fuzzing

The main workspace builds on stable Rust. The fuzz target uses `cargo-fuzz`, which requires nightly because libFuzzer uses sanitizer instrumentation.

```sh
rustup toolchain install nightly
cargo install cargo-fuzz
./scripts/run_config_fuzz.sh
```

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

## Repository layout

```text
src/atlas.rs          runtime coordinator
src/attention.rs      folded online-softmax attention
src/cache.rs          bounded hot-page cache
src/config.rs         runtime configuration
src/conformance.rs    regenerated-page comparison helpers
src/dtype.rs          deterministic storage and accumulator dtype policy
src/kernel.rs         safe CPU dot-product kernel boundary
src/layout.rs         checked layout math and buffer wiping helpers
src/page.rs           page identity and token ranges
src/pipeline.rs       double-buffered cold-page buffers
src/policy.rs         residency policies and telemetry-aware admission
src/position.rs       RoPE and ALiBi helpers
src/regenerator.rs    K/V regeneration trait
src/schema.rs         versioned config schema
src/stabilization.rs  release-readiness metadata
src/stats.rs          runtime counters
src/token.rs          token ledger
src/topology.rs       MHA/MQA/GQA head mapping
src/tuning.rs         page-size tuning helpers
src/validation.rs     long-context validation utilities

crates/cfr-atlas-backend-ref/  reference backend adapter
examples/                      runnable demos and benchmark helpers
tests/                         exactness, invariants, validation and release checks
fuzz/                          optional nightly fuzz target
docs/                          architecture, math, adapter and release documentation
scripts/                       fuzz, supply-chain and release helper scripts
```

## Validation coverage

The test suite covers:

- exact equality against deterministic full-KV baseline attention;
- rejection of invalid configs, invalid ranges, non-finite inputs, and malformed schema data;
- cache byte accounting, global and per-layer budgets, replacement behavior, and LRU invariants;
- transactional behavior for folded attention, cache insertion, cold-page buffers, and validation outputs;
- MHA/MQA/GQA topology mapping, RoPE/ALiBi checks, token ledger monotonicity, and dtype determinism;
- long-context output/logit comparison and memory telemetry;
- release-readiness checks for schema versioning, MSRV policy, dependency posture, and benchmark estimates.

## Security and hardening posture

The current hardening baseline includes:

- `unsafe_code = forbid`;
- checked layout and byte-size math;
- finite-value validation before hot-cache admission;
- transactional cache accounting and replacement;
- explicit buffer wiping on cache eviction, scratch reuse, validation failures, and backend error paths;
- duplicate-field rejection in the versioned config schema;
- release helper scripts for manifest/checksum generation and supply-chain checks.

See `SECURITY.md`, `docs/ARCHITECTURE.md`, `docs/MATH.md`, `docs/ADAPTERS.md`, `docs/STABILIZATION.md`, and `docs/CLAIMS.md` for more detail.

## Status

CFR-Atlas is ready for release-candidate style testing and external review. The core is deterministic and heavily validated against the included reference backend, but a final stable `1.0.0` release should only be cut after integration with a real model backend and a frozen public API review.

## License

Licensed under the MIT License. See `LICENSE` for details.
