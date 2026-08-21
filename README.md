# CFR-Atlas — exact KV attention virtualization for CPU inference

> A safe Rust library for running **exact causal attention** over long contexts while bounding resident K/V-cache memory through deterministic page regeneration.

CFR-Atlas treats historical K/V as a **virtual address space**, not as data that must stay resident forever. The runtime keeps selected pages in a byte-bounded hot cache, regenerates cold pages into reusable scratch buffers, and folds each page into online softmax attention. When the backend regenerates the same K/V rows as its conventional full-KV path, CFR-Atlas preserves the attention result while trading recomputation for lower resident K/V memory.

The core is safe Rust, forbids `unsafe_code`, denies undocumented public items, and has no runtime dependencies.

## What CFR-Atlas is — and is not

CFR-Atlas owns K/V-page identity, bounded residency, exact folded attention, deterministic validation helpers, and runtime telemetry. It does **not** own model weights, tokenizer state, a transformer graph, a matmul implementation, or a lossy compression scheme.

| Concern | CFR-Atlas responsibility | Backend/runtime responsibility |
|---|---|---|
| K/V residency | Hot-page cache, scratch buffers, eviction, accounting | Cache budget selection and admission policy |
| K/V truth | Consume pages through `KvRegenerator` | Replay exact K/V rows from model state |
| Attention | Folded online-softmax reduction | Query production and output integration |
| Model semantics | Validate topology, dtype and position policy | Preserve token history, RoPE/ALiBi, head mapping and rounding |
| Performance | Expose safe tuning and telemetry surfaces | Select kernels, scheduling and deployment configuration |

## Core contract

CFR-Atlas does not prune tokens, quantize K/V, merge context, or approximate attention. The contract is deliberately conditional and reviewable:

```text
if regenerate(page_i) == baseline_kv(page_i) for every causal page,
then folded_attention(query, regenerated_pages) == full_kv_attention(query).
```

Correctness therefore depends on the adapter. A production backend must replay the same token positions, head mapping, positional policy, storage rounding, and K/V rows as its conventional stored-KV path. Cache admission changes latency and residency only; it must not change attention semantics.

## Repository layout

| Path | Purpose |
|---|---|
| `src/` | Dependency-light runtime, page/cache logic, folded attention, policies and validation types |
| `crates/cfr-atlas-backend-ref/` | Deterministic reference adapter used to exercise production integration seams |
| `crates/cfr-atlas-ollama/` | Optional typed public-Ollama integration; model discovery, generation and embeddings with exact K/V explicitly disabled |
| `tests/` | Exactness, cache invariants, topology, validation, performance-surface and stabilization tests |
| `examples/` | Minimal CPU integration, reference adapter, long-context validation and benchmark helpers |
| `docs/` | Architecture, adapter, math, claims, benchmark and release-facing guides |
| `fuzz/` | Optional nightly `cargo-fuzz` target for configuration and page-validation paths |
| `scripts/` | Release, supply-chain and fuzzing helpers |

## Quick start

CFR-Atlas targets Rust `1.75.0` or newer. The core crate has no external runtime dependencies; optional integration crates keep their own dependencies isolated from the core.

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
```

Run the deterministic examples:

```sh
cargo run --release --example toy_cpu
cargo run --release --example reference_backend
cargo run --release --example long_context_validation
cargo run --release --example bench_cfr -- 65536 64 512
cargo run --release --example bench_matrix
```

For optional configuration fuzzing, install nightly Rust and `cargo-fuzz`, then run `./scripts/run_config_fuzz.sh`.

## Minimal integration

An integration implements `KvRegenerator` for its model backend, configures the resident cache, and runs one exact attention request.

```rust
use cfr_atlas::prelude::*;
use std::ops::Range;

struct Backend;

impl KvRegenerator for Backend {
    fn regenerate_page(
        &self,
        key: PageKey,
        token_range: Range<usize>,
        head_dim: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<()> {
        // Replay the backend's normal forward path for (layer, K/V head, range)
        // and write exact row-major K[token][dim] and V[token][dim] values.
        let _ = (key, token_range, head_dim, k_out, v_out);
        Ok(())
    }
}

fn attend(query: &[f32], context_tokens: usize) -> Result<Vec<f32>> {
    let config = Config::builder(512, query.len())
        .hot_cache_bytes(256 << 20)
        .admit_regenerated_pages(true)
        .build()?;

    let mut atlas = CfrAtlas::new(config)?;
    let mut output = vec![0.0; query.len()];
    atlas.attend_exact_with_policy(
        &Backend,
        &KeepRecent { recent_tokens: 2048 },
        AttentionRequest::new(0, 0, query, context_tokens),
        &mut output,
    )?;
    Ok(output)
}
```

See [`docs/ADAPTERS.md`](docs/ADAPTERS.md) for the adapter contract and conformance sequence.

## Validation and hardening

The test suite covers deterministic output equality against a full-KV baseline, invalid configuration and non-finite input rejection, transactional cache and scratch behavior, MHA/MQA/GQA mapping, RoPE/ALiBi and dtype policy, long-context output/logit validation, and release-readiness invariants.

The hardening baseline includes checked layout arithmetic, finite-value validation before cache admission, transactional cache accounting, scratch/page wiping on relevant error paths, duplicate-field rejection in the versioned configuration schema, and optional fuzzing. See [`SECURITY.md`](SECURITY.md) for the security boundary and [`docs/CLAIMS.md`](docs/CLAIMS.md) for the conditions that make the exactness and resident-memory claims valid.

## Benchmarks

The included examples report deterministic resident-KV estimates and exercise the reference workload. They are **not** end-to-end LLM throughput claims. A model-backed Qwen2.5-0.5B page-replay and CFR conformance result, with raw data and an explicit non-end-to-end scope, is maintained in [`results/transformers_qwen2_5_0_5b_cfr.md`](results/transformers_qwen2_5_0_5b_cfr.md). Reproducible runtime benchmark methodology, measured scope, raw data, and interpretation rules are documented in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## Documentation map

| Need | Start here |
|---|---|
| Understand the runtime and memory model | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Implement a model adapter | [`docs/ADAPTERS.md`](docs/ADAPTERS.md) |
| Use supported public Ollama operations | [`docs/OLLAMA.md`](docs/OLLAMA.md) |
| Review folded-softmax math | [`docs/MATH.md`](docs/MATH.md) |
| Validate and falsify integration claims | [`docs/CLAIMS.md`](docs/CLAIMS.md) |
| Reproduce benchmark and tuning results | [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) |
| Review release/stabilization posture | [`docs/STABILIZATION.md`](docs/STABILIZATION.md) |
| Browse every guide by reader goal | [`docs/README.md`](docs/README.md) |

## Status

CFR-Atlas is a release-candidate-quality exact-attention core with a deterministic reference adapter. The optional Ollama crate supports public model discovery, generation and embeddings, but does not claim exact K/V access through the standard Ollama API. A stable virtual-K/V deployment should still require conformance and long-context validation against the target model backend, its tokenizer/position policy, its storage dtype, and its serving loop.

## License

MIT. See [LICENSE](https://github.com/Nixort/CFR-Atlas/blob/main/LICENSE).
