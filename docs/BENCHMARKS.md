# Benchmarks

CFR-Atlas benchmarks must measure a clearly named workload and distinguish three different quantities: runtime, bounded resident-K/V memory, and end-to-end model behavior. The repository benchmark harness targets the included deterministic reference workload; it is not a claim about arbitrary LLM throughput.

## Measurement discipline

Every reported run should record the following:

| Item | Requirement |
|---|---|
| Workload | Context length, head dimension, page size, layer/head scope and cache budget |
| Backend | Reference fixture or named production adapter/version |
| Timing | Release build, warm-up policy, repeated samples and reported median/min/max |
| Validation | Exact output comparison or stated output/logit tolerance after each timed run |
| Memory | Full-KV estimate and CFR resident-KV accounting reported separately from RSS |
| Environment | CPU model, core count, OS, Rust compiler and relevant runtime flags |

A timing sample that fails conformance must not be reported as a successful performance result.

## Included workload

The built-in deterministic reference backend makes the paging path reproducible without downloading model weights or datasets. It supports comparisons among:

- full resident-KV consumption for the same synthetic K/V stream;
- CFR cold-page regeneration with no cache admission;
- CFR with a bounded recent-page hot cache;
- page-size and cache-budget sweeps under identical query/context settings.

The reference workload exercises page generation, folded attention, cache admission and validation. It intentionally does not claim tokenizer, transformer-layer, matrix-kernel, network or full-model serving performance.

## Reproduction

Build and validate before interpreting timings:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
```

Run the deterministic memory estimate and scenario matrix:

```sh
cargo run --release --example bench_cfr -- 65536 64 512
cargo run --release --example bench_matrix
```

The release benchmark harness and report artifacts are maintained under `bench/` and `results/`. The harness emits raw CSV; plot generation turns only those raw rows into the rendered chart. Do not edit generated result tables or figures by hand.

## Result interpretation

A positive resident-memory ratio means the reported full-KV estimate is larger than the active CFR scratch plus configured cache scope. It does **not** imply proportional latency improvement: CFR intentionally trades resident memory for regeneration work. A cache policy can improve reuse, but it cannot make a backend conformance mismatch valid.

When comparing configurations, use the same context length, head dimension, page size, query sequence, build flags and validation policy. Avoid comparing a cache-warm CFR run against a baseline with unrelated initialization or allocator work included.

## Tuning sweeps

The useful tuning dimensions are page size and hot-cache budget. Smaller pages lower per-page scratch requirements but may raise regeneration and loop overhead; larger pages increase locality and per-page work. Larger cache budgets may reduce cold regenerations but increase resident K/V bytes. Publish both the runtime and resident-memory column so the trade-off remains visible.

## Publication checklist

Before adding a chart or performance statement to the README, include:

1. the raw CSV and its schema;
2. a command that regenerates it;
3. the plotting command and generated image path;
4. a baseline definition;
5. the number of samples and summary statistic;
6. validation outcome for every measured configuration;
7. environment details and a statement of what the result does not measure.
