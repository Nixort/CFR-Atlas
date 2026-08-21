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

## Published reference result

The first corpus-backed reference-workload report uses Tiny Shakespeare at context `65,536`, head dimension `64`, page size `512`, an 8 MiB hot-cache budget and a five-run median policy. It records `max_abs_diff = 0` for every reported CFR row. On the recorded environment, `cfr_cold` accounts for `0.25 MiB` active resident K/V versus `32.00 MiB` for full K/V, while bounded `cfr_hot` accounts for `2.50 MiB`. The exact timings, min/max range, environment, limitations, raw CSV and generated chart are maintained in [`../results/cfr_atlas_tinyshakespeare.md`](../results/cfr_atlas_tinyshakespeare.md).

This is evidence for the included deterministic reference workload only. It is not an end-to-end model, process-RSS or universal throughput claim.

## Live Ollama public-API result

The repository also contains a real local-Ollama integration report for `qwen2.5:0.5b`: [`results/ollama_qwen2_5_0_5b_public_api/README.md`](../results/ollama_qwen2_5_0_5b_public_api/README.md). It includes three raw generation samples, a model digest, runtime/environment record, derived JSON summary and chart. The measurements demonstrate that `cfr-atlas-ollama` can discover and invoke a real model through documented public endpoints.

This report is deliberately **not** included in the CFR performance table. Public Ollama endpoints do not supply page-level K/V replay or a stored-K/V baseline, so they cannot produce the conformance and resident-memory evidence required for a CFR virtual-K/V result. The integration remains fail-closed; see [`OLLAMA.md`](OLLAMA.md) for that boundary.

## Reproduction

Build and validate before interpreting timings:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
```

Fetch the public corpus, run the release harness and render its raw output:

```sh
python3 bench/fetch_tinyshakespeare.py
cargo run --release --bin bench-cfr -- \
  --corpus bench/data/tinyshakespeare.txt \
  --context-tokens 65536 \
  --head-dim 64 \
  --page-tokens 512 \
  --hot-cache-bytes 8388608 \
  --recent-tokens 4096 \
  --runs 5 | tee results/cfr_atlas_tinyshakespeare_raw.csv

python3 bench/plot_results.py results/cfr_atlas_tinyshakespeare_raw.csv \
  --output results/cfr_atlas_tinyshakespeare.png \
  --summary results/cfr_atlas_tinyshakespeare_summary.csv
```

The harness emits raw CSV; plot generation turns only those raw rows into the derived summary and rendered chart. Do not edit generated result tables or figures by hand. `examples/bench_cfr.rs` and `examples/bench_matrix.rs` remain lightweight deterministic examples; the `bench/` harness is the release-report path.

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
