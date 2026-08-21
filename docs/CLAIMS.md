# Claims and validation boundary

This document states what CFR-Atlas claims, the conditions required for each claim, and the observations that invalidate it for a particular backend integration. It is intentionally narrower than a product-performance statement.

## Exact-attention claim

For a fixed layer, K/V head, query vector and causal context, CFR-Atlas produces the same attention result as a conventional full-KV path **when** the following conditions hold.

| Condition | Why it is required |
|---|---|
| `KvRegenerator` returns the baseline K and V rows | Regeneration replaces residency; it cannot replace source-of-truth values |
| Page ranges and row order match the causal context | Online folding is sensitive to the ordered token stream |
| Head mapping matches baseline MHA/MQA/GQA behavior | The requested page must belong to the same K/V head |
| Position and dtype policy are replayed identically | RoPE, ALiBi and rounding can change values before attention |
| Attention scale and finite-input policy match | The reducer assumes the same numerical contract as the baseline |
| Accumulation/projection tolerance is stated | Different floating-point execution order may require an explicit tolerance |

The included deterministic reference backend reaches `max_abs_diff = 0` in covered validation paths. That result is evidence for the reference fixture, not a blanket guarantee for an untested model runtime.

## Resident-memory claim

CFR-Atlas bounds **its resident K/V contribution** through the configured hot-cache budget, scratch-page capacity and metadata. It can reduce the always-resident K/V footprint compared with retaining every historical K/V page at once.

This is not a promise about total process RSS. Model weights, allocator arenas, thread stacks, executable code, backend activations, external caches, sanitizers and operating-system behavior remain outside CFR-Atlas page accounting. Benchmark reports must distinguish resident-KV estimates from measured whole-process memory.

## What CFR-Atlas does not claim

CFR-Atlas does not claim end-to-end LLM throughput, latency, token/s, quality, or RSS improvement for an unspecified model/runtime/hardware configuration. It does not claim that arbitrary adapters can replay K/V exactly, or that a residency policy can repair a conformance mismatch.

## Falsification criteria

Reject the exactness claim for a backend configuration if any of the following is observed:

- regenerated K/V differs from stored baseline K/V outside the declared tolerance;
- query-to-K/V head mapping differs between baseline and CFR paths;
- RoPE, ALiBi, token positions or dtype rounding are applied on only one path;
- cache admission, eviction or page scheduling changes output values;
- a partial final page is replayed with a different token range or layout;
- non-finite K/V values enter the hot cache rather than being rejected;
- a benchmark reports resident-memory savings while including reachable K/V pages outside the accounting scope.

Reject the memory-accounting claim if a reachable resident page is omitted from byte accounting, a cache-budget violation is observed, or a report presents an estimate as process RSS without measuring that RSS.

## Evidence shipped in this repository

| Evidence | Scope |
|---|---|
| `tests/exactness.rs` | Deterministic full-KV versus CFR output equality and cache behavior |
| `crates/cfr-atlas-backend-ref/` | Reference adapter topology, position, dtype and page-conformance checks |
| `tests/long_context_validation.rs` | Output/logit comparison and memory telemetry on the reference fixture |
| `tests/property_invariants.rs` | Checked layout, page-range and cache-accounting properties |
| `docs/BENCHMARKS.md` | Measurement scope, reproducibility rules and interpretation limits |
| `SECURITY.md` | Threat model and hardening baseline |

## External review requirement

A production model integration remains unreviewed until its own stored-KV/replayed-KV conformance, output/logit validation, supported-mode matrix, and benchmark environment have been examined. The reference backend and these documents make that review repeatable; they do not substitute for it.
