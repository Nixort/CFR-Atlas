<!--
Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.

License: MIT
You can find the license file in the project root.

CFR-Atlas
The documentation was written for CFR-Atlas.
7 july 2026

CFR-Atlas glossary.
-->

# Glossary

- **Atlas** — the CFR runtime coordinator that walks virtual KV pages and owns
  hot cache, scratch buffers and counters.
- **CFR** — Causal Folded Regeneration: exact attention over regenerated causal
  K/V pages.
- **Cold page** — a KV page not resident in RAM; it exists as a regeneration
  recipe through the backend.
- **Folded attention** — online-softmax attention that consumes K/V rows
  sequentially instead of materializing all rows or all logits at once.
- **Forge** — informal codename for the backend implementing `KvRegenerator`.
- **Hot cache** — byte-bounded RAM storage for recently useful K/V pages.
- **KV cache** — transformer key/value memory used by causal attention.
- **PageKey** — stable identity of a virtual KV page: layer, head and start
  token.
- **Regeneration** — replaying the exact backend computation needed to produce a
  requested K/V page.
- **Residency policy** — speed/RAM policy deciding whether regenerated pages are
  admitted to the hot cache.
- **Scratch page** — temporary K/V buffer reused for cold-page regeneration.
- **Token ledger** — host-runtime record of tokens and positions needed to
  regenerate pages exactly.

## Phase 2 terms

**Adapter crate.** A crate named `cfr-atlas-backend-*` that connects CFR-Atlas to
a concrete inference backend.

**Token ledger.** Append-only record of token ids and absolute positions used for
backend replay.

**MHA.** Multi-head attention. Each query head has its own K/V head.

**MQA.** Multi-query attention. All query heads share one K/V head.

**GQA.** Grouped-query attention. A group of query heads shares one K/V head.

**RoPE.** Rotary positional embedding. CFR-Atlas stores a deterministic adapter
policy through `RopeConfig`.

**ALiBi.** Attention with linear biases. CFR-Atlas stores deterministic adapter
metadata through `AlibiConfig`.

**Dtype policy.** Adapter rule for deterministic storage rounding before K/V rows
are compared or consumed.

**Conformance report.** `PageConformance`, the stored-KV vs regenerated-KV check
used before enabling CFR mode for a backend.

## Phase 3 terms

**Dot-product kernel.** Safe CPU kernel boundary used by `FoldedAttention` for
query/key dot products. The current crate provides scalar and compiler
auto-vectorized safe Rust kernels.

**Page-size tuner.** `PageSizeTuner`, the helper that selects page tokens from
cache estimates, scratch limits, head dimension and context length.

**Double buffering.** Two reusable cold-page buffers that let a runtime prepare
one page while another page is consumed or scheduled.

**Thread executor.** `ThreadPoolExecutor`, a small dependency-free batch executor
for independent cold-page jobs.

**Per-layer budget.** Optional hot-cache byte limit applied to one transformer
layer in addition to the global hot-cache budget.

**Telemetry residency.** A residency policy that uses cache counters and
utilization to decide whether a regenerated page should be admitted.

## Phase 4 terms

**Validation prompt.** `PromptCase`, a tokenized prompt used to validate one
prompt shape such as short, long, repeated, code or dialogue.

**Logit projector.** `LogitProjector`, the boundary that maps an attention output
vector into logits so CFR output can be compared after the model-head step.

**Decode-loop validation.** A sampled loop over prompt positions, layers and
query heads that compares full-KV baseline outputs against CFR folded outputs.

**Memory telemetry.** `MemoryTelemetry`, the per-step report containing baseline
KV bytes, CFR scratch bytes, hot-cache bytes and estimated memory reduction.

**Regression corpus.** `regression_corpus`, the built-in prompt-shape set
used by the long-context validation tests and example.

## Phase 5 terms

**Versioned config schema** — deterministic text representation of `Config`
with an explicit `CONFIG_SCHEMA_VERSION`.

**MSRV** — minimum supported Rust version checked by CI. The current stable
workspace MSRV is `1.75`.

**Claims packet** — review document that states the exactness and
memory-accounting assumptions an external backend must verify.

**SBOM / release manifest** — machine-readable metadata and checksums generated
for release review.
