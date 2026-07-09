<!--
Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.

License: MIT
You can find the license file in the project root.

CFR-Atlas
The documentation was written for CFR-Atlas.
9 july 2026

CFR-Atlas roadmap.
-->

# CFR-Atlas Roadmap

The roadmap is intentionally phased. CFR-Atlas should first be correct and
inspectable, then fast, then integrated with real inference runtimes.

Status key: ✅ done, 🟡 in progress, 🔴 blocked / not started, 🔬 research.

## Phase 0 — Algorithm skeleton  (✅)

**Goal.** Prove the core idea in safe Rust without depending on a model backend.

**Exit criteria.** CFR folded attention matches a full-KV baseline for a
deterministic backend; cache residency is bounded by configuration.

- [x] `PageKey` and `PageRange` for virtual KV identity
- [x] `KvRegenerator` trait for exact cold-page reconstruction
- [x] `FoldedAttention` online-softmax reducer
- [x] `HotCache` with byte budget and LRU eviction
- [x] `ResidencyPolicy`, `NeverAdmit`, `KeepRecent`
- [x] `CfrStatsSnapshot` for integration telemetry
- [x] Integration tests for baseline equality
- [x] Toy CPU example
- [x] Synthetic benchmark example

## Phase 1 — Production crate hygiene  (✅)

**Goal.** Make the repository easy to review, embed and maintain.

**Exit criteria.** CI is green under formatting, clippy, tests and examples;
public API is documented; docs describe architecture, glossary and security
posture.

- [x] Modular crate layout under `src/`
- [x] `README.md` with mental model and integration sample
- [x] `docs/ARCHITECTURE.md`
- [x] `docs/ROADMAP.md`
- [x] `docs/GLOSSARY.md`
- [x] `CONTRIBUTING.md`
- [x] `SECURITY.md`
- [x] `rust-toolchain.toml`
- [x] Document every public item and raise `missing_docs` from `warn` to `deny`
- [x] Add property tests for page boundaries and cache eviction invariants
- [x] Add nightly cargo-fuzz target for config/page validation

## Phase 2 — Backend adapter MVP  (✅)

**Goal.** Connect CFR-Atlas to one real CPU inference backend.

**Exit criteria.** A reference `cfr-atlas-backend-*` crate can regenerate
exact K/V pages through `KvRegenerator`, map MHA/MQA/GQA heads, preserve
positional and dtype policies, and pass stored-KV vs regenerated-KV equality
checks on short contexts.

- [x] Define adapter crate boundary: `cfr-atlas-backend-*`
- [x] Implement token ledger abstraction
- [x] Implement layer/head mapping for MHA, MQA and GQA
- [x] Preserve RoPE / ALiBi positional behavior
- [x] Add deterministic dtype policy for f32 / bf16 / f16 backends
- [x] Add backend conformance test: stored KV vs regenerated KV
- [x] Add short-context adapter smoke test

## Phase 3 — Performance layer  (✅)

**Goal.** Make regeneration competitive on CPU by improving locality and overlap.

**Exit criteria.** On long contexts, bounded-residency CFR mode reduces resident
KV memory by at least 10x while keeping latency within a documented budget for a
chosen backend and CPU.

- [x] SIMD dot-product kernels behind safe API boundaries
- [x] Page-size autotuning for L2/L3 cache behavior
- [x] Prefetch / double-buffered regeneration pipeline
- [x] Optional thread pool for cold-page regeneration
- [x] Per-layer hot-cache budgets
- [x] Telemetry-driven residency policy
- [x] Benchmark matrix: context length, page size, head dim, cache budget

## Phase 4 — Real long-context validation  (✅)

**Goal.** Validate the exactness claim in realistic LLM inference scenarios.

**Exit criteria.** CFR mode produces equivalent logits or bounded numerical drift
versus baseline for selected models, prompts and context lengths.

- [x] Logit-level comparison against full-KV baseline
- [x] Decode-loop integration test
- [x] Multi-layer regeneration correctness checks
- [x] Multi-head and GQA correctness checks
- [x] Long-context memory telemetry
- [x] Regression corpus for prompt shapes: short, long, repeated, code, dialogue

## Phase 5 — Stabilization  (✅)

**Goal.** Turn the crate from skeleton into an embeddable production component.

- [x] Public API review
- [x] `no_std` feasibility study for embedded inference runtimes
- [x] Versioned configuration schema
- [x] MSRV policy and CI matrix
- [x] Custom benchmark harness
- [x] Supply-chain review: `cargo deny`, SBOM, signed release artifacts
- [x] External review packet for exactness and memory-accounting claims

## Cross-cutting tracks

- **Docs:** keep README and ARCHITECTURE aligned with code.
- **Testing:** exactness first; performance second.
- **Performance:** measure RAM, bandwidth, cache misses and latency together.
- **Safety:** no `unsafe` in the CFR core unless a future optimized module proves
  and documents its invariants.
