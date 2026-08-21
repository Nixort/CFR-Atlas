# Roadmap

CFR-Atlas has completed the core exact-attention, reference-adapter, validation and release-hygiene foundation. The remaining work is intentionally integration-led: prove behavior on named real runtimes, measure the memory/compute trade-off reproducibly, and freeze an API only after that evidence exists.

## Current baseline

| Area | Status | Evidence in this repository |
|---|---|---|
| Exact folded attention core | Available | Deterministic full-KV equality and transactional error-path tests |
| Bounded K/V residency | Available | Hot cache budgets, scratch limits, page accounting and telemetry |
| Reference adapter | Available | `cfr-atlas-backend-ref` conformance fixture with topology/position/dtype coverage |
| Long-context validation harness | Available | Output and logit-level validation types plus deterministic regression corpus |
| Release hygiene | Available | MSRV, strict linting, fuzz target, versioned schema and supply-chain helpers |
| Real-model integration | Required before stable deployment | Target-runtime conformance and serving-loop validation are backend-specific |
| Reproducible runtime benchmarks | In progress | Harness and publication artifacts should report a named workload and environment |

## Next milestones

### 1. Named production backend integration

Choose one CPU inference runtime and publish an adapter crate following the `cfr-atlas-backend-*` convention. The acceptance gate is not compilation: it is stored-KV versus regenerated-KV conformance across supported layers, K/V heads, final partial pages, topology, position policy and storage dtype.

### 2. Long-context model validation

For the selected backend, run `validate_decode_step` and `validate_decode_loop` on a versioned regression corpus. Publish the model/runtime version, prompt shapes, context lengths, page size, cache budget, output/logit tolerance and any unsupported modes.

### 3. Reproducible runtime measurements

Publish raw CSV, environment metadata, baseline definition, repeat/median policy, validation outcome and plots for the deterministic reference workload first. Extend to a real backend only after the adapter passes the earlier gates. Report resident-KV accounting separately from process RSS and never infer generic LLM throughput from the reference fixture.

### 4. API stabilization for `1.x`

After a named integration is validated, review the public surface for configuration defaults, error taxonomy, extension traits, semver commitments and documentation examples. New runtime scheduling or kernel work should stay behind backward-compatible APIs until its invariants are reviewed.

## Engineering principles

- **Exactness before speed.** A timing win is not useful if conformance has not passed.
- **Measured claims only.** Every benchmark needs a workload, environment, raw data and validation status.
- **Private refactors first.** Preserve the public API unless a reviewed semver decision requires a change.
- **Resident memory is not RSS.** State the accounting boundary in every report.
- **Safe core by default.** New `unsafe` code requires a separate, documented invariant boundary.
