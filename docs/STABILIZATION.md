<!--
Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.

License: MIT
You can find the license file in the project root.

CFR-Atlas
The documentation was written for CFR-Atlas.
9 july 2026

Phase 5 stabilization packet.
-->

# Phase 5 Stabilization

Phase 5 turns the crate into an embeddable component with explicit release gates.
It does not claim that CFR-Atlas is a finished model runtime. It makes the public
surface, configuration format, MSRV, supply-chain process and review claims
inspectable.

## Public API review

The public API is centralized in `src/lib.rs`. Phase 5 adds
`stabilization_report()` so downstream integrations can inspect the API-review
status without parsing documentation. The current policy is:

- public items must be documented through `missing_docs = "deny"`;
- `unsafe_code = "forbid"` remains active for the core crate;
- root re-exports are intentional and reviewed;
- new experimental areas should be represented by explicit modules, not hidden
  re-exports.

## Versioned configuration schema

`src/schema.rs` defines `CONFIG_SCHEMA_VERSION` and `VersionedConfig`. The format
is deterministic newline-separated key/value text. It is intentionally simple so
it can be checked into application configs, embedded into release artifacts and
migrated later without depending on Rust debug output. Decoding rejects unknown,
missing and duplicate fields so a release config cannot silently shadow an older
value later in the file. The hardening baseline keeps conversion fallbacks
explicit through checked or saturating helpers and avoids hand-written fallback
matches that drift from the linted style used by the workspace.

## MSRV and CI matrix

The stable workspace MSRV is `1.75`, matching `Cargo.toml` and
`rust-toolchain.toml`. CI now checks both MSRV and stable. Fuzzing remains
nightly-only because `cargo-fuzz` uses sanitizer `-Z` flags.

## no_std feasibility

The crate is not `no_std` today. The current blockers are allocation-heavy cache
storage, validation harnesses, examples and optional worker execution. The core
math and page-layout pieces are suitable candidates for a future `alloc`-only
split. This is captured by `NoStdFeasibilityReport` rather than hidden in prose.

## Benchmark harness

`src/bench.rs` provides deterministic benchmark memory estimates and standard
scenarios. Wall-clock examples still exist, but Phase 5 gives release tooling a
small custom harness that can be tested without Criterion or external crates.

## Supply chain and release artifacts

The main crate has zero runtime dependencies. Phase 5 ships:

- `deny.toml` for `cargo deny` policy review;
- `scripts/supply_chain_check.sh` for `cargo deny` review;
- `scripts/release_manifest.sh` for metadata and checksum manifests;
- `scripts/sign_release_artifacts.sh` for detached GPG signatures.

The signing script requires a caller-provided `GPG_SIGNING_KEY`; the repository
must not ship a private key.

## Hardening baseline

The current stabilization baseline includes these additional invariants:

- hot-cache insertion rejects non-finite K/V values before admission;
- hot-cache accounting preflights byte additions before mutating resident usage;
- validation projection writes through temporary buffers and wipes them on errors;
- full-KV validation temporaries are wiped if output finalization fails;
- page-size tuning handles short contexts smaller than the minimum tuning bound;
- deterministic benchmark estimates reject zero-sized or oversized page shapes.
