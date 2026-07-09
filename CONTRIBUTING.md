<!--
Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.

License: MIT
You can find the license file in the project root.

CFR-Atlas
The documentation was written for CFR-Atlas.
9 july 2026

CFR-Atlas contributor guide.
-->

# Contributing to CFR-Atlas

Thanks for your interest in CFR-Atlas. This is an exactness-sensitive inference
component; the bar for changes inside attention, regeneration contracts and
memory accounting is high.

## Ground rules

- **Discuss first.** Open an issue describing the design before large PRs.
- **Small, reviewable commits.** Follow Conventional Commits
  (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `build:`).
- **No `unsafe` in the core.** The crate currently forbids `unsafe_code`. If a
  future optimized module needs it, isolate it and document every invariant with
  a `// SAFETY:` comment.
- **Exactness changes need tests.** Any change to `FoldedAttention`, page
  traversal or regeneration semantics must update or add baseline-comparison
  tests.
- **Memory claims need measurement.** Do not claim xN reduction without a command
  and output that can reproduce the number.

## Workflow

```sh
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test --release
cargo run --release --example toy_cpu
cargo run --release --example bench_cfr -- 65536 64 512
```

All of the above should pass before review. CI enforces the same shape of checks.

## Code style

- Document every public item; `missing_docs` is expected to become denied.
- Track unfinished work in issues or review notes rather than leaving stale inline markers.
- Keep modules small and role-scoped: atlas, attention, cache, page, policy,
  regenerator, stats.
- Keep policy separate from correctness. A residency policy must never alter
  K/V values, token order or causal visibility.
- Prefer explicit memory accounting over hidden allocation.

## Test style

Every correctness test should name the invariant it protects:

- baseline equality;
- page-boundary behavior;
- cache-budget enforcement;
- policy-independent output;
- deterministic regeneration.

## Licensing of contributions

By contributing you agree your work is licensed under the MIT License used by
this repository.
