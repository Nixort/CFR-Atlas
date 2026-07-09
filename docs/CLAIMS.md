<!--
Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.

License: MIT
You can find the license file in the project root.

CFR-Atlas
The documentation was written for CFR-Atlas.
9 july 2026

Exactness and memory-accounting claims packet.
-->

# CFR-Atlas Claims Packet

This document lists the claims that should be reviewed by external integrators.
It is intentionally conservative.

## Exactness claim

For a fixed layer, K/V head, query vector and causal context, CFR-Atlas produces
attention output equal to a full-KV baseline when all of the following hold:

- `KvRegenerator` returns the exact K and V rows that the baseline cache would
  have stored;
- row order and token ranges match the causal context;
- positional encoding, dtype rounding and head mapping are replayed exactly;
- the same attention scale and finite input values are used;
- any numerical drift is bounded by the selected accumulator and projection
  policy.

The core tests cover deterministic exactness, non-finite rejection, stale partial
hot pages, transactional folded attention, adapter conformance and Phase 4
logit-level validation.

## Memory-accounting claim

CFR-Atlas reduces resident K/V memory by replacing always-resident full-KV rows
with a scratch page plus optional hot pages. The reported memory reduction is a
resident-memory estimate, not a total process RSS guarantee. Process RSS can be
higher due to allocator behavior, executable code, model weights, thread stacks,
ASAN shadow memory or external backend buffers.

## What would falsify these claims

An integration should reject the claim for a backend if any of these happen:

- regenerated K/V differs from stored baseline K/V outside tolerance;
- query-to-K/V head mapping differs between baseline and CFR path;
- RoPE, ALiBi or dtype policy is applied in only one path;
- cache admission changes output values rather than only latency and residency;
- resident-byte accounting excludes a page that remains reachable;
- non-finite K/V values enter hot cache instead of being rejected before admission;
- duplicate config-schema fields silently override earlier values.

## External review status

The crate ships a review packet and deterministic validation harness. A real
model backend still needs an external review run over its own weights, tokenizer,
position policy, dtype policy and serving loop.
