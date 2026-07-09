<!--
Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.

License: MIT
You can find the license file in the project root.

CFR-Atlas
The documentation was written for CFR-Atlas.
8 july 2026

CFR-Atlas security policy and threat model.
-->

# Security Policy

## Reporting a vulnerability

Please report security-sensitive issues privately to the project maintainer. Do
not open public issues for problems that could cause memory unsafety, data
exfiltration, incorrect model outputs, or denial of service in a production
inference runtime.

## Trusted computing boundary

CFR-Atlas is not a sandbox. It is a safe Rust memory-virtualization core for
attention. The trusted set is intentionally small:

- `attention` — online-softmax correctness;
- `atlas` — page traversal and scratch-buffer use;
- `cache` — bounded residency and eviction accounting;
- `regenerator` implementors — backend-specific exact K/V reconstruction.

The core crate forbids `unsafe_code`. Backend adapters may use external kernels,
FFI or hardware intrinsics, but those belong to a separate trust boundary and
must be reviewed independently.

## Threat model

In scope:

- malformed configuration causing allocation or dimension errors;
- incorrect page boundaries;
- cache-budget accounting bugs;
- backend regeneration returning wrong K/V rows;
- denial of service from pathological page sizes or context lengths;
- accidental semantic drift from policy changes.

Out of scope:

- malicious model weights;
- compromised host process;
- side channels in CPU microarchitecture;
- vulnerabilities in external inference backends;
- attacks requiring arbitrary native-code execution before CFR-Atlas is called.

## Correctness as security

For an inference memory layer, wrong output can be a security problem. Any change
that affects attention math, page order, causal visibility or regeneration
contracts must include tests comparing CFR output against a full-KV baseline.

## Memory hygiene baseline

The core uses safe Rust and forbids `unsafe_code`. Cold scratch buffers are
zeroed before reuse and wiped after use by default. Hot-cache pages wipe their
live K/V storage on drop, including eviction and replacement. This is intended
to reduce accidental data retention inside the process; it is not a substitute
for process isolation or a formally verified volatile zeroization primitive.

## Capacity-safety baseline

All production K/V matrix and byte-size calculations must use checked arithmetic
before allocation, indexing or cache accounting. Overflow must return
`CfrError::CapacityOverflow`; release builds must never rely on wrapping integer
arithmetic for sizes.
