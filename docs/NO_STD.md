<!--
Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.

License: MIT
You can find the license file in the project root.

CFR-Atlas
The documentation was written for CFR-Atlas.
9 july 2026

no_std feasibility note.
-->

# no_std Feasibility

CFR-Atlas is not `no_std` today. Phase 5 records this explicitly through
`NoStdFeasibilityReport`.

Current blockers:

- `HotCache` uses map allocation;
- validation and benchmark harnesses allocate vectors;
- optional thread execution uses `std::thread`;
- examples and release tooling are standard-library programs.

Feasible future split:

- keep checked layout math, page identity and online softmax in an `alloc`-light
  core;
- gate `HotCache`, validation, workers and examples behind `std` features;
- expose caller-owned scratch buffers for embedded runtimes that want to control
  allocation.
