# CFR-Atlas architecture

CFR-Atlas is a backend-neutral runtime for **virtualizing historical K/V pages** during causal attention. It owns page scheduling, bounded residency, reusable scratch storage, online-softmax folding and operational counters. The embedding runtime remains responsible for producing the query and replaying K/V exactly.

```text
AttentionRequest
      │
      ▼
CfrAtlas page loop ── hot page hit ──► FoldedAttention
      │                                      │
      └─ cold page ─► KvRegenerator ─► scratch K/V ─┘
                              │
                              └─ ResidencyPolicy ─► optional HotCache admission
```

## Architectural contract

| Invariant | Enforced by | Why it matters |
|---|---|---|
| Page identity is stable | `PageKey(layer, head, start_token)` | A regenerated page can be compared to the baseline page it replaces |
| Page range is causal and bounded | `PageRange`, checked layout helpers | The final partial page remains exact and scratch capacity stays valid |
| K/V rows are deterministic | `KvRegenerator` contract | CFR is exact only when replay matches the conventional stored-KV path |
| Residency cannot alter values | `HotCache` and `ResidencyPolicy` | Policy changes cost and RAM, never attention semantics |
| Folding is transactional | `FoldedAttention` and runtime error paths | Failed pages do not leave partial output or invalid cache accounting |
| Resident bytes are explicit | `HotCache`, `CfrStatsSnapshot` | Memory estimates remain distinguishable from whole-process RSS |

## Execution path

`CfrAtlas::attend_exact_with_policy` processes one `(layer, K/V head, query)` request over a causal context. For each page it:

1. validates query/output shape and page bounds;
2. checks whether a shape-compatible hot page is resident;
3. consumes that page directly on a cache hit;
4. otherwise asks `KvRegenerator` to write the exact K/V page into reusable scratch buffers;
5. folds the page into online softmax attention;
6. asks the `ResidencyPolicy` whether the regenerated page should become hot;
7. wipes scratch buffers where configured, including relevant error paths.

The public `attend_exact` method uses `NeverAdmit`. `attend_exact_with_policy` exposes the same computation with caller-controlled cache admission.

## Page identity and data ownership

A virtual page is identified by:

```text
(layer, kv_head, start_token)
```

`PageRange` adds the half-open causal interval `[start, end)`. A final page may be shorter than `Config::page_tokens`; this is an expected boundary condition, not a special approximation path.

The core owns only virtual-page metadata and optional K/V residency. It does not own model weights, tokenizer state, transformer intermediate activations, positional tables or a matmul implementation. Those remain behind the backend adapter boundary.

## Regeneration boundary

`KvRegenerator` is the source-of-truth seam between CFR-Atlas and an actual model runtime. The adapter receives a page key and token range, then writes row-major buffers:

```text
K[token][dimension]
V[token][dimension]
```

A production adapter must preserve the semantics of its conventional stored-KV path: token ledger/absolute positions, layer and K/V-head selection, MHA/MQA/GQA mapping, RoPE or ALiBi policy, storage dtype rounding, and any relevant deterministic execution order. See [`ADAPTERS.md`](ADAPTERS.md) for the full conformance sequence.

## Hot residency and scratch memory

`HotCache` is a byte-bounded LRU cache of complete K/V pages. The approximate resident size of one `f32` page is:

```text
page_bytes = tokens_in_page × head_dim × sizeof(f32) × 2
```

A page larger than the configured budget is rejected. Other insertion paths may evict least-recently-used pages until both global and optional per-layer budgets are respected. A miss falls back to deterministic regeneration; it is not a correctness failure.

Cold pages use `scratch_k` and `scratch_v`. The runtime bounds their shape through `Config::max_scratch_tokens` and reuses their allocation across requests. `DoubleBufferedPipeline` is available for runtimes that need a separate reusable two-slot cold-page helper.

## Folded attention

Conventional attention conceptually materializes every K/V row, computes all logits, applies softmax and then reduces values. CFR-Atlas streams the same page sequence through `FoldedAttention`.

For every page, the reducer updates a running maximum logit, softmax denominator and output accumulator. It never needs a full score vector or every historical K/V page resident at once. The reference reducer uses `f64` for the softmax bookkeeping while accepting `f32` query/K/V buffers. The derivation and assumptions live in [`MATH.md`](MATH.md).

## Policies, tuning and observability

`ResidencyPolicy` determines only whether a regenerated page is admitted after it has been consumed. Included policies are `NeverAdmit` and `KeepRecent`; `TelemetryResidencyPolicy` can use cache utilization and counters. `PageSizeTuner`, `DotProductKernel`, `ThreadPoolExecutor`, per-layer budgets and `DoubleBufferedPipeline` are scheduling/locality controls. They must not mutate token content, K/V values or folded-attention math.

`CfrStatsSnapshot` reports hot hits, cold regenerations, admissions, rejections, evictions, consumed tokens, current resident bytes and current page count. These counters are integration telemetry, not a benchmark by themselves.

## Exactness and validation

For a fixed query and causal context:

```text
full_KV_attention(query, K_baseline, V_baseline)
≈ folded_attention(query, regenerate(page_0), …, regenerate(page_n))
```

The approximation symbol reflects normal floating-point ordering policy. With the deterministic reference backend, the repository tests exact equality for its covered scenarios. A production integration should first prove page-level stored-KV versus regenerated-KV conformance, then run output and logit-level validation through the `validation` module across prompt shapes, layers, heads and positions.

## Memory interpretation and non-goals

The standard full-KV estimate for one attention scope is approximately:

```text
2 × layers × tokens × kv_heads × head_dim × dtype_bytes
```

CFR-Atlas bounds its own resident K/V contribution through hot-cache budget, scratch pages and metadata. It does **not** promise a particular whole-process RSS: model weights, allocator behavior, thread stacks, backend buffers, executable code and instrumentation remain outside this accounting boundary.

CFR-Atlas is not a tokenizer, transformer graph executor, quantizer, lossy KV-compression format, model-specific kernel library or a substitute for backend conformance. Its single purpose is exact K/V attention virtualization under an explicit memory/compute trade-off.
