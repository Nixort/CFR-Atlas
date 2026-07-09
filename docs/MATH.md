<!--
Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.

License: MIT
You can find the license file in the project root.

CFR-Atlas
The documentation was written for CFR-Atlas.
9 july 2026

CFR-Atlas mathematical notes.
-->

# CFR-Atlas Mathematics

This document records the mathematical invariant behind CFR-Atlas: resident
memory may change, but the attention computation remains exact when regenerated
K/V pages match the baseline K/V pages.


## 1. Baseline attention

For one query vector `q`, full-KV attention is usually written as:

```text
logit_t = dot(q, k_t) / sqrt(d)
weight_t = exp(logit_t) / sum_j exp(logit_j)
out = sum_t weight_t * v_t
```

The conceptual baseline needs the entire visible K/V sequence available for the
query. A classic runtime normally keeps that sequence resident in a KV cache.

## 2. Folded online softmax

CFR-Atlas processes the same sequence page by page. The reducer keeps three
pieces of state:

```text
m = running maximum logit
z = running softmax denominator
a = running output accumulator
```

For each token row in the page:

```text
l  = dot(q, k_t) / sqrt(d)
m' = max(m, l)
z' = z * exp(m - m') + exp(l - m')
a' = a * exp(m - m') + v_t * exp(l - m')
```

At the end:

```text
out = a / z
```

This is the same softmax normalization, only streamed through a numerically
stable recurrence. CFR-Atlas can therefore consume a cold page, fold it into the
state, and immediately discard the scratch buffer.

## 3. Exactness invariant

Let `K_i, V_i` be the baseline K/V rows for page `i`. Let `R(i)` be the backend
regenerator result for the same page.

```text
if R(i) == (K_i, V_i) for every page i,
then Fold(q, R(0), R(1), ..., R(n)) == Attention(q, K_all, V_all)
```

In real CPU code, bit identity depends on deterministic floating-point order.
The reference reducer uses stable online-softmax bookkeeping and `f64`
accumulation for the softmax state.

## 4. Memory model

Classic resident KV grows approximately as:

```text
2 * layers * tokens * kv_heads * head_dim * dtype_bytes
```

CFR-Atlas bounds the resident part:

```text
hot_cache_budget + scratch_page + token_ledger + metadata
```

The project directly controls `hot_cache_budget` and `scratch_page`. The host
inference runtime owns token history, embeddings, weights, graph execution and
backend-specific replay state.

## 5. Benchmark note

The included benchmark is intentionally small and deterministic. It compares a
resident full-KV path against CFR page regeneration for one attention scope. It
is not a full LLM benchmark, but it proves the core property: the folded output
can match the baseline while using a bounded resident K/V working set.

Representative deterministic output:

```text
context_tokens=65536
head_dim=64
page_tokens=512
estimated_memory_reduction=128.00x
max_abs_diff=0e0
```
