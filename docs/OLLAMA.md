# Ollama integration boundary

`cfr-atlas-ollama` is an **optional workspace crate** for inspecting a reachable Ollama service and using selected documented public API operations. It is an integration and evidence-collection layer; it is **not** a `KvRegenerator` implementation and does not enable CFR virtual K/V execution.

> **Fail-closed rule.** The public Ollama API wrappers in this repository can discover model metadata, generate text, and produce embeddings. They cannot prove or replay the per-layer K/V rows required by the CFR-Atlas exactness contract. Therefore `OllamaClient::require_exact_kv_access()` always returns `OllamaError::ExactKvAccessUnavailable`.

The standard Ollama local API is served at `http://localhost:11434/api`. The crate's default client targets the corresponding base URL `http://localhost:11434`. [1]

## Supported surface

| API surface | Crate API | Purpose in a CFR workflow | Exact K/V result |
|---|---|---|---|
| `GET /api/tags` | `OllamaClient::list_models` | List installed names, digests, families and quantization metadata | No K/V data [2] |
| `POST /api/show` | `OllamaClient::show_model` | Record capabilities, raw `model_info` and parsed attention topology | No K/V data |
| `POST /api/generate` with `stream: false` | `OllamaClient::generate` | Exercise ordinary non-streaming generation and record token counts | No K/V data [3] |
| `POST /api/embed` | `OllamaClient::embed` | Produce embeddings for ordinary application use | No K/V data [4] |

`OllamaModelRecord` preserves the requested and reported identity, digest, format, family, quantization label, capabilities, parsed block/head/context/RoPE topology, and raw `model_info`. This is useful validation evidence for a future adapter, but topology metadata must not be confused with K/V tensors or token-position replay.

## Run model discovery

Install and start Ollama, pull at least one model, then execute the example from the repository root:

```sh
cargo run -p cfr-atlas-ollama --example ollama_discovery
```

The example lists local models, queries the first model's record, prints topology fields when available, and reports the explicit exact-K/V rejection. It does not run an attention request through CFR-Atlas.

The optional crate provides a dependency-free blocking transport for the standard local HTTP endpoint. An HTTPS proxy, remote endpoint, authentication layer, or test double should implement `OllamaTransport` and be supplied through `OllamaClient::with_transport`. Tests use this seam and do not require a running Ollama process.

## Why this is not an adapter

The core adapter contract requires a backend to replay the exact K/V rows for a requested `(layer, K/V head, token range)`, using the same token history, absolute positions, positional policy, head mapping, and storage rounding as the baseline path. It also requires output buffers to be fully overwritten on success. See [`ADAPTERS.md`](ADAPTERS.md) for the complete conformance sequence.

The public endpoints exposed here return model metadata, generated text, embeddings, and aggregate token counts. They do not supply the per-layer K/V buffers, a stable token-to-position trace, or a mechanism to reproduce a historical page from the runtime's normal forward path. A generation response is therefore not a substitute for `KvRegenerator` conformance. Keeping virtual K/V disabled prevents a superficially working but semantically unsupported integration.

## Path to a conformant extension

A future Ollama-specific sidecar or native runtime extension may enable exact CFR only if it provides a versioned, explicit interface for all of the following requirements:

| Required sidecar capability | CFR reason | Required validation |
|---|---|---|
| Per-layer K/V export and page replay | Implements `KvRegenerator` over requested half-open token ranges | Stored-K/V versus replayed-K/V comparison across pages and layers |
| Token IDs and absolute positions | Preserves the baseline causal history | Initial, middle and final page tests |
| MHA/MQA/GQA topology mapping | Ensures `PageKey.head` identifies the K/V head | Mapping tests for each supported topology |
| RoPE/ALiBi and storage-dtype policy | Preserves model semantics before attention consumes K/V | Position and rounding conformance tests |
| Runtime/build/model identity | Makes results reproducible and regression-safe | Versioned record paired with a model digest |
| Logit or output projection hook | Confirms page conformance survives real attention use | Long-context decode-loop validation |

The extension must remain disabled until it passes the general adapter gate and publishes its validation policy, environment, model digest, runtime version, workload, and resident-memory accounting. The integration should keep returning the same fail-closed error for every unsupported runtime configuration.

## Minimal application use

The public wrappers are useful even without exact K/V execution:

```rust
use cfr_atlas_ollama::{GenerateRequest, OllamaClient};

let client = OllamaClient::default();
let answer = client.generate(&GenerateRequest {
    model: "gemma4".to_owned(),
    prompt: "Summarize the adapter boundary in one sentence.".to_owned(),
    keep_alive: Some("5m".to_owned()),
    num_ctx: Some(8_192),
})?;
assert!(answer.done);
# Ok::<(), cfr_atlas_ollama::OllamaError>(())
```

This snippet uses a documented non-streaming generation request; it establishes only that the public generation path works. It makes no claim about K/V export, cache residency, attention equivalence, or model serving performance. [3]

## References

[1]: https://docs.ollama.com/api "Ollama API introduction and base URLs"
[2]: https://docs.ollama.com/api/tags "Ollama API: list models"
[3]: https://docs.ollama.com/api/generate "Ollama API: generate a response"
[4]: https://docs.ollama.com/api/embed "Ollama API: generate embeddings"
