// Copyright Nixort <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
//! Exercise the typed public-Ollama integration against a running local service.
//!
//! Run with `OLLAMA_MODEL=qwen2.5:0.5b cargo run -p cfr-atlas-ollama --example ollama_public_api`.

use cfr_atlas_ollama::{ExactKvAccess, GenerateRequest, OllamaClient, OllamaError};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen2.5:0.5b".to_owned());
    let client = OllamaClient::default();
    let models = client.list_models()?;
    if !models.iter().any(|candidate| candidate.name == model) {
        return Err(format!("Ollama model {model:?} is not installed").into());
    }

    let record = client.show_model(&model)?;
    println!("model: {}", record.requested_model);
    println!("digest: {}", record.digest.as_deref().unwrap_or("unknown"));
    println!("family: {}", record.family.as_deref().unwrap_or("unknown"));
    println!("context length: {:?}", record.topology.context_length);

    let response = client.generate(&GenerateRequest {
        model,
        prompt: "Reply with exactly this token and no explanation: CFR_OLLAMA_PUBLIC_API_OK."
            .to_owned(),
        keep_alive: Some("0".to_owned()),
        num_ctx: Some(2048),
    })?;
    println!("generation completed: {}", response.done);
    println!("generated tokens: {:?}", response.eval_count);
    println!("response: {}", response.response.trim());

    assert_eq!(
        client.exact_kv_access(),
        ExactKvAccess::UnavailableThroughPublicApi
    );
    match client.require_exact_kv_access() {
        Err(OllamaError::ExactKvAccessUnavailable) => {
            println!("exact K/V: unavailable through the public API (expected)");
        }
        Err(error) => return Err(Box::new(error)),
        Ok(()) => return Err("public Ollama must not enable exact K/V access".into()),
    }
    Ok(())
}
