// Copyright Nixort <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
//! Discover a local Ollama model without enabling CFR virtual K/V execution.

use cfr_atlas_ollama::{OllamaClient, OllamaError};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = OllamaClient::default();
    let models = client.list_models()?;
    let Some(model) = models.first() else {
        println!("No local Ollama models are installed.");
        return Ok(());
    };

    let record = client.show_model(&model.name)?;
    println!("model: {}", record.requested_model);
    println!("family: {}", record.family.as_deref().unwrap_or("unknown"));
    println!("context length: {:?}", record.topology.context_length);
    println!("exact K/V access: {:?}", record.exact_kv_access);

    match client.require_exact_kv_access() {
        Err(OllamaError::ExactKvAccessUnavailable) => println!(
            "CFR virtual K/V remains disabled: public Ollama does not expose K/V tensors or page replay."
        ),
        Err(error) => return Err(Box::new(error)),
        Ok(()) => unreachable!("the public Ollama adapter must fail closed"),
    }
    Ok(())
}
