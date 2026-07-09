// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! Long-context validation example.

use cfr_atlas::prelude::*;
use cfr_atlas_backend_ref::ReferenceBackend;

fn main() -> Result<()> {
    let prompt = regression_corpus()?
        .into_iter()
        .find(|case| case.shape() == PromptShape::Long)
        .ok_or(CfrError::InvalidLedger("long validation prompt missing"))?;
    let topology = AttentionTopology::gqa(8, 2)?;
    let layers = [0, 1, 2, 3];
    let query_heads = [0, 4, 7];
    let decode_steps = [127, 511, 1_023];
    let head_dim = 32;

    let backend = ReferenceBackend::new(
        prompt.to_ledger()?,
        topology,
        PositionEncoding::None,
        DTypePolicy::f32(),
        4,
        head_dim,
    )?;
    let config = Config::builder(64, head_dim)
        .hot_cache_bytes(0)
        .admit_regenerated_pages(false)
        .build()?;
    let mut atlas = CfrAtlas::new(config)?;
    let projector = DeterministicLogitProjector::new(64, 0.125)?;
    let plan = ValidationPlan::new(&prompt, topology, &layers, &query_heads, &decode_steps)
        .with_tolerances(f32::EPSILON, f32::EPSILON);

    let report = validate_decode_loop(&mut atlas, &backend, &NeverAdmit, &projector, &plan)?;

    println!("validation_prompt={}", report.prompt_shape.as_str());
    println!("validated_steps={}", report.steps.len());
    println!("worst_output_diff={:e}", report.worst_output_diff());
    println!("worst_logit_diff={:e}", report.worst_logit_diff());
    println!(
        "best_memory_reduction={:.2}x",
        report.best_memory_reduction()
    );
    for step in report.steps.iter().take(4) {
        println!(
            "layer={},query_head={},kv_head={},context_tokens={},baseline_kv_bytes={},cfr_resident_bytes={},reduction={:.2}x",
            step.layer,
            step.query_head,
            step.mapping.kv_head,
            step.context_tokens,
            step.memory.baseline_kv_bytes,
            step.memory.cfr_resident_bytes,
            step.memory.estimated_memory_reduction,
        );
    }

    Ok(())
}
