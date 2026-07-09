// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! Long-context validation tests.

use cfr_atlas::prelude::*;
use cfr_atlas_backend_ref::ReferenceBackend;

fn backend_from_prompt(
    prompt: &PromptCase,
    topology: AttentionTopology,
    layers: u32,
    head_dim: usize,
) -> Result<ReferenceBackend> {
    ReferenceBackend::new(
        prompt.to_ledger()?,
        topology,
        PositionEncoding::None,
        DTypePolicy::f32(),
        layers,
        head_dim,
    )
}

fn config(page_tokens: usize, head_dim: usize, hot_cache_bytes: usize) -> Result<Config> {
    Config::builder(page_tokens, head_dim)
        .hot_cache_bytes(hot_cache_bytes)
        .admit_regenerated_pages(true)
        .build()
}

#[test]
fn logit_level_step_validation_matches_full_kv() -> Result<()> {
    let prompt = regression_corpus()?
        .into_iter()
        .find(|case| case.shape() == PromptShape::Short)
        .ok_or(CfrError::InvalidLedger("short validation prompt missing"))?;
    let topology = AttentionTopology::mha(4)?;
    let layers = 2;
    let head_dim = 16;
    let backend = backend_from_prompt(&prompt, topology, layers, head_dim)?;
    let mut atlas = CfrAtlas::new(config(4, head_dim, 0)?)?;
    let projector = DeterministicLogitProjector::new(32, 0.25)?;
    let query = deterministic_query(&prompt, 4, 1, 3, head_dim)?;

    let report = validate_decode_step(
        &mut atlas,
        &backend,
        &NeverAdmit,
        &projector,
        StepValidationRequest::new(layers, topology, 1, 3, &query, 5)
            .with_tolerances(f32::EPSILON, f32::EPSILON),
    )?;

    assert!(report.max_abs_output <= f32::EPSILON);
    assert!(report.max_abs_logits <= f32::EPSILON);
    assert_eq!(report.mapping.kv_head, 3);
    Ok(())
}

#[test]
fn decode_loop_covers_layers_heads_and_gqa() -> Result<()> {
    let prompt = PromptCase::new(PromptShape::Code, vec![11, 22, 33, 44, 55, 66, 77, 88])?;
    let topology = AttentionTopology::gqa(8, 2)?;
    let layers = [0, 1];
    let query_heads = [0, 3, 4, 7];
    let decode_steps = [0, 3, 7];
    let head_dim = 16;
    let backend = backend_from_prompt(&prompt, topology, 2, head_dim)?;
    let mut atlas = CfrAtlas::new(config(4, head_dim, 1 << 20)?)?;
    let projector = DeterministicLogitProjector::new(24, 0.125)?;
    let plan = ValidationPlan::new(&prompt, topology, &layers, &query_heads, &decode_steps)
        .with_tolerances(f32::EPSILON, f32::EPSILON);

    let report = validate_decode_loop(
        &mut atlas,
        &backend,
        &KeepRecent { recent_tokens: 16 },
        &projector,
        &plan,
    )?;

    assert_eq!(report.prompt_shape, PromptShape::Code);
    assert_eq!(
        report.steps.len(),
        layers.len() * query_heads.len() * decode_steps.len()
    );
    assert!(report.steps.iter().any(|step| step.mapping.kv_head == 0));
    assert!(report.steps.iter().any(|step| step.mapping.kv_head == 1));
    assert!(report.worst_output_diff() <= f32::EPSILON);
    assert!(report.worst_logit_diff() <= f32::EPSILON);
    Ok(())
}

#[test]
fn long_context_memory_telemetry_reports_reduction() -> Result<()> {
    let prompt = regression_corpus()?
        .into_iter()
        .find(|case| case.shape() == PromptShape::Long)
        .ok_or(CfrError::InvalidLedger("long validation prompt missing"))?;
    let topology = AttentionTopology::gqa(8, 2)?;
    let layers = [0, 1, 2, 3];
    let query_heads = [7];
    let decode_steps = [1_023];
    let head_dim = 32;
    let backend = backend_from_prompt(&prompt, topology, 4, head_dim)?;
    let mut atlas = CfrAtlas::new(config(64, head_dim, 0)?)?;
    let projector = DeterministicLogitProjector::new(16, 0.0625)?;
    let plan = ValidationPlan::new(&prompt, topology, &layers, &query_heads, &decode_steps)
        .with_tolerances(f32::EPSILON, f32::EPSILON);

    let report = validate_decode_loop(&mut atlas, &backend, &NeverAdmit, &projector, &plan)?;

    assert!(report.best_memory_reduction() >= 10.0);
    assert!(report
        .steps
        .iter()
        .all(|step| step.memory.baseline_kv_bytes > step.memory.cfr_resident_bytes));
    Ok(())
}

#[test]
fn regression_corpus_contains_all_prompt_shapes() -> Result<()> {
    let corpus = regression_corpus()?;
    for shape in [
        PromptShape::Short,
        PromptShape::Long,
        PromptShape::Repeated,
        PromptShape::Code,
        PromptShape::Dialogue,
    ] {
        assert!(corpus.iter().any(|case| case.shape() == shape));
    }
    assert!(corpus.iter().all(|case| !case.is_empty()));
    Ok(())
}
