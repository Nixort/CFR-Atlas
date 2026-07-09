// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! `Phase 3` performance-layer regression tests.

use cfr_atlas::prelude::*;
use std::ops::Range;

#[derive(Debug, Clone, Copy)]
struct TinyBackend;

impl KvRegenerator for TinyBackend {
    fn regenerate_page(
        &self,
        key: PageKey,
        token_range: Range<usize>,
        head_dim: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<()> {
        let tokens = checked_range_len("performance tiny range", &token_range)?;
        let expected = checked_matrix_len("performance tiny output", tokens, head_dim)?;
        expect_len("performance tiny K", expected, k_out.len())?;
        expect_len("performance tiny V", expected, v_out.len())?;
        for (local_token, token) in token_range.enumerate() {
            let row =
                checked_row_range("performance tiny row", local_token, head_dim, k_out.len())?;
            for (dim, offset) in row.enumerate() {
                let token = usize_to_f32_checked("performance token", token)?;
                let dim = usize_to_f32_checked("performance dim", dim)?;
                let layer = u32_to_f32_checked("performance layer", key.layer)?;
                let head = u32_to_f32_checked("performance head", key.head)?;
                k_out[offset] = token.mul_add(0.01, dim.mul_add(0.1, layer));
                v_out[offset] = token.mul_add(0.02, dim.mul_add(0.2, head));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct FailingBackend;

impl KvRegenerator for FailingBackend {
    fn regenerate_page(
        &self,
        _key: PageKey,
        _token_range: Range<usize>,
        _head_dim: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<()> {
        wipe_f32(k_out);
        wipe_f32(v_out);
        Err(CfrError::Regenerator(
            "planned regeneration failure".to_owned(),
        ))
    }
}

#[test]
fn dot_kernel_matches_scalar() -> Result<()> {
    let lhs = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let rhs = [0.5f32, 0.25, 0.125, 0.0625, 1.0, 2.0, 3.0, 4.0, 5.0];
    let scalar = DotProductKernel::Scalar.dot(&lhs, &rhs)?;
    let auto = DotProductKernel::AutoVectorized.dot(&lhs, &rhs)?;
    assert!((scalar - auto).abs() <= f64::EPSILON);
    Ok(())
}

#[test]
fn page_tuner_selects_budget_safe_candidate() -> Result<()> {
    let input = PageTuningInput::new(64, 8192)
        .page_token_bounds(64, 2048)
        .cache_bytes(512 * 1024, 8 * 1024 * 1024)
        .max_scratch_bytes(512 * 1024);
    let result = PageSizeTuner::tune(input)?;
    assert!(result.page_tokens >= 64);
    assert!(result.kv_bytes <= input.max_scratch_bytes);
    assert!(!result.candidates.is_empty());
    Ok(())
}

#[test]
fn hot_cache_respects_layer_budget() -> Result<()> {
    let mut cache = HotCache::new(1 << 20);
    let key0 = PageKey::new(0, 0, 0);
    let key1 = PageKey::new(0, 0, 64);
    let tokens = 64;
    let head_dim = 8;
    let len = checked_matrix_len("layer budget len", tokens, head_dim)?;
    let k = vec![1.0; len];
    let v = vec![2.0; len];
    let bytes = checked_kv_bytes("layer budget bytes", tokens, head_dim)?;
    let _old_layer_budget = cache.set_layer_budget(0, bytes);
    assert!(cache.insert(key0, tokens, head_dim, &k, &v)?);
    assert!(cache.insert(key1, tokens, head_dim, &k, &v)?);
    assert_eq!(cache.layer_used_bytes(0), bytes);
    assert!(cache.page_tokens(&key0).is_none() || cache.page_tokens(&key1).is_none());
    Ok(())
}

#[test]
fn telemetry_policy_uses_context() {
    let stats = CfrStatsSnapshot {
        hot_hits: 0,
        cold_regenerations: 8,
        cache_admissions: 0,
        cache_admission_rejections: 0,
        cache_evictions: 0,
        consumed_tokens: 512,
        hot_cache_bytes: 0,
        hot_cache_pages: 0,
    };
    let policy = TelemetryResidencyPolicy::balanced(256);
    let context = ResidencyContext {
        key: PageKey::new(0, 0, 768),
        page_tokens: 128,
        context_tokens: 1024,
        stats: &stats,
        hot_cache_max_bytes: 1 << 20,
        hot_cache_used_bytes: 0,
    };
    assert_eq!(
        policy.decide_with_context(&context),
        ResidencyDecision::Admit
    );
}

#[test]
fn double_buffered_pipeline_regenerates_alternating_pages() -> Result<()> {
    let backend = TinyBackend;
    let mut pipeline = DoubleBufferedPipeline::new();
    let first = pipeline.regenerate_next(&backend, PageKey::new(0, 0, 0), 0..4, 8)?;
    assert_eq!(first.tokens(), 4);
    let second = pipeline.regenerate_next(&backend, PageKey::new(0, 0, 4), 4..8, 8)?;
    assert_eq!(second.key(), PageKey::new(0, 0, 4));
    assert_eq!(second.k().len(), 32);
    Ok(())
}

#[test]
fn double_buffered_pipeline_is_transactional_on_failure() -> Result<()> {
    let mut pipeline = DoubleBufferedPipeline::new();
    let first_key = PageKey::new(0, 0, 0);
    pipeline.regenerate_next(&TinyBackend, first_key, 0..4, 8)?;

    let result = pipeline.regenerate_next(&FailingBackend, PageKey::new(0, 0, 4), 4..8, 8);
    assert!(matches!(result, Err(CfrError::Regenerator(_))));
    assert_eq!(pipeline.buffers()[0].key(), first_key);
    assert_eq!(pipeline.buffers()[0].tokens(), 4);
    assert_eq!(pipeline.buffers()[1].tokens(), 0);

    let next_key = PageKey::new(0, 0, 8);
    let next = pipeline.regenerate_next(&TinyBackend, next_key, 8..12, 8)?;
    assert_eq!(next.key(), next_key);
    Ok(())
}

#[test]
fn thread_pool_executor_preserves_order() -> Result<()> {
    let executor = ThreadPoolExecutor::new(ThreadPoolConfig::new(2)?);
    let jobs: Vec<_> = (0..6).map(|value| move || Ok(value * value)).collect();
    let values = executor.run(jobs)?;
    assert_eq!(values, vec![0, 1, 4, 9, 16, 25]);
    Ok(())
}

#[test]
fn atlas_exposes_layer_budget_and_kernel() -> Result<()> {
    let config = Config::builder(64, 8)
        .hot_cache_bytes(1 << 20)
        .admit_regenerated_pages(true)
        .build()?;
    let mut atlas = CfrAtlas::new(config)?;
    atlas.set_dot_kernel(DotProductKernel::Scalar);
    assert_eq!(atlas.dot_kernel(), DotProductKernel::Scalar);
    let evicted = atlas.set_layer_hot_cache_bytes(0, 4096);
    assert_eq!(evicted, 0);
    assert_eq!(atlas.layer_hot_cache_bytes(0), Some(4096));
    Ok(())
}

#[test]
fn page_tuner_handles_context_smaller_than_min_bound() -> Result<()> {
    let input = PageTuningInput::new(32, 16).page_token_bounds(64, 256);
    let result = PageSizeTuner::tune(input)?;
    assert_eq!(result.page_tokens, 16);
    assert_eq!(result.candidates.len(), 1);
    Ok(())
}
