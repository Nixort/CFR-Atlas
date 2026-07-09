// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! Integration tests proving exact equality with baseline attention for deterministic regeneration.
//!
//! These tests are the seal for the core invariant: cache residency may change
//! performance, but it must not change the attention output.

use cfr_atlas::prelude::*;
use std::ops::Range;

#[derive(Debug, Clone)]
struct ToyRegenerator;

impl ToyRegenerator {
    fn kv_value(layer: u32, head: u32, token: usize, dim: usize, value: bool) -> Result<f32> {
        let layer = u32_to_f32_checked("test layer index must fit exact f32", layer)?;
        let head = u32_to_f32_checked("test head index must fit exact f32", head)?;
        let token = usize_to_f32_checked("test token index must fit exact f32", token)?;
        let dim = usize_to_f32_checked("test dim index must fit exact f32", dim)?;
        let bias = if value { 0.37 } else { 0.11 };
        let phase = layer.mul_add(
            0.031,
            head.mul_add(0.017, token.mul_add(0.007, dim.mul_add(0.013, bias))),
        );
        Ok(phase.sin())
    }
}

impl KvRegenerator for ToyRegenerator {
    fn regenerate_page(
        &self,
        key: PageKey,
        token_range: Range<usize>,
        head_dim: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<()> {
        let tokens = checked_range_len("test range length", &token_range)?;
        let expected = checked_matrix_len("test output length", tokens, head_dim)?;
        expect_len("test K output", expected, k_out.len())?;
        expect_len("test V output", expected, v_out.len())?;

        for (local_t, token) in token_range.enumerate() {
            let row = checked_row_range("test row range", local_t, head_dim, k_out.len())?;
            for (dim, offset) in row.enumerate() {
                k_out[offset] = Self::kv_value(key.layer, key.head, token, dim, false)?;
                v_out[offset] = Self::kv_value(key.layer, key.head, token, dim, true)?;
            }
        }
        Ok(())
    }
}

fn baseline_attention(
    regenerator: &ToyRegenerator,
    layer: u32,
    head: u32,
    query: &[f32],
    context_tokens: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Vec<f32>> {
    let key = PageKey::new(layer, head, 0);
    let len = checked_matrix_len("baseline length", context_tokens, head_dim)?;
    let mut k = vec![0.0; len];
    let mut v = vec![0.0; len];
    regenerator.regenerate_page(key, 0..context_tokens, head_dim, &mut k, &mut v)?;

    let mut folded = FoldedAttention::new(head_dim, scale)?;
    folded.consume_page(query, &k, &v, context_tokens)?;

    let mut out = vec![0.0; head_dim];
    folded.finish_into(&mut out)?;
    wipe_f32(&mut k);
    wipe_f32(&mut v);
    Ok(out)
}

#[test]
fn cfr_matches_baseline_without_hot_cache() -> Result<()> {
    let head_dim = 32;
    let context_tokens = 257;
    let config = Config::builder(64, head_dim)
        .hot_cache_bytes(0)
        .admit_regenerated_pages(false)
        .build()?;

    let regenerator = ToyRegenerator;
    let query: Vec<f32> = (0..head_dim)
        .map(|i| {
            let i = usize_to_f32_checked("test query index must fit exact f32", i)?;
            Ok((i * 0.019).cos())
        })
        .collect::<Result<_>>()?;

    let expected = baseline_attention(
        &regenerator,
        3,
        2,
        &query,
        context_tokens,
        head_dim,
        config.scale,
    )?;

    let mut atlas = CfrAtlas::new(config)?;
    let mut actual = vec![0.0; head_dim];
    atlas.attend_exact(
        &regenerator,
        AttentionRequest::new(3, 2, &query, context_tokens),
        &mut actual,
    )?;

    for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff <= 1e-6,
            "dim {i}: actual {a}, expected {b}, diff {diff}"
        );
    }
    Ok(())
}

#[test]
fn cfr_uses_bounded_hot_cache() -> Result<()> {
    let head_dim = 16;
    let page_tokens = 32;
    let one_page_bytes = checked_kv_bytes("test one page bytes", page_tokens, head_dim)?;
    let config = Config::builder(page_tokens, head_dim)
        .hot_cache_bytes(one_page_bytes)
        .admit_regenerated_pages(true)
        .build()?;

    let regenerator = ToyRegenerator;
    let policy = KeepRecent { recent_tokens: 128 };
    let query: Vec<f32> = (0..head_dim)
        .map(|i| {
            let i = usize_to_f32_checked("test query index must fit exact f32", i)?;
            Ok((i * 0.023).sin())
        })
        .collect::<Result<_>>()?;

    let mut atlas = CfrAtlas::new(config)?;
    let mut out = vec![0.0; head_dim];
    atlas.attend_exact_with_policy(
        &regenerator,
        &policy,
        AttentionRequest::new(0, 0, &query, 256),
        &mut out,
    )?;

    assert!(atlas.hot_cache().used_bytes() <= one_page_bytes);
    assert!(atlas.hot_cache().len() <= 1);
    assert_eq!(atlas.stats().consumed_tokens, 256);
    Ok(())
}

#[test]
fn explicit_hot_page_is_used() -> Result<()> {
    let head_dim = 8;
    let page_tokens = 4;
    let config = Config::new(page_tokens, head_dim, 10_000)?;
    let regenerator = ToyRegenerator;
    let mut atlas = CfrAtlas::new(config)?;

    let key = PageKey::new(0, 0, 0);
    let len = checked_matrix_len("test hot page length", page_tokens, head_dim)?;
    let mut k = vec![0.0; len];
    let mut v = vec![0.0; len];
    regenerator.regenerate_page(key, 0..page_tokens, head_dim, &mut k, &mut v)?;

    assert!(atlas.insert_hot_page(key, page_tokens, &k, &v)?);

    let query = vec![0.1; head_dim];
    let mut out = vec![0.0; head_dim];
    atlas.attend_exact(
        &regenerator,
        AttentionRequest::new(0, 0, &query, page_tokens),
        &mut out,
    )?;

    assert_eq!(atlas.stats().hot_hits, 1);
    assert_eq!(atlas.stats().cold_regenerations, 0);
    Ok(())
}

#[test]
fn config_rejects_overflowing_page_shape() {
    let result = Config::builder(usize::MAX, 2).build();
    assert!(matches!(result, Err(CfrError::CapacityOverflow { .. })));
}

#[test]
fn cache_rejects_zero_token_page() {
    let mut cache = HotCache::new(1024);
    let result = cache.insert(PageKey::new(0, 0, 0), 0, 8, &[], &[]);
    assert!(matches!(result, Err(CfrError::InvalidPage { .. })));
}

#[test]
fn attention_rejects_non_finite_values() -> Result<()> {
    let mut folded = FoldedAttention::new(2, 1.0)?;
    let query = [1.0, 0.0];
    let k = [0.0, 1.0];
    let v = [f32::NAN, 1.0];
    let result = folded.consume_page(&query, &k, &v, 1);
    assert!(matches!(result, Err(CfrError::Numeric(_))));
    Ok(())
}

#[test]
fn stale_partial_hot_page_is_regenerated_not_reused() -> Result<()> {
    let head_dim = 8;
    let page_tokens = 8;
    let one_page_bytes = checked_kv_bytes("stale page bytes", page_tokens, head_dim)?;
    let hot_cache_bytes = checked_mul("stale hot-cache bytes", one_page_bytes, 2)?;
    let config = Config::builder(page_tokens, head_dim)
        .hot_cache_bytes(hot_cache_bytes)
        .admit_regenerated_pages(true)
        .build()?;

    let regenerator = ToyRegenerator;
    let policy = KeepRecent { recent_tokens: 8 };
    let query = vec![0.05; head_dim];
    let mut atlas = CfrAtlas::new(config)?;
    let mut first = vec![0.0; head_dim];
    let mut second = vec![0.0; head_dim];

    atlas.attend_exact_with_policy(
        &regenerator,
        &policy,
        AttentionRequest::new(0, 0, &query, 10),
        &mut first,
    )?;
    atlas.attend_exact_with_policy(
        &regenerator,
        &policy,
        AttentionRequest::new(0, 0, &query, 16),
        &mut second,
    )?;

    assert!(atlas.stats().cache_evictions >= 1);
    assert!(second.iter().all(|value| value.is_finite()));
    Ok(())
}

#[test]
fn folded_attention_invalid_page_does_not_change_state() -> Result<()> {
    let mut folded = FoldedAttention::new(2, 1.0)?;
    let query = [1.0, 0.0];
    let k = [0.0, 1.0];
    let v = [0.25, 0.75];
    folded.consume_page(&query, &k, &v, 1)?;
    let consumed_before = folded.consumed_tokens();

    let bad_v = [f32::NAN, 1.0];
    let result = folded.consume_page(&query, &k, &bad_v, 1);
    assert!(matches!(result, Err(CfrError::Numeric(_))));
    assert_eq!(folded.consumed_tokens(), consumed_before);
    Ok(())
}

#[test]
fn conformance_rejects_non_finite_reference_values() {
    let regenerator = ToyRegenerator;
    let key = PageKey::new(0, 0, 0);
    let reference_k = [f32::NAN, 0.0];
    let reference_v = [0.0, 0.0];
    let result =
        compare_regenerated_page(&regenerator, key, 0..1, 2, &reference_k, &reference_v, 0.0);
    assert!(matches!(result, Err(CfrError::Numeric(_))));
}
