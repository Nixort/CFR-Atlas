// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 7 july 2026

//! Minimal `CFR-Atlas` CPU example using a deterministic toy backend.
//!
//! The example demonstrates hot-page admission, exact regeneration and folded
//! attention without depending on a concrete `LLM` runtime.

use cfr_atlas::prelude::*;
use std::ops::Range;

struct ToyBackend;

impl KvRegenerator for ToyBackend {
    fn regenerate_page(
        &self,
        key: PageKey,
        token_range: Range<usize>,
        head_dim: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<()> {
        let tokens = checked_range_len("toy range length", &token_range)?;
        let expected = checked_matrix_len("toy output length", tokens, head_dim)?;
        expect_len("toy K output", expected, k_out.len())?;
        expect_len("toy V output", expected, v_out.len())?;

        for (local_t, token) in token_range.enumerate() {
            let row = checked_row_range("toy row range", local_t, head_dim, k_out.len())?;
            for (dim, offset) in row.enumerate() {
                let layer = u32_to_f32_checked("toy layer index must fit exact f32", key.layer)?;
                let head = u32_to_f32_checked("toy head index must fit exact f32", key.head)?;
                let token = usize_to_f32_checked("toy token index must fit exact f32", token)?;
                let dim = usize_to_f32_checked("toy dim index must fit exact f32", dim)?;
                let x = layer.mul_add(0.01, head.mul_add(0.02, token.mul_add(0.001, dim * 0.003)));
                k_out[offset] = x.sin();
                v_out[offset] = (x + 0.5).cos();
            }
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let head_dim = 64;
    let context_tokens = 4096;
    let page_tokens = 256;
    let hot_cache_bytes = checked_mul(
        "toy hot-cache bytes",
        checked_kv_bytes("toy KV bytes", page_tokens, head_dim)?,
        4,
    )?;

    let config = Config::builder(page_tokens, head_dim)
        .hot_cache_bytes(hot_cache_bytes)
        .admit_regenerated_pages(true)
        .build()?;

    let backend = ToyBackend;
    let mut atlas = CfrAtlas::new(config)?;
    let policy = KeepRecent {
        recent_tokens: 1024,
    };

    let query: Vec<f32> = (0..head_dim)
        .map(|i| {
            let i = usize_to_f32_checked("toy query index must fit exact f32", i)?;
            Ok((i * 0.01).sin())
        })
        .collect::<Result<_>>()?;
    let mut output = vec![0.0; head_dim];

    atlas.attend_exact_with_policy(
        &backend,
        &policy,
        AttentionRequest::new(0, 0, &query, context_tokens),
        &mut output,
    )?;

    println!("first_output_values={:?}", &output[..8]);
    println!("stats={:?}", atlas.stats());
    Ok(())
}
