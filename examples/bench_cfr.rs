// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 8 july 2026

//! Small `CFR-Atlas` benchmark comparing resident `KV` memory with streaming regeneration.
//!
//! The benchmark reports memory reduction, execution time and maximum absolute
//! difference against a deterministic full-`KV` baseline.

use cfr_atlas::prelude::*;
use std::ops::Range;
use std::time::Instant;

struct DeterministicBackend;

impl DeterministicBackend {
    fn kv(layer: u32, head: u32, token: usize, dim: usize, value: bool) -> Result<f32> {
        let layer = u32_to_f32_checked("benchmark layer index must fit exact f32", layer)?;
        let head = u32_to_f32_checked("benchmark head index must fit exact f32", head)?;
        let token = usize_to_f32_checked("benchmark token index must fit exact f32", token)?;
        let dim = usize_to_f32_checked("benchmark dim index must fit exact f32", dim)?;
        let bias = if value { 0.61 } else { 0.13 };
        let phase = layer.mul_add(
            0.031,
            head.mul_add(0.017, token.mul_add(0.0007, dim.mul_add(0.013, bias))),
        );
        Ok(phase.sin())
    }
}

impl KvRegenerator for DeterministicBackend {
    fn regenerate_page(
        &self,
        key: PageKey,
        token_range: Range<usize>,
        head_dim: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<()> {
        let tokens = checked_range_len("benchmark range length", &token_range)?;
        let expected = checked_matrix_len("benchmark output length", tokens, head_dim)?;
        expect_len("benchmark K output", expected, k_out.len())?;
        expect_len("benchmark V output", expected, v_out.len())?;

        for (local_t, token) in token_range.enumerate() {
            let row = checked_row_range("benchmark row range", local_t, head_dim, k_out.len())?;
            for (dim, offset) in row.enumerate() {
                k_out[offset] = Self::kv(key.layer, key.head, token, dim, false)?;
                v_out[offset] = Self::kv(key.layer, key.head, token, dim, true)?;
            }
        }
        Ok(())
    }
}

fn baseline(
    backend: &DeterministicBackend,
    query: &[f32],
    context_tokens: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Vec<f32>> {
    let len = checked_matrix_len("benchmark baseline length", context_tokens, head_dim)?;
    let mut k = vec![0.0; len];
    let mut v = vec![0.0; len];
    backend.regenerate_page(
        PageKey::new(0, 0, 0),
        0..context_tokens,
        head_dim,
        &mut k,
        &mut v,
    )?;

    let mut reducer = FoldedAttention::new(head_dim, scale)?;
    reducer.consume_page(query, &k, &v, context_tokens)?;
    let mut out = vec![0.0; head_dim];
    reducer.finish_into(&mut out)?;
    wipe_f32(&mut k);
    wipe_f32(&mut v);
    Ok(out)
}

fn cfr(
    backend: &DeterministicBackend,
    query: &[f32],
    context_tokens: usize,
    head_dim: usize,
    page_tokens: usize,
) -> Result<(Vec<f32>, CfrStatsSnapshot)> {
    let config = Config::builder(page_tokens, head_dim)
        .hot_cache_bytes(0)
        .admit_regenerated_pages(false)
        .build()?;
    let mut atlas = CfrAtlas::new(config)?;
    let mut out = vec![0.0; head_dim];
    atlas.attend_exact(
        backend,
        AttentionRequest::new(0, 0, query, context_tokens),
        &mut out,
    )?;
    Ok((out, atlas.stats()))
}

fn parse_arg(args: &[String], index: usize, default: usize) -> usize {
    let Some(value) = args.get(index) else {
        return default;
    };
    value.parse::<usize>().map_or(default, |parsed| parsed)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let context_tokens = parse_arg(&args, 1, 65_536);
    let head_dim = parse_arg(&args, 2, 64);
    let page_tokens = parse_arg(&args, 3, 512);

    let backend = DeterministicBackend;
    let query: Vec<f32> = (0..head_dim)
        .map(|i| {
            let i = usize_to_f32_checked("benchmark query index must fit exact f32", i)?;
            Ok((i * 0.01).cos())
        })
        .collect::<Result<_>>()?;
    let head_dim_f64 = usize_to_f64_checked("benchmark head_dim must fit exact f64", head_dim)?;
    let scale = f64_to_f32_checked("benchmark scale must fit f32", 1.0 / head_dim_f64.sqrt())?;

    if context_tokens == 0 || head_dim == 0 || page_tokens == 0 {
        return Err(CfrError::InvalidConfig(
            "context_tokens, head_dim and page_tokens must be non-zero",
        ));
    }

    let baseline_bytes = checked_kv_bytes("benchmark baseline KV bytes", context_tokens, head_dim)?;
    let cfr_scratch_bytes = checked_kv_bytes("benchmark scratch KV bytes", page_tokens, head_dim)?;

    let t0 = Instant::now();
    let b = baseline(&backend, &query, context_tokens, head_dim, scale)?;
    let baseline_time = t0.elapsed();

    let t1 = Instant::now();
    let (c, stats) = cfr(&backend, &query, context_tokens, head_dim, page_tokens)?;
    let cfr_time = t1.elapsed();

    let max_abs_diff = max_abs_diff_finite("benchmark output diff", &b, &c)?;

    println!("context_tokens={context_tokens}");
    println!("head_dim={head_dim}");
    println!("page_tokens={page_tokens}");
    println!("baseline_kv_bytes={baseline_bytes}");
    println!("cfr_scratch_bytes={cfr_scratch_bytes}");
    println!(
        "estimated_memory_reduction={:.2}x",
        usize_to_f64_checked("baseline bytes must fit exact f64", baseline_bytes)?
            / usize_to_f64_checked("scratch bytes must fit exact f64", cfr_scratch_bytes)?
    );
    println!(
        "baseline_time_ms={:.3}",
        baseline_time.as_secs_f64() * 1000.0
    );
    println!("cfr_time_ms={:.3}", cfr_time.as_secs_f64() * 1000.0);
    println!("max_abs_diff={max_abs_diff:e}");
    println!("stats={stats:?}");

    Ok(())
}
