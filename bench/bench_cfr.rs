// Copyright Nixort <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// Reproducible CFR-Atlas reference-workload benchmark.

//! Corpus-backed benchmark for the deterministic CFR-Atlas reference workload.
//!
//! It compares a fully materialized K/V stream with cold CFR paging and a
//! bounded hot-cache configuration. Every CFR timing result is checked against
//! an unmeasured full-KV reference output before it is emitted as CSV.

use cfr_atlas::prelude::*;
use std::env;
use std::fs;
use std::hint::black_box;
use std::ops::Range;
use std::process::ExitCode;
use std::time::Instant;

type AppResult<T> = std::result::Result<T, String>;

#[derive(Debug)]
struct Options {
    corpus_path: String,
    context_tokens: usize,
    head_dim: usize,
    page_tokens: usize,
    hot_cache_bytes: usize,
    recent_tokens: usize,
    runs: usize,
}

#[derive(Debug, Clone, Copy)]
struct TimingSummary {
    median_ms: f64,
    min_ms: f64,
    max_ms: f64,
    contexts_per_second: f64,
}

#[derive(Debug)]
struct MethodResult {
    summary: TimingSummary,
    resident_kv_bytes: usize,
    hot_hits_per_run: u64,
    cold_regenerations_per_run: u64,
    max_abs_diff: f32,
}

struct CorpusBackend<'a> {
    tokens: &'a [u8],
}

impl CorpusBackend<'_> {
    fn value(&self, key: PageKey, token: usize, dimension: usize, is_value: bool) -> Result<f32> {
        let byte = f32::from(self.tokens[token % self.tokens.len()]);
        let lane = usize_to_f32_checked("benchmark dimension", dimension)?;
        let head = u32_to_f32_checked("benchmark head", key.head)?;
        let layer = u32_to_f32_checked("benchmark layer", key.layer)?;
        let bias = if is_value { 0.73 } else { 0.19 };
        Ok((byte.mul_add(
            0.0031,
            lane.mul_add(0.017, head.mul_add(0.031, layer.mul_add(0.047, bias))),
        ))
        .sin())
    }
}

impl KvRegenerator for CorpusBackend<'_> {
    fn regenerate_page(
        &self,
        key: PageKey,
        token_range: Range<usize>,
        head_dim: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<()> {
        let tokens = checked_range_len("benchmark token range", &token_range)?;
        let expected = checked_matrix_len("benchmark page matrix", tokens, head_dim)?;
        expect_len("benchmark K output", expected, k_out.len())?;
        expect_len("benchmark V output", expected, v_out.len())?;
        for (local_token, token) in token_range.enumerate() {
            let row = checked_row_range("benchmark page row", local_token, head_dim, expected)?;
            for (dimension, offset) in row.enumerate() {
                k_out[offset] = self.value(key, token, dimension, false)?;
                v_out[offset] = self.value(key, token, dimension, true)?;
            }
        }
        Ok(())
    }
}

fn parse_options() -> AppResult<Options> {
    let mut options = Options {
        corpus_path: "bench/data/tinyshakespeare.txt".to_owned(),
        context_tokens: 65_536,
        head_dim: 64,
        page_tokens: 512,
        hot_cache_bytes: 8 << 20,
        recent_tokens: 4096,
        runs: 5,
    };
    let arguments: Vec<String> = env::args().skip(1).collect();
    let mut index = 0usize;
    while index < arguments.len() {
        let flag = &arguments[index];
        if flag == "--help" {
            return Err(usage());
        }
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--corpus" => options.corpus_path = value.clone(),
            "--context-tokens" => options.context_tokens = parse_usize(flag, value)?,
            "--head-dim" => options.head_dim = parse_usize(flag, value)?,
            "--page-tokens" => options.page_tokens = parse_usize(flag, value)?,
            "--hot-cache-bytes" => options.hot_cache_bytes = parse_usize(flag, value)?,
            "--recent-tokens" => options.recent_tokens = parse_usize(flag, value)?,
            "--runs" => options.runs = parse_usize(flag, value)?,
            _ => return Err(format!("unknown argument: {flag}\n{}", usage())),
        }
        index = index.saturating_add(2);
    }
    if options.context_tokens == 0
        || options.head_dim == 0
        || options.page_tokens == 0
        || options.runs == 0
    {
        return Err(
            "context tokens, head dimension, page tokens and runs must be non-zero".to_owned(),
        );
    }
    Ok(options)
}

fn parse_usize(flag: &str, value: &str) -> AppResult<usize> {
    value
        .parse::<usize>()
        .map_err(|_| format!("invalid integer for {flag}: {value}"))
}

fn usage() -> String {
    "Usage: cargo run --release --bin bench-cfr -- [options]\n\
     --corpus PATH --context-tokens N --head-dim N --page-tokens N\n\
     --hot-cache-bytes N --recent-tokens N --runs N"
        .to_owned()
}

fn full_kv(
    backend: &CorpusBackend<'_>,
    query: &[f32],
    context_tokens: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Vec<f32>> {
    let len = checked_matrix_len("benchmark full KV matrix", context_tokens, head_dim)?;
    let mut k = vec![0.0; len];
    let mut v = vec![0.0; len];
    backend.regenerate_page(
        PageKey::new(0, 0, 0),
        0..context_tokens,
        head_dim,
        &mut k,
        &mut v,
    )?;
    let mut attention = FoldedAttention::new(head_dim, scale)?;
    attention.consume_page(query, &k, &v, context_tokens)?;
    let mut output = vec![0.0; head_dim];
    attention.finish_into(&mut output)?;
    wipe_f32(&mut k);
    wipe_f32(&mut v);
    Ok(output)
}

fn require_exact(name: &'static str, actual: &[f32], reference: &[f32]) -> Result<f32> {
    let diff = max_abs_diff_finite(name, actual, reference)?;
    if diff == 0.0 {
        Ok(diff)
    } else {
        Err(CfrError::Numeric("benchmark exactness mismatch"))
    }
}

fn summarize(mut samples: Vec<f64>) -> TimingSummary {
    samples.sort_by(f64::total_cmp);
    let median_ms = samples[samples.len() / 2];
    TimingSummary {
        median_ms,
        min_ms: samples[0],
        max_ms: samples[samples.len() - 1],
        contexts_per_second: 1000.0 / median_ms,
    }
}

fn measure_full_kv(
    backend: &CorpusBackend<'_>,
    query: &[f32],
    options: &Options,
    scale: f32,
    reference: &[f32],
) -> Result<MethodResult> {
    let mut samples = Vec::with_capacity(options.runs);
    for _ in 0..options.runs {
        let start = Instant::now();
        let output = full_kv(
            backend,
            black_box(query),
            options.context_tokens,
            options.head_dim,
            scale,
        )?;
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
        require_exact("full-KV self check", &output, reference)?;
    }
    Ok(MethodResult {
        summary: summarize(samples),
        resident_kv_bytes: checked_kv_bytes(
            "full KV bytes",
            options.context_tokens,
            options.head_dim,
        )?,
        hot_hits_per_run: 0,
        cold_regenerations_per_run: 0,
        max_abs_diff: 0.0,
    })
}

fn measure_cfr(
    backend: &CorpusBackend<'_>,
    query: &[f32],
    options: &Options,
    reference: &[f32],
    admit: bool,
) -> Result<MethodResult> {
    let config = Config::builder(options.page_tokens, options.head_dim)
        .hot_cache_bytes(if admit { options.hot_cache_bytes } else { 0 })
        .admit_regenerated_pages(admit)
        .build()?;
    let scratch_bytes = config.kv_page_bytes(options.page_tokens)?;
    let mut atlas = CfrAtlas::new(config)?;
    let policy = KeepRecent {
        recent_tokens: options.recent_tokens,
    };
    let mut warm_output = vec![0.0; options.head_dim];
    if admit {
        atlas.attend_exact_with_policy(
            backend,
            &policy,
            AttentionRequest::new(0, 0, query, options.context_tokens),
            &mut warm_output,
        )?;
    } else {
        atlas.attend_exact(
            backend,
            AttentionRequest::new(0, 0, query, options.context_tokens),
            &mut warm_output,
        )?;
    }
    require_exact("CFR warm-up check", &warm_output, reference)?;
    atlas.reset_counters();

    let mut samples = Vec::with_capacity(options.runs);
    let mut max_abs_diff = 0.0f32;
    for _ in 0..options.runs {
        let mut output = vec![0.0; options.head_dim];
        let start = Instant::now();
        if admit {
            atlas.attend_exact_with_policy(
                backend,
                &policy,
                AttentionRequest::new(0, 0, black_box(query), options.context_tokens),
                &mut output,
            )?;
        } else {
            atlas.attend_exact(
                backend,
                AttentionRequest::new(0, 0, black_box(query), options.context_tokens),
                &mut output,
            )?;
        }
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
        max_abs_diff = max_abs_diff.max(require_exact("CFR exactness", &output, reference)?);
    }
    let stats = atlas.stats();
    let runs = usize_to_u64_saturating(options.runs);
    let resident_kv_bytes = scratch_bytes.saturating_add(stats.hot_cache_bytes);
    Ok(MethodResult {
        summary: summarize(samples),
        resident_kv_bytes,
        hot_hits_per_run: stats.hot_hits / runs,
        cold_regenerations_per_run: stats.cold_regenerations / runs,
        max_abs_diff,
    })
}

fn print_result(method: &str, options: &Options, full_kv_bytes: usize, result: &MethodResult) {
    println!(
        "{method},tinyshakespeare,{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{:.9e}",
        options.context_tokens,
        options.head_dim,
        options.page_tokens,
        options.hot_cache_bytes,
        options.recent_tokens,
        options.runs,
        result.summary.median_ms,
        result.summary.min_ms,
        result.summary.max_ms,
        result.summary.contexts_per_second,
        full_kv_bytes,
        result.resident_kv_bytes,
        result.hot_hits_per_run,
        result.cold_regenerations_per_run,
        result.max_abs_diff,
    );
}

fn run() -> AppResult<()> {
    let options = parse_options()?;
    let corpus = fs::read(&options.corpus_path)
        .map_err(|error| format!("cannot read {}: {error}", options.corpus_path))?;
    if corpus.is_empty() {
        return Err(format!("corpus is empty: {}", options.corpus_path));
    }
    let backend = CorpusBackend { tokens: &corpus };
    let query: Vec<f32> = (0..options.head_dim)
        .map(|dimension| {
            Ok((usize_to_f32_checked("benchmark query dimension", dimension)? * 0.013).cos())
        })
        .collect::<Result<_>>()
        .map_err(|error| error.to_string())?;
    let head_dim = usize_to_f64_checked("benchmark head dimension", options.head_dim)
        .map_err(|error| error.to_string())?;
    let scale = f64_to_f32_checked("benchmark attention scale", 1.0 / head_dim.sqrt())
        .map_err(|error| error.to_string())?;
    let reference = full_kv(
        &backend,
        &query,
        options.context_tokens,
        options.head_dim,
        scale,
    )
    .map_err(|error| error.to_string())?;
    let full = measure_full_kv(&backend, &query, &options, scale, &reference)
        .map_err(|error| error.to_string())?;
    let cold = measure_cfr(&backend, &query, &options, &reference, false)
        .map_err(|error| error.to_string())?;
    let hot = measure_cfr(&backend, &query, &options, &reference, true)
        .map_err(|error| error.to_string())?;
    let full_kv_bytes = full.resident_kv_bytes;

    println!("method,source,context_tokens,head_dim,page_tokens,hot_cache_bytes,recent_tokens,runs,median_ms,min_ms,max_ms,contexts_per_second,full_kv_bytes,cfr_resident_kv_bytes,hot_hits_per_run,cold_regenerations_per_run,max_abs_diff");
    print_result("full_kv", &options, full_kv_bytes, &full);
    print_result("cfr_cold", &options, full_kv_bytes, &cold);
    print_result("cfr_hot", &options, full_kv_bytes, &hot);
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("bench-cfr: {error}");
            ExitCode::FAILURE
        }
    }
}
