// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! `Phase 3` benchmark matrix for page tuning and bounded-residency estimates.

use cfr_atlas::prelude::*;
use std::time::Instant;

fn run_case(
    context_tokens: usize,
    head_dim: usize,
    page_tokens: usize,
    hot_cache_budget_bytes: usize,
) -> Result<()> {
    let baseline_bytes = checked_kv_bytes("bench matrix baseline bytes", context_tokens, head_dim)?;
    let scratch_bytes = checked_kv_bytes("bench matrix scratch bytes", page_tokens, head_dim)?;
    let input = PageTuningInput::new(head_dim, context_tokens)
        .page_token_bounds(64, page_tokens.max(64))
        .max_scratch_bytes(scratch_bytes.max(1));
    let tuned = PageSizeTuner::tune(input)?;
    let started = Instant::now();
    let synthetic_scan = tuned
        .candidates
        .iter()
        .filter(|candidate| candidate.fits_scratch_budget)
        .count();
    let elapsed = started.elapsed();
    let reduction = usize_to_f64_checked("bench matrix baseline f64", baseline_bytes)?
        / usize_to_f64_checked("bench matrix scratch f64", scratch_bytes)?;

    println!(
        "context_tokens={context_tokens},head_dim={head_dim},page_tokens={page_tokens},hot_cache_budget_bytes={hot_cache_budget_bytes},baseline_kv_bytes={baseline_bytes},scratch_kv_bytes={scratch_bytes},estimated_reduction={reduction:.2},tuned_page_tokens={},candidates={},scan_us={}",
        tuned.page_tokens,
        synthetic_scan,
        elapsed.as_micros()
    );
    Ok(())
}

fn main() -> Result<()> {
    let contexts = [4096usize, 16_384, 65_536];
    let head_dims = [64usize, 128];
    let page_tokens = [128usize, 512, 1024];
    let hot_cache_budgets = [0usize, 64 << 20];
    println!("context_tokens,head_dim,page_tokens,hot_cache_budget_bytes,baseline_kv_bytes,scratch_kv_bytes,estimated_reduction,tuned_page_tokens,candidates,scan_us");
    for context in contexts {
        for head_dim in head_dims {
            for page in page_tokens {
                for budget in hot_cache_budgets {
                    run_case(context, head_dim, page, budget)?;
                }
            }
        }
    }
    Ok(())
}
