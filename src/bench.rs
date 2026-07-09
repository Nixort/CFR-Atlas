// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! Dependency-free benchmark estimates used by the stabilization harness.
//!
//! The executable examples still measure wall-clock time. This module provides
//! deterministic memory estimates that can be tested in `CI` and reused by release
//! tooling without depending on an external benchmark framework.

use crate::layout::{checked_add, checked_kv_bytes, f64_to_f32_checked, usize_to_f64_checked};
use crate::{CfrError, Result};

/// One benchmark scenario for resident full-`KV` versus bounded `CFR` memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BenchScenario {
    /// Causal context length in tokens.
    pub context_tokens: usize,
    /// Attention head dimension.
    pub head_dim: usize,
    /// `CFR` virtual page size in tokens.
    pub page_tokens: usize,
    /// Resident hot-cache budget counted with the scratch page.
    pub hot_cache_budget_bytes: usize,
}

impl BenchScenario {
    /// Creates a benchmark scenario.
    #[must_use]
    pub const fn new(
        context_tokens: usize,
        head_dim: usize,
        page_tokens: usize,
        hot_cache_budget_bytes: usize,
    ) -> Self {
        Self {
            context_tokens,
            head_dim,
            page_tokens,
            hot_cache_budget_bytes,
        }
    }
}

/// Deterministic memory estimate for one benchmark scenario.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BenchEstimate {
    /// Full-`KV` resident bytes for one layer/head.
    pub baseline_kv_bytes: usize,
    /// Scratch K/V bytes for one `CFR` page.
    pub cfr_scratch_bytes: usize,
    /// Scratch plus configured hot-cache budget.
    pub cfr_resident_budget_bytes: usize,
    /// Baseline bytes divided by bounded `CFR` resident bytes.
    pub estimated_memory_reduction: f32,
}

/// Estimates memory pressure for one scenario.
pub fn estimate_benchmark_memory(scenario: BenchScenario) -> Result<BenchEstimate> {
    if scenario.context_tokens == 0 {
        return Err(CfrError::InvalidConfig(
            "benchmark context_tokens must be non-zero",
        ));
    }
    if scenario.head_dim == 0 {
        return Err(CfrError::InvalidConfig(
            "benchmark head_dim must be non-zero",
        ));
    }
    if scenario.page_tokens == 0 || scenario.page_tokens > scenario.context_tokens {
        return Err(CfrError::InvalidConfig(
            "benchmark page_tokens must be in 1..=context_tokens",
        ));
    }
    let baseline_kv_bytes = checked_kv_bytes(
        "benchmark baseline KV bytes",
        scenario.context_tokens,
        scenario.head_dim,
    )?;
    let cfr_scratch_bytes = checked_kv_bytes(
        "benchmark CFR scratch bytes",
        scenario.page_tokens,
        scenario.head_dim,
    )?;
    let cfr_resident_budget_bytes = checked_add(
        "benchmark CFR resident bytes",
        cfr_scratch_bytes,
        scenario.hot_cache_budget_bytes,
    )?;
    let estimated_memory_reduction = if cfr_resident_budget_bytes == 0 {
        0.0
    } else {
        let baseline = usize_to_f64_checked("benchmark baseline bytes", baseline_kv_bytes)?;
        let resident = usize_to_f64_checked("benchmark resident bytes", cfr_resident_budget_bytes)?;
        f64_to_f32_checked("benchmark memory reduction", baseline / resident)?
    };
    Ok(BenchEstimate {
        baseline_kv_bytes,
        cfr_scratch_bytes,
        cfr_resident_budget_bytes,
        estimated_memory_reduction,
    })
}

/// Returns the default stabilization benchmark scenarios.
#[must_use]
pub fn stabilization_benchmark_scenarios() -> Vec<BenchScenario> {
    let mut scenarios = Vec::new();
    for context_tokens in [4_096, 16_384, 65_536] {
        for head_dim in [64, 128] {
            for page_tokens in [128, 512, 1_024] {
                scenarios.push(BenchScenario::new(context_tokens, head_dim, page_tokens, 0));
                scenarios.push(BenchScenario::new(
                    context_tokens,
                    head_dim,
                    page_tokens,
                    64 << 20,
                ));
            }
        }
    }
    scenarios
}
