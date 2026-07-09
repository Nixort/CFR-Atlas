// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! Page-size autotuning for cache-local CPU regeneration.

use crate::layout::checked_kv_bytes;
use crate::{CfrError, Result};

/// Input constraints for page-size autotuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageTuningInput {
    /// Attention head dimension.
    pub head_dim: usize,
    /// Minimum page size to consider.
    pub min_page_tokens: usize,
    /// Maximum page size to consider.
    pub max_page_tokens: usize,
    /// Estimated private `L2` cache bytes.
    pub l2_bytes: usize,
    /// Estimated shared `L3` cache bytes.
    pub l3_bytes: usize,
    /// Maximum scratch bytes the runtime is willing to reserve per K/V page.
    pub max_scratch_bytes: usize,
    /// Current causal context length.
    pub context_tokens: usize,
}

impl PageTuningInput {
    /// Creates a tuning input with conservative cache defaults.
    #[must_use]
    pub const fn new(head_dim: usize, context_tokens: usize) -> Self {
        Self {
            head_dim,
            min_page_tokens: 64,
            max_page_tokens: 2048,
            l2_bytes: 1 << 20,
            l3_bytes: 32 << 20,
            max_scratch_bytes: 8 << 20,
            context_tokens,
        }
    }

    /// Sets the candidate token range.
    #[must_use]
    pub const fn page_token_bounds(mut self, min_tokens: usize, max_tokens: usize) -> Self {
        self.min_page_tokens = min_tokens;
        self.max_page_tokens = max_tokens;
        self
    }

    /// Sets cache estimates in bytes.
    #[must_use]
    pub const fn cache_bytes(mut self, l2_bytes: usize, l3_bytes: usize) -> Self {
        self.l2_bytes = l2_bytes;
        self.l3_bytes = l3_bytes;
        self
    }

    /// Sets the maximum scratch bytes for one regenerated K/V page.
    #[must_use]
    pub const fn max_scratch_bytes(mut self, bytes: usize) -> Self {
        self.max_scratch_bytes = bytes;
        self
    }
}

/// One candidate considered by the tuner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageTuningCandidate {
    /// Candidate page tokens.
    pub page_tokens: usize,
    /// K/V bytes for one page.
    pub kv_bytes: usize,
    /// Whether the candidate fits the cache-local target.
    pub fits_cache_target: bool,
    /// Whether the candidate fits the configured scratch budget.
    pub fits_scratch_budget: bool,
}

/// Selected page-size plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageTuningResult {
    /// Selected page size.
    pub page_tokens: usize,
    /// Estimated K/V bytes for one selected page.
    pub kv_bytes: usize,
    /// Cache-local byte target used by the tuner.
    pub cache_target_bytes: usize,
    /// All candidates inspected in ascending order.
    pub candidates: Vec<PageTuningCandidate>,
}

/// Conservative page-size tuner for CPU locality.
#[derive(Debug, Clone, Copy, Default)]
pub struct PageSizeTuner;

impl PageSizeTuner {
    /// Tunes page size from cache, scratch and context constraints.
    pub fn tune(input: PageTuningInput) -> Result<PageTuningResult> {
        validate_input(input)?;
        let cache_target_bytes = cache_target(input);
        let mut candidates = Vec::new();
        let mut selected = None;
        let mut page_tokens = input.min_page_tokens.min(input.context_tokens);

        while page_tokens <= input.max_page_tokens && page_tokens <= input.context_tokens {
            let kv_bytes =
                checked_kv_bytes("page tuning candidate bytes", page_tokens, input.head_dim)?;
            let fits_cache_target = kv_bytes <= cache_target_bytes;
            let fits_scratch_budget = kv_bytes <= input.max_scratch_bytes;
            let candidate = PageTuningCandidate {
                page_tokens,
                kv_bytes,
                fits_cache_target,
                fits_scratch_budget,
            };
            if fits_cache_target && fits_scratch_budget {
                selected = Some(candidate);
            }
            candidates.push(candidate);
            let Some(next) = page_tokens.checked_mul(2) else {
                break;
            };
            page_tokens = next;
        }

        let selected = selected.or_else(|| {
            candidates
                .iter()
                .copied()
                .find(|candidate| candidate.fits_scratch_budget)
        });
        let Some(selected) = selected else {
            return Err(CfrError::InvalidConfig(
                "no page-size candidate fits scratch budget",
            ));
        };

        Ok(PageTuningResult {
            page_tokens: selected.page_tokens,
            kv_bytes: selected.kv_bytes,
            cache_target_bytes,
            candidates,
        })
    }
}

const fn validate_input(input: PageTuningInput) -> Result<()> {
    if input.head_dim == 0 {
        return Err(CfrError::InvalidConfig("head_dim must be non-zero"));
    }
    if input.context_tokens == 0 {
        return Err(CfrError::InvalidConfig("context_tokens must be non-zero"));
    }
    if input.min_page_tokens == 0 || input.max_page_tokens < input.min_page_tokens {
        return Err(CfrError::InvalidConfig("invalid page token bounds"));
    }
    if input.max_scratch_bytes == 0 {
        return Err(CfrError::InvalidConfig(
            "max_scratch_bytes must be non-zero",
        ));
    }
    Ok(())
}

fn cache_target(input: PageTuningInput) -> usize {
    let l2_target = input.l2_bytes / 2;
    let l3_target = input.l3_bytes / 16;
    l2_target
        .clamp(1, l3_target.max(1))
        .min(input.max_scratch_bytes)
}
