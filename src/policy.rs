// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! Residency policy hooks for regenerated pages.
//!
//! Policies may change speed and RAM pressure, but they must never change model
//! quality because dropped pages remain exactly regenerable.

use crate::layout::usize_to_u64_saturating;
use crate::{CfrStatsSnapshot, PageKey};

/// Residency decision for a regenerated page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidencyDecision {
    /// Do not store the regenerated page.
    Drop,
    /// Store the regenerated page if hot-cache budget allows it.
    Admit,
}

/// Telemetry available to residency policies.
#[derive(Debug, Clone, Copy)]
pub struct ResidencyContext<'a> {
    /// Page being considered for admission.
    pub key: PageKey,
    /// Token rows in the regenerated page.
    pub page_tokens: usize,
    /// Full causal context length.
    pub context_tokens: usize,
    /// Immutable stats snapshot captured before the decision.
    pub stats: &'a CfrStatsSnapshot,
    /// Global hot-cache budget in bytes.
    pub hot_cache_max_bytes: usize,
    /// Global hot-cache bytes currently resident.
    pub hot_cache_used_bytes: usize,
}

/// Policy hook that decides whether a regenerated cold page should become hot.
///
/// The policy must never affect model quality. It only changes speed and RAM.
pub trait ResidencyPolicy {
    /// Decides whether a page should be admitted after exact regeneration.
    fn decide(&self, key: PageKey, page_tokens: usize, context_tokens: usize) -> ResidencyDecision;

    /// Decides with cache telemetry. Policies that do not need telemetry can
    /// keep implementing only [`ResidencyPolicy::decide`].
    fn decide_with_context(&self, context: &ResidencyContext<'_>) -> ResidencyDecision {
        self.decide(context.key, context.page_tokens, context.context_tokens)
    }
}

/// Never admits regenerated pages. This gives the lowest resident memory.
#[derive(Debug, Default, Clone, Copy)]
pub struct NeverAdmit;

impl ResidencyPolicy for NeverAdmit {
    #[inline]
    fn decide(
        &self,
        _key: PageKey,
        _page_tokens: usize,
        _context_tokens: usize,
    ) -> ResidencyDecision {
        ResidencyDecision::Drop
    }
}

/// Keeps pages near the causal frontier hot.
#[derive(Debug, Clone, Copy)]
pub struct KeepRecent {
    /// Number of most recent tokens considered worth keeping resident.
    pub recent_tokens: usize,
}

impl ResidencyPolicy for KeepRecent {
    #[inline]
    fn decide(&self, key: PageKey, page_tokens: usize, context_tokens: usize) -> ResidencyDecision {
        let Some(end) = key.start_token.checked_add(page_tokens) else {
            return ResidencyDecision::Drop;
        };
        let frontier = context_tokens.saturating_sub(self.recent_tokens);
        if end >= frontier {
            ResidencyDecision::Admit
        } else {
            ResidencyDecision::Drop
        }
    }
}

/// Telemetry-driven policy for balanced memory and speed.
#[derive(Debug, Clone, Copy)]
pub struct TelemetryResidencyPolicy {
    /// Always keep this many newest tokens eligible for residency.
    pub recent_tokens: usize,
    /// Maximum allowed hot-cache utilization in permille before new admissions stop.
    pub max_utilization_per_mille: u16,
    /// Admit more aggressively while the hit rate is below this permille value.
    pub target_hit_rate_per_mille: u16,
}

impl TelemetryResidencyPolicy {
    /// Creates a balanced telemetry policy.
    #[must_use]
    pub const fn balanced(recent_tokens: usize) -> Self {
        Self {
            recent_tokens,
            max_utilization_per_mille: 900,
            target_hit_rate_per_mille: 350,
        }
    }
}

impl ResidencyPolicy for TelemetryResidencyPolicy {
    fn decide(&self, key: PageKey, page_tokens: usize, context_tokens: usize) -> ResidencyDecision {
        KeepRecent {
            recent_tokens: self.recent_tokens,
        }
        .decide(key, page_tokens, context_tokens)
    }

    fn decide_with_context(&self, context: &ResidencyContext<'_>) -> ResidencyDecision {
        if self.decide(context.key, context.page_tokens, context.context_tokens)
            == ResidencyDecision::Drop
        {
            return ResidencyDecision::Drop;
        }
        if context.hot_cache_max_bytes == 0 {
            return ResidencyDecision::Drop;
        }
        if utilization_per_mille(context.hot_cache_used_bytes, context.hot_cache_max_bytes)
            >= u64::from(self.max_utilization_per_mille)
        {
            return ResidencyDecision::Drop;
        }
        if hit_rate_per_mille(context.stats) < u64::from(self.target_hit_rate_per_mille)
            || context.stats.hot_cache_pages == 0
        {
            ResidencyDecision::Admit
        } else {
            ResidencyDecision::Drop
        }
    }
}

fn utilization_per_mille(used: usize, max: usize) -> u64 {
    if max == 0 {
        return 1_000;
    }
    let used = usize_to_u64_saturating(used);
    let max = usize_to_u64_saturating(max);
    used.saturating_mul(1_000) / max.max(1)
}

const fn hit_rate_per_mille(stats: &CfrStatsSnapshot) -> u64 {
    let total = stats.hot_hits.saturating_add(stats.cold_regenerations);
    if total == 0 {
        0
    } else {
        stats.hot_hits.saturating_mul(1_000) / total
    }
}
