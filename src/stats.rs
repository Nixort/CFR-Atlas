// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 6 july 2026

//! Runtime counters and immutable statistics snapshots.
//!
//! This module provides lightweight observability for hot hits, cold regeneration,
//! cache pressure and folded-attention token consumption.

/// Mutable execution counters collected by [`crate::CfrAtlas`].
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CfrCounters {
    /// Number of resident hot-cache page hits.
    pub hot_hits: u64,
    /// Number of cold pages regenerated.
    pub cold_regenerations: u64,
    /// Number of page insertions into hot cache that succeeded.
    pub cache_admissions: u64,
    /// Number of page insertions rejected by cache budget.
    pub cache_admission_rejections: u64,
    /// Number of pages evicted from hot cache.
    pub cache_evictions: u64,
    /// Number of tokens consumed by folded attention.
    pub consumed_tokens: u64,
}

impl CfrCounters {
    /// Resets all counters to zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Returns an immutable snapshot.
    #[must_use]
    pub const fn snapshot(
        &self,
        hot_cache_bytes: usize,
        hot_cache_pages: usize,
    ) -> CfrStatsSnapshot {
        CfrStatsSnapshot {
            hot_hits: self.hot_hits,
            cold_regenerations: self.cold_regenerations,
            cache_admissions: self.cache_admissions,
            cache_admission_rejections: self.cache_admission_rejections,
            cache_evictions: self.cache_evictions,
            consumed_tokens: self.consumed_tokens,
            hot_cache_bytes,
            hot_cache_pages,
        }
    }
}

/// Read-only statistics snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CfrStatsSnapshot {
    /// Number of resident hot-cache page hits.
    pub hot_hits: u64,
    /// Number of cold pages regenerated.
    pub cold_regenerations: u64,
    /// Number of page insertions into hot cache that succeeded.
    pub cache_admissions: u64,
    /// Number of page insertions rejected by cache budget.
    pub cache_admission_rejections: u64,
    /// Number of pages evicted from hot cache.
    pub cache_evictions: u64,
    /// Number of tokens consumed by folded attention.
    pub consumed_tokens: u64,
    /// Current hot-cache bytes.
    pub hot_cache_bytes: usize,
    /// Current hot-cache page count.
    pub hot_cache_pages: usize,
}
