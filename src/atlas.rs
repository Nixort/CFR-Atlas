// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas runtime state and public configuration surface.

//! Public runtime state for virtual `KV` attention.
//!
//! Page execution lives in the private `execution` module. This module keeps
//! the stable public object, its configuration surface, cache accessors and
//! runtime counters in one place.

use crate::cache::InsertOutcome;
use crate::layout::usize_to_u64_saturating;
use crate::scratch::ScratchBuffers;
use crate::{
    CfrCounters, CfrStatsSnapshot, Config, DotProductKernel, FoldedAttention, HotCache, PageKey,
    Result,
};

/// One attention call over a causal context.
#[derive(Debug, Clone, Copy)]
pub struct AttentionRequest<'a> {
    /// Transformer layer id.
    pub layer: u32,
    /// K/V head id.
    pub head: u32,
    /// Query vector for the current token.
    pub query: &'a [f32],
    /// Number of causal context tokens visible to the query.
    pub context_tokens: usize,
}

impl<'a> AttentionRequest<'a> {
    /// Creates a request for exact folded attention.
    #[must_use]
    pub const fn new(layer: u32, head: u32, query: &'a [f32], context_tokens: usize) -> Self {
        Self {
            layer,
            head,
            query,
            context_tokens,
        }
    }
}

/// Main `CFR-Atlas` object.
///
/// It owns a bounded hot cache, reusable cold-page scratch storage, an online
/// attention reducer and execution counters. It does not own model weights or
/// token history; those remain behind the [`crate::KvRegenerator`] boundary.
#[derive(Debug)]
pub struct CfrAtlas {
    pub(crate) config: Config,
    pub(crate) cache: HotCache,
    pub(crate) scratch: ScratchBuffers,
    pub(crate) folded: FoldedAttention,
    pub(crate) counters: CfrCounters,
}

impl CfrAtlas {
    /// Creates a new atlas from a validated configuration.
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;
        let folded = FoldedAttention::new(config.head_dim, config.scale)?;
        Ok(Self {
            cache: HotCache::new(config.hot_cache_bytes),
            config,
            scratch: ScratchBuffers::default(),
            folded,
            counters: CfrCounters::default(),
        })
    }

    /// Returns the immutable configuration.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &Config {
        &self.config
    }

    /// Returns the immutable hot cache.
    #[inline]
    #[must_use]
    pub const fn hot_cache(&self) -> &HotCache {
        &self.cache
    }

    /// Returns the mutable hot cache.
    #[inline]
    pub fn hot_cache_mut(&mut self) -> &mut HotCache {
        &mut self.cache
    }

    /// Returns execution counters.
    #[inline]
    #[must_use]
    pub const fn counters(&self) -> &CfrCounters {
        &self.counters
    }

    /// Sets a per-layer hot-cache budget and evicts pages if needed.
    #[must_use]
    pub fn set_layer_hot_cache_bytes(&mut self, layer: u32, bytes: usize) -> usize {
        let evicted = self.cache.set_layer_budget(layer, bytes);
        self.counters.cache_evictions = self
            .counters
            .cache_evictions
            .saturating_add(usize_to_u64_saturating(evicted));
        evicted
    }

    /// Removes a per-layer hot-cache budget.
    pub fn clear_layer_hot_cache_bytes(&mut self, layer: u32) {
        self.cache.clear_layer_budget(layer);
    }

    /// Returns the configured hot-cache budget for one layer.
    #[must_use]
    pub fn layer_hot_cache_bytes(&self, layer: u32) -> Option<usize> {
        self.cache.layer_budget(layer)
    }

    /// Returns the resident hot-cache bytes used by one layer.
    #[must_use]
    pub fn layer_used_hot_cache_bytes(&self, layer: u32) -> usize {
        self.cache.layer_used_bytes(layer)
    }

    /// Resets execution counters.
    pub fn reset_counters(&mut self) {
        self.counters.reset();
    }

    /// Selects the dot-product kernel used by folded attention.
    pub fn set_dot_kernel(&mut self, kernel: DotProductKernel) {
        self.folded.set_kernel(kernel);
    }

    /// Returns the selected dot-product kernel.
    #[must_use]
    pub const fn dot_kernel(&self) -> DotProductKernel {
        self.folded.kernel()
    }

    /// Returns a point-in-time statistics snapshot.
    #[must_use]
    pub fn stats(&self) -> CfrStatsSnapshot {
        self.counters
            .snapshot(self.cache.used_bytes(), self.cache.len())
    }

    /// Inserts a known hot K/V page.
    ///
    /// This is useful for newest tokens, prompt prefill, or runtime-specific
    /// speculative residency.
    pub fn insert_hot_page(
        &mut self,
        key: PageKey,
        tokens: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<bool> {
        match self
            .cache
            .insert_internal(key, tokens, self.config.head_dim, k, v)?
        {
            InsertOutcome::Inserted { evicted } => {
                self.counters.cache_admissions = self.counters.cache_admissions.saturating_add(1);
                self.counters.cache_evictions = self
                    .counters
                    .cache_evictions
                    .saturating_add(usize_to_u64_saturating(evicted));
                Ok(true)
            }
            InsertOutcome::RejectedTooLarge => {
                self.counters.cache_admission_rejections =
                    self.counters.cache_admission_rejections.saturating_add(1);
                Ok(false)
            }
        }
    }
}
