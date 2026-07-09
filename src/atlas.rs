// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! Atlas runtime coordinator for virtual `KV` pages, scratch buffers and folded attention.
//!
//! This module owns the execution path that decides whether a page is resident,
//! regenerates cold pages and streams every K/V block into the reducer.

use crate::cache::InsertOutcome;
use crate::layout::{checked_add, expect_len, prepare_zeroed, usize_to_u64_saturating, wipe_f32};
use crate::{
    CfrCounters, CfrError, CfrStatsSnapshot, Config, DotProductKernel, FoldedAttention, HotCache,
    KvRegenerator, NeverAdmit, PageKey, ResidencyContext, ResidencyDecision, ResidencyPolicy,
    Result,
};

struct ColdPageJob<'a> {
    key: PageKey,
    start: usize,
    end: usize,
    head_dim: usize,
    query: &'a [f32],
    context_tokens: usize,
}

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
/// It owns scratch buffers, a bounded hot cache, an online attention reducer and
/// execution counters. It does not own model weights or token history; those
/// belong to your [`KvRegenerator`] backend.
#[derive(Debug)]
pub struct CfrAtlas {
    config: Config,
    cache: HotCache,
    scratch_k: Vec<f32>,
    scratch_v: Vec<f32>,
    folded: FoldedAttention,
    counters: CfrCounters,
}

impl CfrAtlas {
    /// Creates a new atlas from a validated configuration.
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;
        let folded = FoldedAttention::new(config.head_dim, config.scale)?;
        Ok(Self {
            cache: HotCache::new(config.hot_cache_bytes),
            config,
            scratch_k: Vec::new(),
            scratch_v: Vec::new(),
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

    /// Exact `CFR` attention for one `(layer, head, query)` over a causal context.
    ///
    /// The result is semantically equivalent to materializing all K/V rows and
    /// applying normal attention, assuming the regenerator returns exact K/V.
    pub fn attend_exact<R: KvRegenerator>(
        &mut self,
        regenerator: &R,
        request: AttentionRequest<'_>,
        output: &mut [f32],
    ) -> Result<()> {
        self.attend_exact_with_policy(regenerator, &NeverAdmit, request, output)
    }

    /// Exact `CFR` attention with a custom residency policy.
    pub fn attend_exact_with_policy<R: KvRegenerator, P: ResidencyPolicy>(
        &mut self,
        regenerator: &R,
        policy: &P,
        request: AttentionRequest<'_>,
        output: &mut [f32],
    ) -> Result<()> {
        expect_len("query", self.config.head_dim, request.query.len())?;
        expect_len("attention output", self.config.head_dim, output.len())?;
        if request.context_tokens == 0 {
            return Err(CfrError::InvalidConfig("context_tokens must be non-zero"));
        }

        self.folded.reset();

        let page_tokens = self.config.page_tokens;
        let head_dim = self.config.head_dim;
        let mut start = 0usize;

        while start < request.context_tokens {
            let end =
                checked_add("attention page end", start, page_tokens)?.min(request.context_tokens);
            let tokens = end - start;
            let key = PageKey::new(request.layer, request.head, start);
            self.ensure_scratch_limit(key, tokens)?;

            if !self.consume_hot_page_if_shape_matches(key, tokens, request.query)? {
                let job = ColdPageJob {
                    key,
                    start,
                    end,
                    head_dim,
                    query: request.query,
                    context_tokens: request.context_tokens,
                };
                self.regenerate_consume_and_maybe_admit(regenerator, policy, &job)?;
            }

            start = end;
        }

        self.folded.finish_into(output)
    }

    fn consume_hot_page_if_shape_matches(
        &mut self,
        key: PageKey,
        tokens: usize,
        query: &[f32],
    ) -> Result<bool> {
        let Some(resident_tokens) = self.cache.page_tokens(&key) else {
            return Ok(false);
        };

        if resident_tokens != tokens {
            if self.cache.remove(&key) {
                self.counters.cache_evictions = self.counters.cache_evictions.saturating_add(1);
            }
            return Ok(false);
        }

        let view = self.cache.get(&key).ok_or(CfrError::InvalidPage {
            key,
            message: "hot page disappeared before consumption",
        })?;
        self.folded.consume_page(query, view.k, view.v, tokens)?;
        self.counters.hot_hits = self.counters.hot_hits.saturating_add(1);
        self.counters.consumed_tokens = self
            .counters
            .consumed_tokens
            .saturating_add(usize_to_u64_saturating(tokens));
        Ok(true)
    }

    const fn ensure_scratch_limit(&self, key: PageKey, tokens: usize) -> Result<()> {
        if tokens > self.config.max_scratch_tokens {
            return Err(CfrError::InvalidPage {
                key,
                message: "page exceeds max_scratch_tokens",
            });
        }
        Ok(())
    }

    fn regenerate_consume_and_maybe_admit<R: KvRegenerator, P: ResidencyPolicy>(
        &mut self,
        regenerator: &R,
        policy: &P,
        job: &ColdPageJob<'_>,
    ) -> Result<()> {
        let tokens = job.end - job.start;
        self.ensure_scratch_limit(job.key, tokens)?;
        let needed = self.config.page_f32_len(tokens)?;
        prepare_zeroed(&mut self.scratch_k, needed);
        prepare_zeroed(&mut self.scratch_v, needed);

        if let Err(err) = regenerator.regenerate_page(
            job.key,
            job.start..job.end,
            job.head_dim,
            &mut self.scratch_k[..needed],
            &mut self.scratch_v[..needed],
        ) {
            self.wipe_scratch_if_enabled();
            return Err(err);
        }

        self.counters.cold_regenerations = self.counters.cold_regenerations.saturating_add(1);

        if let Err(err) = self.folded.consume_page(
            job.query,
            &self.scratch_k[..needed],
            &self.scratch_v[..needed],
            tokens,
        ) {
            self.wipe_scratch_if_enabled();
            return Err(err);
        }
        self.counters.consumed_tokens = self
            .counters
            .consumed_tokens
            .saturating_add(usize_to_u64_saturating(tokens));

        let stats = self.stats();
        let decision_context = ResidencyContext {
            key: job.key,
            page_tokens: tokens,
            context_tokens: job.context_tokens,
            stats: &stats,
            hot_cache_max_bytes: self.cache.max_bytes(),
            hot_cache_used_bytes: self.cache.used_bytes(),
        };

        if self.config.admit_regenerated_pages
            && policy.decide_with_context(&decision_context) == ResidencyDecision::Admit
        {
            let insert_result = self.cache.insert_internal(
                job.key,
                tokens,
                job.head_dim,
                &self.scratch_k[..needed],
                &self.scratch_v[..needed],
            );
            match insert_result {
                Ok(InsertOutcome::Inserted { evicted }) => {
                    self.counters.cache_admissions =
                        self.counters.cache_admissions.saturating_add(1);
                    self.counters.cache_evictions = self
                        .counters
                        .cache_evictions
                        .saturating_add(usize_to_u64_saturating(evicted));
                }
                Ok(InsertOutcome::RejectedTooLarge) => {
                    self.counters.cache_admission_rejections =
                        self.counters.cache_admission_rejections.saturating_add(1);
                }
                Err(err) => {
                    self.wipe_scratch_if_enabled();
                    return Err(err);
                }
            }
        }

        self.wipe_scratch_if_enabled();
        Ok(())
    }

    fn wipe_scratch_if_enabled(&mut self) {
        if self.config.wipe_scratch_after_use {
            wipe_f32(&mut self.scratch_k);
            wipe_f32(&mut self.scratch_v);
        }
    }
}

impl Drop for CfrAtlas {
    fn drop(&mut self) {
        wipe_f32(&mut self.scratch_k);
        wipe_f32(&mut self.scratch_v);
    }
}
