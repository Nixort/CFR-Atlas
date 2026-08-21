// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas page execution path.

//! Private exact-attention page execution for [`crate::CfrAtlas`].

use crate::cache::InsertOutcome;
use crate::layout::{checked_add, expect_len, usize_to_u64_saturating};
use crate::{
    AttentionRequest, CfrAtlas, CfrError, KvRegenerator, NeverAdmit, PageKey, ResidencyContext,
    ResidencyDecision, ResidencyPolicy, Result,
};

struct ColdPageJob<'a> {
    key: PageKey,
    start: usize,
    end: usize,
    head_dim: usize,
    query: &'a [f32],
    context_tokens: usize,
}

impl CfrAtlas {
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
        self.execute_pages(regenerator, policy, request)?;
        self.folded.finish_into(output)
    }

    fn execute_pages<R: KvRegenerator, P: ResidencyPolicy>(
        &mut self,
        regenerator: &R,
        policy: &P,
        request: AttentionRequest<'_>,
    ) -> Result<()> {
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
        Ok(())
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

        let regenerated = {
            let (k_out, v_out) = self.scratch.outputs(needed);
            regenerator.regenerate_page(job.key, job.start..job.end, job.head_dim, k_out, v_out)
        };
        if let Err(error) = regenerated {
            self.wipe_scratch_if_enabled();
            return Err(error);
        }
        self.counters.cold_regenerations = self.counters.cold_regenerations.saturating_add(1);

        let consumed = {
            let (k, v) = self.scratch.page(needed);
            self.folded.consume_page(job.query, k, v, tokens)
        };
        if let Err(error) = consumed {
            self.wipe_scratch_if_enabled();
            return Err(error);
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
            let inserted = {
                let (k, v) = self.scratch.page(needed);
                self.cache
                    .insert_internal(job.key, tokens, job.head_dim, k, v)
            };
            match inserted {
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
                Err(error) => {
                    self.wipe_scratch_if_enabled();
                    return Err(error);
                }
            }
        }

        self.wipe_scratch_if_enabled();
        Ok(())
    }

    fn wipe_scratch_if_enabled(&mut self) {
        if self.config.wipe_scratch_after_use {
            self.scratch.wipe();
        }
    }
}
