// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! Bounded hot-page cache for resident K/V blocks.
//!
//! This module keeps the small RAM budget explicit and evicts pages with a
//! dependency-free `LRU` policy. Phase 3 adds optional per-layer byte budgets
//! without changing the exactness semantics of regenerated pages.

use crate::layout::{
    checked_kv_bytes, checked_matrix_len, expect_all_finite, expect_len, usize_to_u64_saturating,
    wipe_f32,
};
use crate::{CfrError, PageKey, Result};
use std::collections::HashMap;

#[derive(Debug)]
struct HotPage {
    tokens: usize,
    k: Box<[f32]>,
    v: Box<[f32]>,
    bytes: usize,
    last_touch: u64,
}

impl Drop for HotPage {
    fn drop(&mut self) {
        wipe_f32(self.k.as_mut());
        wipe_f32(self.v.as_mut());
    }
}

/// Borrowed view over a resident K/V page.
pub struct PageView<'a> {
    /// Number of token rows in the page.
    pub tokens: usize,
    /// Row-major key matrix `[token][head_dim]`.
    pub k: &'a [f32],
    /// Row-major value matrix `[token][head_dim]`.
    pub v: &'a [f32],
}

/// Result of trying to insert a hot page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertOutcome {
    /// Page was inserted; `evicted` is the number of pages removed to fit it.
    Inserted {
        /// Number of pages evicted before insertion.
        evicted: usize,
    },
    /// Page is larger than the applicable cache budget.
    RejectedTooLarge,
}

/// Bounded resident K/V cache.
#[derive(Debug)]
pub struct HotCache {
    max_bytes: usize,
    used_bytes: usize,
    clock: u64,
    entries: HashMap<PageKey, HotPage>,
    layer_budgets: HashMap<u32, usize>,
    layer_used_bytes: HashMap<u32, usize>,
}

impl HotCache {
    /// Creates an empty hot cache with a byte budget.
    #[must_use]
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            used_bytes: 0,
            clock: 0,
            entries: HashMap::new(),
            layer_budgets: HashMap::new(),
            layer_used_bytes: HashMap::new(),
        }
    }

    /// Maximum resident bytes.
    #[inline]
    #[must_use]
    pub const fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Current resident bytes.
    #[inline]
    #[must_use]
    pub const fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    /// Current resident page count.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Sets or replaces a per-layer hot-cache budget.
    ///
    /// Existing pages from that layer are evicted with LRU order until the layer
    /// usage fits the new budget. The global cache budget still applies.
    #[must_use]
    pub fn set_layer_budget(&mut self, layer: u32, bytes: usize) -> usize {
        self.layer_budgets.insert(layer, bytes);
        self.evict_layer_until(layer, 0)
    }

    /// Removes a per-layer budget and keeps existing resident pages.
    pub fn clear_layer_budget(&mut self, layer: u32) {
        self.layer_budgets.remove(&layer);
    }

    /// Returns the configured budget for a layer.
    #[must_use]
    pub fn layer_budget(&self, layer: u32) -> Option<usize> {
        self.layer_budgets.get(&layer).copied()
    }

    /// Returns resident bytes used by one layer.
    #[must_use]
    pub fn layer_used_bytes(&self, layer: u32) -> usize {
        self.layer_used_bytes
            .get(&layer)
            .copied()
            .map_or(0, |bytes| bytes)
    }

    /// Clears all resident pages but preserves configured layer budgets.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.used_bytes = 0;
        self.layer_used_bytes.clear();
        self.clock = 0;
    }

    /// Returns the resident token count for a page without touching LRU state.
    #[must_use]
    pub fn page_tokens(&self, key: &PageKey) -> Option<usize> {
        self.entries.get(key).map(|page| page.tokens)
    }

    /// Removes a resident page and wipes it through [`HotPage`]'s drop path.
    #[must_use]
    pub fn remove(&mut self, key: &PageKey) -> bool {
        self.remove_page(key).is_some()
    }

    /// Gets a resident page and updates its `LRU` timestamp.
    pub fn get(&mut self, key: &PageKey) -> Option<PageView<'_>> {
        if !self.entries.contains_key(key) {
            return None;
        }
        let now = self.tick();
        let page = self.entries.get_mut(key)?;
        page.last_touch = now;
        Some(PageView {
            tokens: page.tokens,
            k: &page.k,
            v: &page.v,
        })
    }

    /// Inserts a page if it fits into the cache budget.
    pub fn insert(
        &mut self,
        key: PageKey,
        tokens: usize,
        head_dim: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<bool> {
        Ok(matches!(
            self.insert_internal(key, tokens, head_dim, k, v)?,
            InsertOutcome::Inserted { .. }
        ))
    }

    pub(crate) fn insert_internal(
        &mut self,
        key: PageKey,
        tokens: usize,
        head_dim: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<InsertOutcome> {
        if tokens == 0 {
            return Err(CfrError::InvalidPage {
                key,
                message: "hot page token count must be non-zero",
            });
        }
        if head_dim == 0 {
            return Err(CfrError::InvalidConfig("head_dim must be non-zero"));
        }

        let expected = checked_matrix_len("hot page matrix length", tokens, head_dim)?;
        expect_len("hot page K", expected, k.len())?;
        expect_len("hot page V", expected, v.len())?;
        expect_all_finite("hot page K contains a non-finite value", k)?;
        expect_all_finite("hot page V contains a non-finite value", v)?;

        let bytes = checked_kv_bytes("hot page bytes", tokens, head_dim)?;
        if bytes > self.max_bytes {
            return Ok(InsertOutcome::RejectedTooLarge);
        }
        if self
            .layer_budget(key.layer)
            .is_some_and(|budget| bytes > budget)
        {
            return Ok(InsertOutcome::RejectedTooLarge);
        }

        let mut page = HotPage {
            tokens,
            k: k.to_vec().into_boxed_slice(),
            v: v.to_vec().into_boxed_slice(),
            bytes,
            last_touch: 0,
        };
        self.preflight_insert_accounting(key, bytes)?;
        self.remove_page(&key);

        let layer_evicted = self.evict_layer_until(key.layer, bytes);
        let global_evicted = self.evict_until(bytes);
        let evicted = layer_evicted.saturating_add(global_evicted);
        let next_used_bytes =
            self.used_bytes
                .checked_add(bytes)
                .ok_or(CfrError::CapacityOverflow {
                    name: "hot cache used bytes",
                })?;
        let next_layer_bytes = self.layer_used_bytes(key.layer).checked_add(bytes).ok_or(
            CfrError::CapacityOverflow {
                name: "layer hot cache used bytes",
            },
        )?;
        page.last_touch = self.tick();

        self.used_bytes = next_used_bytes;
        self.layer_used_bytes.insert(key.layer, next_layer_bytes);
        self.entries.insert(key, page);
        Ok(InsertOutcome::Inserted { evicted })
    }

    fn tick(&mut self) -> u64 {
        if self.clock == u64::MAX {
            self.renormalize_clock();
        }
        self.clock = self.clock.saturating_add(1);
        self.clock
    }

    fn renormalize_clock(&mut self) {
        let mut keys: Vec<_> = self
            .entries
            .iter()
            .map(|(key, page)| (*key, page.last_touch))
            .collect();
        keys.sort_by_key(|(_, last_touch)| *last_touch);
        for (index, (key, _)) in keys.into_iter().enumerate() {
            if let Some(page) = self.entries.get_mut(&key) {
                page.last_touch = usize_to_u64_saturating(index + 1).min(u64::MAX - 1);
            }
        }
        self.clock = usize_to_u64_saturating(self.entries.len()).min(u64::MAX - 1);
    }

    fn evict_layer_until(&mut self, layer: u32, incoming_bytes: usize) -> usize {
        let Some(budget) = self.layer_budget(layer) else {
            return 0;
        };
        let mut evicted = 0usize;
        while self.layer_used_bytes(layer) > budget.saturating_sub(incoming_bytes) {
            let victim = self
                .entries
                .iter()
                .filter(|(key, _)| key.layer == layer)
                .min_by_key(|(_, page)| page.last_touch)
                .map(|(key, _)| *key);
            let Some(victim_key) = victim else {
                break;
            };
            if self.remove_page(&victim_key).is_some() {
                evicted = evicted.saturating_add(1);
            }
        }
        evicted
    }

    fn evict_until(&mut self, incoming_bytes: usize) -> usize {
        let mut evicted = 0usize;
        while self.used_bytes > self.max_bytes.saturating_sub(incoming_bytes) {
            let victim = self
                .entries
                .iter()
                .min_by_key(|(_, page)| page.last_touch)
                .map(|(key, _)| *key);
            let Some(victim_key) = victim else {
                break;
            };
            if self.remove_page(&victim_key).is_some() {
                evicted = evicted.saturating_add(1);
            }
        }
        evicted
    }

    fn remove_page(&mut self, key: &PageKey) -> Option<HotPage> {
        let page = self.entries.remove(key)?;
        self.used_bytes = self.used_bytes.saturating_sub(page.bytes);
        self.sub_layer_usage(key.layer, page.bytes);
        Some(page)
    }

    fn preflight_insert_accounting(&self, key: PageKey, bytes: usize) -> Result<()> {
        let replaced_bytes = self.entries.get(&key).map_or(0, |page| page.bytes);
        let used_after_replace = self.used_bytes.saturating_sub(replaced_bytes);
        let layer_after_replace = self
            .layer_used_bytes(key.layer)
            .saturating_sub(replaced_bytes);
        let _next_used =
            used_after_replace
                .checked_add(bytes)
                .ok_or(CfrError::CapacityOverflow {
                    name: "hot cache used bytes",
                })?;
        let _next_layer =
            layer_after_replace
                .checked_add(bytes)
                .ok_or(CfrError::CapacityOverflow {
                    name: "layer hot cache used bytes",
                })?;
        Ok(())
    }

    fn sub_layer_usage(&mut self, layer: u32, bytes: usize) {
        let used = self.layer_used_bytes(layer).saturating_sub(bytes);
        if used == 0 {
            self.layer_used_bytes.remove(&layer);
        } else {
            self.layer_used_bytes.insert(layer, used);
        }
    }
}
