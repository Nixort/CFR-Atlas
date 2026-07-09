// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! Double-buffered cold-page regeneration pipeline.

use crate::layout::{checked_matrix_len, checked_range_len, prepare_zeroed, wipe_f32};
use crate::{KvRegenerator, PageKey, Result};
use std::ops::Range;

/// One regenerated page buffer.
#[derive(Debug)]
pub struct ColdPageBuffer {
    key: PageKey,
    token_range: Range<usize>,
    tokens: usize,
    k: Vec<f32>,
    v: Vec<f32>,
}

impl ColdPageBuffer {
    /// Creates an empty page buffer.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            key: PageKey::new(0, 0, 0),
            token_range: 0..0,
            tokens: 0,
            k: Vec::new(),
            v: Vec::new(),
        }
    }

    /// Page identity.
    #[must_use]
    pub const fn key(&self) -> PageKey {
        self.key
    }

    /// Token range represented by this buffer.
    #[must_use]
    pub fn token_range(&self) -> Range<usize> {
        self.token_range.clone()
    }

    /// Number of token rows.
    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.tokens
    }

    /// Key matrix.
    #[must_use]
    pub fn k(&self) -> &[f32] {
        &self.k
    }

    /// Value matrix.
    #[must_use]
    pub fn v(&self) -> &[f32] {
        &self.v
    }

    fn regenerate<R: KvRegenerator>(
        &mut self,
        regenerator: &R,
        key: PageKey,
        token_range: Range<usize>,
        head_dim: usize,
    ) -> Result<()> {
        let tokens = checked_range_len("cold-page pipeline range", &token_range)?;
        let len = checked_matrix_len("cold-page pipeline matrix", tokens, head_dim)?;
        prepare_zeroed(&mut self.k, len);
        prepare_zeroed(&mut self.v, len);
        if let Err(err) = regenerator.regenerate_page(
            key,
            token_range.clone(),
            head_dim,
            &mut self.k[..len],
            &mut self.v[..len],
        ) {
            self.clear();
            return Err(err);
        }
        self.key = key;
        self.token_range = token_range;
        self.tokens = tokens;
        Ok(())
    }

    fn clear(&mut self) {
        wipe_f32(&mut self.k);
        wipe_f32(&mut self.v);
        self.key = PageKey::new(0, 0, 0);
        self.token_range = 0..0;
        self.tokens = 0;
    }
}

impl Default for ColdPageBuffer {
    fn default() -> Self {
        Self::empty()
    }
}

impl Drop for ColdPageBuffer {
    fn drop(&mut self) {
        self.clear();
    }
}

/// Two-slot regeneration pipeline.
#[derive(Debug, Default)]
pub struct DoubleBufferedPipeline {
    buffers: [ColdPageBuffer; 2],
    next_slot: usize,
}

impl DoubleBufferedPipeline {
    /// Creates an empty double-buffered pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Regenerates a page into the inactive slot and returns a stable view.
    pub fn regenerate_next<R: KvRegenerator>(
        &mut self,
        regenerator: &R,
        key: PageKey,
        token_range: Range<usize>,
        head_dim: usize,
    ) -> Result<&ColdPageBuffer> {
        let slot = self.next_slot;
        let mut regenerated = ColdPageBuffer::empty();
        regenerated.regenerate(regenerator, key, token_range, head_dim)?;
        std::mem::swap(&mut self.buffers[slot], &mut regenerated);
        self.next_slot = 1usize.saturating_sub(slot);
        Ok(&self.buffers[slot])
    }

    /// Returns both backing buffers for diagnostics.
    #[must_use]
    pub const fn buffers(&self) -> &[ColdPageBuffer; 2] {
        &self.buffers
    }
}
