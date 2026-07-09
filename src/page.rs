// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 6 july 2026

//! Stable virtual page identity and token-range validation.
//!
//! This module is the chart that maps causal context tokens into deterministic
//! layer/head pages that can be stored hot or regenerated cold.

use crate::{CfrError, Result};
use std::ops::Range;

/// Stable identity of a virtual `KV` page in a causal transformer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PageKey {
    /// Transformer layer index.
    pub layer: u32,
    /// Attention head index.
    pub head: u32,
    /// First token represented by the page.
    pub start_token: usize,
}

impl PageKey {
    /// Creates a page key.
    #[inline]
    #[must_use]
    pub const fn new(layer: u32, head: u32, start_token: usize) -> Self {
        Self {
            layer,
            head,
            start_token,
        }
    }
}

/// Concrete token range associated with a [`PageKey`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageRange {
    /// Page key.
    pub key: PageKey,
    /// Half-open token range `[start, end)`.
    pub tokens: Range<usize>,
}

impl PageRange {
    /// Creates and validates a page range.
    pub const fn new(key: PageKey, tokens: Range<usize>) -> Result<Self> {
        if tokens.start != key.start_token {
            return Err(CfrError::InvalidPage {
                key,
                message: "range start must equal key.start_token",
            });
        }
        if tokens.end <= tokens.start {
            return Err(CfrError::InvalidPage {
                key,
                message: "page token range must be non-empty",
            });
        }
        Ok(Self { key, tokens })
    }

    /// Number of tokens in the page range.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.tokens.end.saturating_sub(self.tokens.start)
    }

    /// Whether the page range is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}
