// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 8 july 2026

//! Backend contract for exact K/V page regeneration.
//!
//! A real inference runtime implements this trait by replaying the precise model
//! computation needed to recreate the requested K/V page.

use crate::{PageKey, Result};
use std::ops::Range;

/// Backend contract for exact K/V page regeneration.
///
/// A production `LLM` runtime implements this trait by replaying exactly the
/// computation needed to produce a requested layer/head `KV` block. The output
/// layout is row-major: `[token][head_dim]` for both K and V.
pub trait KvRegenerator {
    /// Regenerates one page of keys and values.
    fn regenerate_page(
        &self,
        key: PageKey,
        token_range: Range<usize>,
        head_dim: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<()>;
}
