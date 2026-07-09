// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! Backend conformance helpers.
//!
//! These helpers compare regenerated pages against reference stored K/V pages.

use crate::layout::{
    checked_matrix_len, checked_range_len, expect_len, max_abs_diff_finite, wipe_f32,
};
use crate::{CfrError, KvRegenerator, PageKey, Result};
use std::ops::Range;

/// Result of comparing one regenerated page against a stored reference page.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PageConformance {
    /// Page identity that was checked.
    pub key: PageKey,
    /// Number of token rows checked.
    pub tokens: usize,
    /// Maximum absolute K difference.
    pub max_abs_k: f32,
    /// Maximum absolute V difference.
    pub max_abs_v: f32,
    /// Tolerance used for the check.
    pub tolerance: f32,
}

impl PageConformance {
    /// Returns true when both K and V are within tolerance.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.max_abs_k <= self.tolerance && self.max_abs_v <= self.tolerance
    }
}

/// Compares a regenerated page against stored reference K/V matrices.
pub fn compare_regenerated_page<R: KvRegenerator>(
    regenerator: &R,
    key: PageKey,
    token_range: Range<usize>,
    head_dim: usize,
    reference_k: &[f32],
    reference_v: &[f32],
    tolerance: f32,
) -> Result<PageConformance> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(CfrError::InvalidConfig(
            "conformance tolerance must be finite and non-negative",
        ));
    }
    if token_range.start != key.start_token {
        return Err(CfrError::InvalidPage {
            key,
            message: "conformance range start must equal key.start_token",
        });
    }
    if token_range.start >= token_range.end {
        return Err(CfrError::InvalidPage {
            key,
            message: "conformance range must be non-empty",
        });
    }
    if head_dim == 0 {
        return Err(CfrError::InvalidConfig("head_dim must be non-zero"));
    }

    let tokens = checked_range_len("conformance range length", &token_range)?;
    let expected = checked_matrix_len("conformance page length", tokens, head_dim)?;
    expect_len("reference K", expected, reference_k.len())?;
    expect_len("reference V", expected, reference_v.len())?;

    let mut regenerated_k = vec![0.0; expected];
    let mut regenerated_v = vec![0.0; expected];
    if let Err(err) = regenerator.regenerate_page(
        key,
        token_range,
        head_dim,
        &mut regenerated_k,
        &mut regenerated_v,
    ) {
        wipe_f32(&mut regenerated_k);
        wipe_f32(&mut regenerated_v);
        return Err(err);
    }

    let max_abs_k = match max_abs_diff_finite("conformance K", reference_k, &regenerated_k) {
        Ok(value) => value,
        Err(err) => {
            wipe_f32(&mut regenerated_k);
            wipe_f32(&mut regenerated_v);
            return Err(err);
        }
    };
    let max_abs_v = match max_abs_diff_finite("conformance V", reference_v, &regenerated_v) {
        Ok(value) => value,
        Err(err) => {
            wipe_f32(&mut regenerated_k);
            wipe_f32(&mut regenerated_v);
            return Err(err);
        }
    };
    wipe_f32(&mut regenerated_k);
    wipe_f32(&mut regenerated_v);

    Ok(PageConformance {
        key,
        tokens,
        max_abs_k,
        max_abs_v,
        tolerance,
    })
}

/// Asserts that a regenerated page matches stored K/V within tolerance.
pub fn assert_regenerated_page<R: KvRegenerator>(
    regenerator: &R,
    key: PageKey,
    token_range: Range<usize>,
    head_dim: usize,
    reference_k: &[f32],
    reference_v: &[f32],
    tolerance: f32,
) -> Result<PageConformance> {
    let report = compare_regenerated_page(
        regenerator,
        key,
        token_range,
        head_dim,
        reference_k,
        reference_v,
        tolerance,
    )?;
    if report.passed() {
        Ok(report)
    } else {
        Err(CfrError::Regenerator(format!(
            "page conformance failed: max_abs_k={}, max_abs_v={}, tolerance={}",
            report.max_abs_k, report.max_abs_v, report.tolerance
        )))
    }
}
