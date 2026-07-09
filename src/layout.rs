// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! Checked layout, capacity and buffer hygiene helpers.
//!
//! Core code, backend adapters, examples and tests use this module as the
//! single audited implementation for shape math, numeric narrowing and data
//! wiping.

use crate::{CfrError, Result};
use std::ops::Range;

const F32_EXACT_INT_MAX: usize = 16_777_216;
const F64_EXACT_INT_MAX_U64: u64 = 9_007_199_254_740_991;

/// Validates a slice length.
#[inline]
pub const fn expect_len(name: &'static str, expected: usize, got: usize) -> Result<()> {
    if expected == got {
        Ok(())
    } else {
        Err(CfrError::Dimension {
            name,
            expected,
            got,
        })
    }
}

/// Checked `usize` addition with a stable error.
#[inline]
pub fn checked_add(name: &'static str, left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or(CfrError::CapacityOverflow { name })
}

/// Checked `usize` multiplication with a stable error.
#[inline]
pub fn checked_mul(name: &'static str, left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or(CfrError::CapacityOverflow { name })
}

/// Checked length for a non-empty half-open range.
#[inline]
pub fn checked_range_len(name: &'static str, range: &Range<usize>) -> Result<usize> {
    if range.start >= range.end {
        return Err(CfrError::InvalidConfig(name));
    }
    range
        .end
        .checked_sub(range.start)
        .ok_or(CfrError::CapacityOverflow { name })
}

/// Checked row-major matrix length: `rows * columns`.
#[inline]
pub fn checked_matrix_len(name: &'static str, rows: usize, columns: usize) -> Result<usize> {
    checked_mul(name, rows, columns)
}

/// Checked byte count for K and V `f32` pages together.
#[inline]
pub fn checked_kv_bytes(name: &'static str, tokens: usize, head_dim: usize) -> Result<usize> {
    let values = checked_matrix_len(name, tokens, head_dim)?;
    let one_matrix = checked_mul(name, values, std::mem::size_of::<f32>())?;
    checked_mul(name, one_matrix, 2)
}

/// Checked row range for a row-major matrix.
#[inline]
pub fn checked_row_range(
    name: &'static str,
    row: usize,
    width: usize,
    total_len: usize,
) -> Result<Range<usize>> {
    let start = checked_matrix_len(name, row, width)?;
    let end = checked_add(name, start, width)?;
    if end > total_len {
        return Err(CfrError::Dimension {
            name,
            expected: end,
            got: total_len,
        });
    }
    Ok(start..end)
}

/// Converts a `usize` to `f32` only when the integer is exactly representable.
#[inline]
pub const fn usize_to_f32_checked(name: &'static str, value: usize) -> Result<f32> {
    if value > F32_EXACT_INT_MAX {
        return Err(CfrError::Numeric(name));
    }
    Ok(usize_to_f32_exact(value))
}

/// Converts a `u32` to `f32` only when the integer is exactly representable.
#[inline]
pub fn u32_to_f32_checked(name: &'static str, value: u32) -> Result<f32> {
    let value =
        usize::try_from(value).map_err(|_| CfrError::Numeric("u32 value does not fit usize"))?;
    usize_to_f32_checked(name, value)
}

/// Converts a `usize` to `f64` only when the integer is exactly representable.
#[inline]
pub fn usize_to_f64_checked(name: &'static str, value: usize) -> Result<f64> {
    let value_u64 =
        u64::try_from(value).map_err(|_| CfrError::Numeric("usize value does not fit u64"))?;
    if value_u64 > F64_EXACT_INT_MAX_U64 {
        return Err(CfrError::Numeric(name));
    }
    Ok(usize_to_f64_exact(value))
}

/// Converts a `u64` to `f64` only when the integer is exactly representable.
#[inline]
pub const fn u64_to_f64_checked(name: &'static str, value: u64) -> Result<f64> {
    if value > F64_EXACT_INT_MAX_U64 {
        return Err(CfrError::Numeric(name));
    }
    Ok(u64_to_f64_exact(value))
}

/// Converts `usize` to `u64`, saturating on unusual targets where it cannot fit.
#[inline]
#[must_use]
pub fn usize_to_u64_saturating(value: usize) -> u64 {
    u64::try_from(value).map_or(u64::MAX, |value| value)
}

/// Converts a `u64` to `f32` only after the shared checked narrowing path.
#[inline]
pub fn u64_to_f32_checked(name: &'static str, value: u64) -> Result<f32> {
    let value = u64_to_f64_checked(name, value)?;
    f64_to_f32_checked(name, value)
}

/// Converts a finite `f64` to `f32` after range validation.
#[inline]
pub fn f64_to_f32_checked(name: &'static str, value: f64) -> Result<f32> {
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        return Err(CfrError::Numeric(name));
    }
    Ok(f64_to_f32_narrow(value))
}

#[inline]
#[allow(clippy::cast_precision_loss)]
const fn usize_to_f32_exact(value: usize) -> f32 {
    value as f32
}

#[inline]
#[allow(clippy::cast_precision_loss)]
const fn usize_to_f64_exact(value: usize) -> f64 {
    value as f64
}

#[inline]
#[allow(clippy::cast_precision_loss)]
const fn u64_to_f64_exact(value: u64) -> f64 {
    value as f64
}

#[inline]
#[allow(clippy::cast_possible_truncation)]
const fn f64_to_f32_narrow(value: f64) -> f32 {
    value as f32
}

/// Ensures a scratch buffer can expose `len` elements and zeroes all live data.
#[inline]
pub fn prepare_zeroed(buffer: &mut Vec<f32>, len: usize) {
    if buffer.len() < len {
        buffer.resize(len, 0.0);
    }
    buffer.fill(0.0);
}

/// Ensures an f64 scratch buffer can expose `len` elements and zeroes all live data.
#[inline]
pub fn prepare_zeroed_f64(buffer: &mut Vec<f64>, len: usize) {
    if buffer.len() < len {
        buffer.resize(len, 0.0);
    }
    buffer.fill(0.0);
}

/// Clears live f32 data before a buffer is reused or dropped.
#[inline]
pub fn wipe_f32(values: &mut [f32]) {
    values.fill(0.0);
}

/// Clears live f64 data before a buffer is reused or dropped.
#[inline]
pub fn wipe_f64(values: &mut [f64]) {
    values.fill(0.0);
}

/// Validates that every value in a slice is finite.
#[inline]
pub fn expect_all_finite(name: &'static str, values: &[f32]) -> Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(CfrError::Numeric(name))
    }
}

/// Maximum absolute difference with explicit non-finite rejection.
pub fn max_abs_diff_finite(name: &'static str, left: &[f32], right: &[f32]) -> Result<f32> {
    expect_len(name, left.len(), right.len())?;
    let mut max_diff = 0.0f32;
    for (a, b) in left.iter().zip(right.iter()) {
        if !a.is_finite() || !b.is_finite() {
            return Err(CfrError::Numeric("non-finite value during finite diff"));
        }
        max_diff = max_diff.max((a - b).abs());
    }
    Ok(max_diff)
}
