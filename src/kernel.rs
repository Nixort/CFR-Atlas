// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! Safe CPU dot-product kernel boundary for `Phase 3`.

use crate::layout::expect_len;
use crate::Result;

/// Dot-product kernel selected by an inference runtime.
///
/// The `AutoVectorized` path is written as safe Rust that LLVM can vectorize.
/// Explicit architecture intrinsics can be added later behind this same API
/// without exposing unsafe code to the rest of `CFR-Atlas`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DotProductKernel {
    /// Straight scalar dot product with `f64` accumulation.
    Scalar,
    /// Unrolled safe-Rust kernel intended for compiler auto-vectorization.
    AutoVectorized,
}

impl DotProductKernel {
    /// Returns the default kernel for CPU inference.
    #[must_use]
    pub const fn default_cpu() -> Self {
        Self::AutoVectorized
    }

    /// Computes a dot product with `f64` accumulation.
    pub fn dot(self, lhs: &[f32], rhs: &[f32]) -> Result<f64> {
        match self {
            Self::Scalar => dot_scalar(lhs, rhs),
            Self::AutoVectorized => dot_auto_vectorized(lhs, rhs),
        }
    }
}

impl Default for DotProductKernel {
    fn default() -> Self {
        Self::default_cpu()
    }
}

/// Computes a checked scalar dot product with `f64` accumulation.
#[inline]
pub fn dot_scalar(lhs: &[f32], rhs: &[f32]) -> Result<f64> {
    expect_len("dot product rhs", lhs.len(), rhs.len())?;
    Ok(dot_scalar_inner(lhs, rhs))
}

/// Computes a checked unrolled dot product that keeps a safe API boundary.
#[inline]
pub fn dot_auto_vectorized(lhs: &[f32], rhs: &[f32]) -> Result<f64> {
    expect_len("dot product rhs", lhs.len(), rhs.len())?;
    Ok(dot_auto_vectorized_inner(lhs, rhs))
}

fn dot_scalar_inner(lhs: &[f32], rhs: &[f32]) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(left, right)| f64::from(*left) * f64::from(*right))
        .sum()
}

fn dot_auto_vectorized_inner(lhs: &[f32], rhs: &[f32]) -> f64 {
    let mut chunks_l = lhs.chunks_exact(8);
    let mut chunks_r = rhs.chunks_exact(8);
    let mut acc0 = 0.0f64;
    let mut acc1 = 0.0f64;
    let mut acc2 = 0.0f64;
    let mut acc3 = 0.0f64;
    let mut acc4 = 0.0f64;
    let mut acc5 = 0.0f64;
    let mut acc6 = 0.0f64;
    let mut acc7 = 0.0f64;

    for (left, right) in chunks_l.by_ref().zip(chunks_r.by_ref()) {
        acc0 = f64::from(left[0]).mul_add(f64::from(right[0]), acc0);
        acc1 = f64::from(left[1]).mul_add(f64::from(right[1]), acc1);
        acc2 = f64::from(left[2]).mul_add(f64::from(right[2]), acc2);
        acc3 = f64::from(left[3]).mul_add(f64::from(right[3]), acc3);
        acc4 = f64::from(left[4]).mul_add(f64::from(right[4]), acc4);
        acc5 = f64::from(left[5]).mul_add(f64::from(right[5]), acc5);
        acc6 = f64::from(left[6]).mul_add(f64::from(right[6]), acc6);
        acc7 = f64::from(left[7]).mul_add(f64::from(right[7]), acc7);
    }

    let mut acc = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
    for (left, right) in chunks_l.remainder().iter().zip(chunks_r.remainder().iter()) {
        acc = f64::from(*left).mul_add(f64::from(*right), acc);
    }
    acc
}
