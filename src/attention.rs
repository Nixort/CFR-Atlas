// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 8 july 2026

//! Folded online-softmax attention reducer.

use crate::layout::{
    checked_add, checked_matrix_len, expect_all_finite, expect_len, f64_to_f32_checked,
    prepare_zeroed_f64, wipe_f32, wipe_f64,
};
use crate::{CfrError, DotProductKernel, Result};

/// Online exact softmax attention reducer for one query vector.
#[derive(Debug)]
pub struct FoldedAttention {
    head_dim: usize,
    scale: f32,
    max_logit: f64,
    denom: f64,
    acc: Vec<f64>,
    logits: Vec<f64>,
    kernel: DotProductKernel,
    consumed_tokens: usize,
}

impl FoldedAttention {
    /// Creates a reducer for a single attention head.
    pub fn new(head_dim: usize, scale: f32) -> Result<Self> {
        if head_dim == 0 {
            return Err(CfrError::InvalidConfig("head_dim must be non-zero"));
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(CfrError::InvalidConfig("scale must be positive and finite"));
        }
        Ok(Self {
            head_dim,
            scale,
            max_logit: f64::NEG_INFINITY,
            denom: 0.0,
            acc: vec![0.0; head_dim],
            logits: Vec::new(),
            kernel: DotProductKernel::default_cpu(),
            consumed_tokens: 0,
        })
    }

    /// Selects a dot-product kernel.
    pub fn set_kernel(&mut self, kernel: DotProductKernel) {
        self.kernel = kernel;
    }

    /// Returns the selected dot-product kernel.
    #[must_use]
    pub const fn kernel(&self) -> DotProductKernel {
        self.kernel
    }

    /// Resets the reducer for a new query.
    pub fn reset(&mut self) {
        self.max_logit = f64::NEG_INFINITY;
        self.denom = 0.0;
        wipe_f64(&mut self.acc);
        wipe_f64(&mut self.logits);
        self.consumed_tokens = 0;
    }

    /// Number of token rows consumed so far.
    #[inline]
    #[must_use]
    pub const fn consumed_tokens(&self) -> usize {
        self.consumed_tokens
    }

    /// Consumes one K/V page transactionally.
    ///
    /// All shape, finite-value and counter checks happen before reducer state is
    /// changed. If this method returns an error, the previously accumulated
    /// attention state is preserved.
    pub fn consume_page(
        &mut self,
        query: &[f32],
        k: &[f32],
        v: &[f32],
        tokens: usize,
    ) -> Result<()> {
        if tokens == 0 {
            return Err(CfrError::InvalidConfig(
                "attention page tokens must be non-zero",
            ));
        }
        expect_len("query", self.head_dim, query.len())?;
        let expected = checked_matrix_len("attention page length", tokens, self.head_dim)?;
        expect_len("page K", expected, k.len())?;
        expect_len("page V", expected, v.len())?;
        expect_all_finite("query contains a non-finite value", query)?;
        expect_all_finite("page K contains a non-finite value", k)?;
        expect_all_finite("page V contains a non-finite value", v)?;

        let new_consumed = checked_add(
            "consumed attention token count",
            self.consumed_tokens,
            tokens,
        )?;

        prepare_zeroed_f64(&mut self.logits, tokens);
        for (index, k_row) in k.chunks_exact(self.head_dim).enumerate() {
            let logit = self.kernel.dot(query, k_row)? * f64::from(self.scale);
            if !logit.is_finite() {
                wipe_f64(&mut self.logits);
                return Err(CfrError::Numeric("attention logit is not finite"));
            }
            self.logits[index] = logit;
        }

        for (row_index, v_row) in v.chunks_exact(self.head_dim).enumerate() {
            let logit = self.logits[row_index];
            self.consume_one(logit, v_row);
        }

        self.consumed_tokens = new_consumed;
        Ok(())
    }

    fn consume_one(&mut self, logit: f64, value: &[f32]) {
        let new_max = self.max_logit.max(logit);
        let old_weight = if self.max_logit.is_finite() {
            (self.max_logit - new_max).exp()
        } else {
            0.0
        };
        let new_weight = (logit - new_max).exp();

        for (acc, value_i) in self.acc.iter_mut().zip(value.iter()) {
            *acc = (*acc).mul_add(old_weight, f64::from(*value_i) * new_weight);
        }

        self.denom = self.denom.mul_add(old_weight, new_weight);
        self.max_logit = new_max;
    }

    /// Writes the normalized attention output into `output` on success only.
    pub fn finish_into(&self, output: &mut [f32]) -> Result<()> {
        expect_len("attention output", self.head_dim, output.len())?;
        if self.consumed_tokens == 0 {
            return Err(CfrError::Numeric(
                "cannot finish attention with zero tokens",
            ));
        }
        if !(self.denom.is_finite() && self.denom > 0.0) {
            return Err(CfrError::Numeric("softmax denominator is invalid"));
        }

        let mut tmp = vec![0.0f32; self.head_dim];
        for (dst, src) in tmp.iter_mut().zip(self.acc.iter()) {
            let value = *src / self.denom;
            *dst = match f64_to_f32_checked("attention output is not finite", value) {
                Ok(value) => value,
                Err(err) => {
                    wipe_f32(&mut tmp);
                    return Err(err);
                }
            };
        }
        output.copy_from_slice(&tmp);
        wipe_f32(&mut tmp);
        Ok(())
    }
}

impl Drop for FoldedAttention {
    fn drop(&mut self) {
        wipe_f64(&mut self.acc);
        wipe_f64(&mut self.logits);
    }
}
