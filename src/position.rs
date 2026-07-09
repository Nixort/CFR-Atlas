// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! Positional policy used by backend adapters.
//!
//! The `CFR` core does not prescribe `RoPE` or `ALiBi`, but it provides small,
//! deterministic building blocks so adapters can preserve positional semantics.

use crate::layout::{f64_to_f32_checked, u64_to_f64_checked, usize_to_f64_checked};
use crate::{CfrError, Result};

/// Position encoding selected by a backend adapter.
#[derive(Debug, Clone, PartialEq)]
pub enum PositionEncoding {
    /// No positional transform in the adapter boundary.
    None,
    /// Rotary positional embedding applied to K rows during regeneration.
    Rope(RopeConfig),
    /// `ALiBi` metadata. `ALiBi` affects attention logits, not regenerated K rows.
    Alibi(AlibiConfig),
}

impl PositionEncoding {
    /// Applies the key-side positional transform for one regenerated row.
    pub fn apply_key(&self, position: u64, key_row: &mut [f32]) -> Result<()> {
        match self {
            Self::None | Self::Alibi(_) => Ok(()),
            Self::Rope(config) => config.apply_key(position, key_row),
        }
    }

    /// Returns `ALiBi` bias if this encoding carries `ALiBi` metadata.
    pub fn alibi_bias(&self, head: u32, query_position: u64, key_position: u64) -> Result<f32> {
        match self {
            Self::Alibi(config) => config.bias(head, query_position, key_position),
            Self::None | Self::Rope(_) => Ok(0.0),
        }
    }
}

impl Default for PositionEncoding {
    fn default() -> Self {
        Self::None
    }
}

/// `RoPE` configuration for deterministic adapter replay.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeConfig {
    /// Frequency base, commonly `10000.0`.
    pub base: f32,
    /// Number of leading dimensions affected by `RoPE`.
    pub dims: usize,
}

impl RopeConfig {
    /// Creates and validates a `RoPE` configuration.
    pub fn new(base: f32, dims: usize) -> Result<Self> {
        if !base.is_finite() || base <= 1.0 {
            return Err(CfrError::InvalidConfig("RoPE base must be finite and > 1"));
        }
        if dims == 0 || dims % 2 != 0 {
            return Err(CfrError::InvalidConfig(
                "RoPE dims must be non-zero and even",
            ));
        }
        Ok(Self { base, dims })
    }

    /// Applies `RoPE` to one key row in place.
    pub fn apply_key(&self, position: u64, key_row: &mut [f32]) -> Result<()> {
        if self.dims > key_row.len() {
            return Err(CfrError::Dimension {
                name: "RoPE dims",
                expected: self.dims,
                got: key_row.len(),
            });
        }
        let position_f64 =
            u64_to_f64_checked("RoPE position exceeds exact f64 integer range", position)?;
        let base = f64::from(self.base);
        let dims = usize_to_f64_checked("RoPE dims exceed exact f64 integer range", self.dims)?;
        for pair in (0..self.dims).step_by(2) {
            let pair_f64 =
                usize_to_f64_checked("RoPE pair index exceeds exact f64 integer range", pair)?;
            let exponent = pair_f64 / dims;
            let angle = position_f64 / base.powf(exponent);
            let (sine, cosine) = angle.sin_cos();
            let x0 = f64::from(key_row[pair]);
            let x1 = f64::from(key_row[pair + 1]);
            key_row[pair] = f64_to_f32_checked(
                "RoPE transformed key value is outside f32 range",
                x0.mul_add(cosine, -(x1 * sine)),
            )?;
            key_row[pair + 1] = f64_to_f32_checked(
                "RoPE transformed key value is outside f32 range",
                x0.mul_add(sine, x1 * cosine),
            )?;
        }
        Ok(())
    }
}

/// `ALiBi` slope table used by adapters that need deterministic logit bias replay.
#[derive(Debug, Clone, PartialEq)]
pub struct AlibiConfig {
    slopes: Vec<f32>,
}

impl AlibiConfig {
    /// Creates an `ALiBi` config from explicit per-K/V-head slopes.
    pub fn new(slopes: Vec<f32>) -> Result<Self> {
        if slopes.is_empty() {
            return Err(CfrError::InvalidConfig("ALiBi slopes must be non-empty"));
        }
        if slopes.iter().any(|slope| !slope.is_finite()) {
            return Err(CfrError::InvalidConfig("ALiBi slopes must be finite"));
        }
        if slopes.iter().any(|slope| *slope < 0.0) {
            return Err(CfrError::InvalidConfig("ALiBi slopes must be non-negative"));
        }
        Ok(Self { slopes })
    }

    /// Returns all slopes.
    #[must_use]
    pub fn slopes(&self) -> &[f32] {
        &self.slopes
    }

    /// Returns the slope for one K/V head.
    pub fn slope(&self, head: u32) -> Result<f32> {
        let index = usize::try_from(head)
            .map_err(|_| CfrError::InvalidTopology("head index does not fit usize"))?;
        self.slopes
            .get(index)
            .copied()
            .ok_or(CfrError::InvalidTopology("ALiBi head is out of range"))
    }

    /// Computes the causal `ALiBi` bias for `(query_position, key_position)`.
    pub fn bias(&self, head: u32, query_position: u64, key_position: u64) -> Result<f32> {
        if key_position > query_position {
            return Err(CfrError::InvalidConfig(
                "ALiBi key_position must not exceed query_position",
            ));
        }
        let distance = u64_to_f64_checked(
            "ALiBi distance exceeds exact f64 integer range",
            query_position - key_position,
        )?;
        let bias = -f64::from(self.slope(head)?) * distance;
        f64_to_f32_checked("ALiBi bias is outside f32 range", bias)
    }
}
