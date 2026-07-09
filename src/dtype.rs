// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! Deterministic dtype policy for backend adapters.
//!
//! The reference `CFR` core consumes `f32`, but `Phase 2` adapters may need to
//! emulate storage rounding used by `f32`, `bf16` or `f16` CPU backends.

use crate::{CfrError, Result};

/// Storage dtype emulated by a backend adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageDType {
    /// Native `f32` values with no storage rounding.
    F32,
    /// Brain floating point 16, rounded to nearest even then expanded to `f32`.
    Bf16,
    /// IEEE-754 binary16, rounded to nearest even then expanded to `f32`.
    F16,
}

/// Accumulator dtype requested by a backend adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumulatorDType {
    /// Accumulate in `f32`.
    F32,
    /// Accumulate in `f64`.
    F64,
}

/// Deterministic dtype policy for K/V regeneration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DTypePolicy {
    storage: StorageDType,
    accumulator: AccumulatorDType,
}

impl DTypePolicy {
    /// Creates a dtype policy.
    #[must_use]
    pub const fn new(storage: StorageDType, accumulator: AccumulatorDType) -> Self {
        Self {
            storage,
            accumulator,
        }
    }

    /// Pure `f32` policy.
    #[must_use]
    pub const fn f32() -> Self {
        Self::new(StorageDType::F32, AccumulatorDType::F64)
    }

    /// `bf16` storage-emulation policy.
    #[must_use]
    pub const fn bf16() -> Self {
        Self::new(StorageDType::Bf16, AccumulatorDType::F64)
    }

    /// `f16` storage-emulation policy.
    #[must_use]
    pub const fn f16() -> Self {
        Self::new(StorageDType::F16, AccumulatorDType::F64)
    }

    /// Storage dtype.
    #[must_use]
    pub const fn storage(&self) -> StorageDType {
        self.storage
    }

    /// Accumulator dtype.
    #[must_use]
    pub const fn accumulator(&self) -> AccumulatorDType {
        self.accumulator
    }

    /// Applies deterministic storage rounding to one value.
    #[must_use]
    pub fn round_f32(&self, value: f32) -> f32 {
        match self.storage {
            StorageDType::F32 => value,
            StorageDType::Bf16 => round_to_bf16(value),
            StorageDType::F16 => round_to_f16(value),
        }
    }

    /// Applies deterministic storage rounding in place.
    pub fn round_slice_in_place(&self, values: &mut [f32]) {
        for value in values {
            *value = self.round_f32(*value);
        }
    }

    /// Validates policy consistency.
    pub const fn validate(&self) -> Result<()> {
        let _ = self;
        Ok(())
    }
}

impl Default for DTypePolicy {
    fn default() -> Self {
        Self::f32()
    }
}

fn round_to_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        return value;
    }
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7fff + lsb) & 0xffff_0000;
    f32::from_bits(rounded)
}

fn round_to_f16(value: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(value))
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = u16_from_u32_or((bits >> 16) & 0x8000, 0x8000);
    let exp = i32_from_u32_or((bits >> 23) & 0xff, 0xff);
    let mant = bits & 0x007f_ffff;

    if exp == 0xff {
        return if mant == 0 {
            sign | 0x7c00
        } else {
            sign | 0x7e00
        };
    }

    let half_exp = exp - 127 + 15;
    if half_exp >= 0x1f {
        return sign | 0x7c00;
    }

    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = mant | 0x0080_0000;
        let shift = u32_from_i32_or(14 - half_exp, 0);
        let mut half_mant = u16_from_u32_or(mantissa >> shift, u16::MAX);
        let round_bit = (mantissa >> (shift - 1)) & 1;
        let sticky_mask = (1u32 << (shift - 1)) - 1;
        let sticky = mantissa & sticky_mask;
        if round_bit != 0 && (sticky != 0 || (half_mant & 1) != 0) {
            half_mant = half_mant.saturating_add(1);
        }
        return sign | half_mant;
    }

    let half_exp_bits = u16_from_i32_or(half_exp, 0x1f);
    let mant_bits = u16_from_u32_or(mant >> 13, 0x03ff);
    let mut half = sign | (half_exp_bits << 10) | mant_bits;
    let round = mant & 0x1fff;
    if round > 0x1000 || (round == 0x1000 && (half & 1) != 0) {
        half = half.saturating_add(1);
    }
    half
}

fn u16_from_u32_or(value: u32, fallback: u16) -> u16 {
    u16::try_from(value).map_or(fallback, |value| value)
}

fn u16_from_i32_or(value: i32, fallback: u16) -> u16 {
    u16::try_from(value).map_or(fallback, |value| value)
}

fn i32_from_u32_or(value: u32, fallback: i32) -> i32 {
    i32::try_from(value).map_or(fallback, |value| value)
}

fn u32_from_i32_or(value: i32, fallback: u32) -> u32 {
    u32::try_from(value).map_or(fallback, |value| value)
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x03ff;

    match exp {
        0 => {
            if mant == 0 {
                f32::from_bits(sign)
            } else {
                let magnitude = f32::from(mant) * 2.0_f32.powi(-24);
                if sign == 0 {
                    magnitude
                } else {
                    -magnitude
                }
            }
        }
        0x1f => {
            if mant == 0 {
                f32::from_bits(sign | 0x7f80_0000)
            } else {
                f32::from_bits(sign | 0x7fc0_0000)
            }
        }
        _ => {
            let exp32 = u32::from(exp + 112) << 23;
            let mant32 = u32::from(mant) << 13;
            f32::from_bits(sign | exp32 | mant32)
        }
    }
}

/// Converts invalid external dtype names into a `CFR` error.
pub fn parse_storage_dtype(name: &str) -> Result<StorageDType> {
    match name {
        "f32" | "float32" => Ok(StorageDType::F32),
        "bf16" | "bfloat16" => Ok(StorageDType::Bf16),
        "f16" | "float16" => Ok(StorageDType::F16),
        _ => Err(CfrError::InvalidConfig("unknown storage dtype")),
    }
}
