// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 7 july 2026

//! `CFR-Atlas` runtime configuration and builder validation.
//!
//! This module defines the page size, head dimension, hot-cache budget and
//! scratch behavior used by the exact attention core.

use crate::layout::{
    checked_kv_bytes, checked_matrix_len, f64_to_f32_checked, usize_to_f64_checked,
};
use crate::{CfrError, Result};

/// Runtime configuration for the `CFR-Atlas` attention core.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Number of tokens per virtual `KV` page.
    pub page_tokens: usize,
    /// Attention head dimension.
    pub head_dim: usize,
    /// Maximum resident hot-cache bytes for K/V pages.
    pub hot_cache_bytes: usize,
    /// Attention logit scale, usually `1.0 / sqrt(head_dim)`.
    pub scale: f32,
    /// Whether regenerated cold pages may be admitted into hot cache.
    pub admit_regenerated_pages: bool,
    /// Upper bound for scratch page tokens. Normally equals `page_tokens`.
    pub max_scratch_tokens: usize,
    /// Whether cold scratch buffers are zeroed after every page.
    pub wipe_scratch_after_use: bool,
}

impl Config {
    /// Creates a validated configuration with a standard attention scale.
    pub fn new(page_tokens: usize, head_dim: usize, hot_cache_bytes: usize) -> Result<Self> {
        Self::builder(page_tokens, head_dim)
            .hot_cache_bytes(hot_cache_bytes)
            .build()
    }

    /// Starts a builder with required page and head dimensions.
    #[must_use]
    pub const fn builder(page_tokens: usize, head_dim: usize) -> ConfigBuilder {
        ConfigBuilder {
            page_tokens,
            head_dim,
            hot_cache_bytes: 0,
            scale: None,
            admit_regenerated_pages: false,
            max_scratch_tokens: None,
            wipe_scratch_after_use: true,
        }
    }

    /// Number of `f32` values required for one K or V matrix page.
    pub fn page_f32_len(&self, tokens: usize) -> Result<usize> {
        checked_matrix_len("page f32 length", tokens, self.head_dim)
    }

    /// Number of bytes required for K and V together for `tokens` rows.
    pub fn kv_page_bytes(&self, tokens: usize) -> Result<usize> {
        checked_kv_bytes("KV page bytes", tokens, self.head_dim)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.page_tokens == 0 {
            return Err(CfrError::InvalidConfig("page_tokens must be non-zero"));
        }
        if self.head_dim == 0 {
            return Err(CfrError::InvalidConfig("head_dim must be non-zero"));
        }
        if self.max_scratch_tokens == 0 {
            return Err(CfrError::InvalidConfig(
                "max_scratch_tokens must be non-zero",
            ));
        }
        if self.page_tokens > self.max_scratch_tokens {
            return Err(CfrError::InvalidConfig(
                "page_tokens must be <= max_scratch_tokens",
            ));
        }
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(CfrError::InvalidConfig("scale must be positive and finite"));
        }
        self.page_f32_len(self.page_tokens)?;
        self.page_f32_len(self.max_scratch_tokens)?;
        self.kv_page_bytes(self.page_tokens)?;
        Ok(())
    }
}

/// Builder for [`Config`].
#[derive(Debug, Clone)]
pub struct ConfigBuilder {
    page_tokens: usize,
    head_dim: usize,
    hot_cache_bytes: usize,
    scale: Option<f32>,
    admit_regenerated_pages: bool,
    max_scratch_tokens: Option<usize>,
    wipe_scratch_after_use: bool,
}

impl ConfigBuilder {
    /// Sets the resident hot-cache budget in bytes.
    #[must_use]
    pub const fn hot_cache_bytes(mut self, bytes: usize) -> Self {
        self.hot_cache_bytes = bytes;
        self
    }

    /// Sets a custom attention scale.
    #[must_use]
    pub const fn scale(mut self, scale: f32) -> Self {
        self.scale = Some(scale);
        self
    }

    /// Allows or disables insertion of regenerated pages into the hot cache.
    #[must_use]
    pub const fn admit_regenerated_pages(mut self, enabled: bool) -> Self {
        self.admit_regenerated_pages = enabled;
        self
    }

    /// Sets the maximum scratch token capacity.
    #[must_use]
    pub const fn max_scratch_tokens(mut self, tokens: usize) -> Self {
        self.max_scratch_tokens = Some(tokens);
        self
    }

    /// Enables or disables wiping scratch buffers after each cold page.
    #[must_use]
    pub const fn wipe_scratch_after_use(mut self, enabled: bool) -> Self {
        self.wipe_scratch_after_use = enabled;
        self
    }

    /// Builds and validates the configuration.
    pub fn build(self) -> Result<Config> {
        let scale = match self.scale {
            Some(scale) => scale,
            None => default_attention_scale(self.head_dim)?,
        };
        let config = Config {
            page_tokens: self.page_tokens,
            head_dim: self.head_dim,
            hot_cache_bytes: self.hot_cache_bytes,
            scale,
            admit_regenerated_pages: self.admit_regenerated_pages,
            max_scratch_tokens: self
                .max_scratch_tokens
                .map_or(self.page_tokens, |tokens| tokens),
            wipe_scratch_after_use: self.wipe_scratch_after_use,
        };
        config.validate()?;
        Ok(config)
    }
}

fn default_attention_scale(head_dim: usize) -> Result<f32> {
    let head_dim = usize_to_f64_checked("head_dim must fit exact f64 scale math", head_dim)?;
    f64_to_f32_checked(
        "default attention scale is outside f32 range",
        1.0 / head_dim.sqrt(),
    )
}
