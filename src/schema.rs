// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! Versioned configuration schema for stable embedding and release upgrades.
//!
//! `CFR-Atlas` keeps its runtime [`Config`] intentionally small. This module
//! gives that config a deterministic, dependency-free text representation so
//! embedders can store, diff and migrate settings without relying on Rust struct
//! layout or debug formatting.

use crate::{CfrError, Config, Result};

/// Current version of the `CFR-Atlas` configuration schema.
pub const CONFIG_SCHEMA_VERSION: u32 = 1;

const MAGIC: &str = "cfr_atlas_config_schema";

/// Runtime configuration bundled with an explicit schema version.
#[derive(Debug, Clone, PartialEq)]
pub struct VersionedConfig {
    schema_version: u32,
    config: Config,
}

impl VersionedConfig {
    /// Wraps a validated [`Config`] with the current schema version.
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            schema_version: CONFIG_SCHEMA_VERSION,
            config,
        })
    }

    /// Returns the schema version carried by this value.
    #[must_use]
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Returns the wrapped runtime config.
    #[must_use]
    pub const fn config(&self) -> &Config {
        &self.config
    }

    /// Consumes the wrapper and returns the runtime config.
    #[must_use]
    pub const fn into_config(self) -> Config {
        self.config
    }

    /// Encodes the configuration as deterministic newline-separated key/value pairs.
    #[must_use]
    pub fn encode(&self) -> String {
        format!(
            concat!(
                "{}={}\n",
                "page_tokens={}\n",
                "head_dim={}\n",
                "hot_cache_bytes={}\n",
                "scale={:?}\n",
                "admit_regenerated_pages={}\n",
                "max_scratch_tokens={}\n",
                "wipe_scratch_after_use={}\n",
            ),
            MAGIC,
            self.schema_version,
            self.config.page_tokens,
            self.config.head_dim,
            self.config.hot_cache_bytes,
            self.config.scale,
            self.config.admit_regenerated_pages,
            self.config.max_scratch_tokens,
            self.config.wipe_scratch_after_use,
        )
    }

    /// Decodes and validates a versioned configuration.
    pub fn decode(input: &str) -> Result<Self> {
        let parsed = ParsedConfig::from_text(input)?;
        let schema_version = require_u32(parsed.schema_version, "missing config schema version")?;
        if schema_version != CONFIG_SCHEMA_VERSION {
            return Err(CfrError::InvalidConfig("unsupported config schema version"));
        }
        let config = Config {
            page_tokens: require_usize(parsed.page_tokens, "missing page_tokens")?,
            head_dim: require_usize(parsed.head_dim, "missing head_dim")?,
            hot_cache_bytes: require_usize(parsed.hot_cache_bytes, "missing hot_cache_bytes")?,
            scale: require_f32(parsed.scale, "missing scale")?,
            admit_regenerated_pages: require_bool(
                parsed.admit_regenerated_pages,
                "missing admit_regenerated_pages",
            )?,
            max_scratch_tokens: require_usize(
                parsed.max_scratch_tokens,
                "missing max_scratch_tokens",
            )?,
            wipe_scratch_after_use: require_bool(
                parsed.wipe_scratch_after_use,
                "missing wipe_scratch_after_use",
            )?,
        };
        Self::new(config)
    }
}

#[derive(Default)]
struct ParsedConfig {
    schema_version: Option<u32>,
    page_tokens: Option<usize>,
    head_dim: Option<usize>,
    hot_cache_bytes: Option<usize>,
    scale: Option<f32>,
    admit_regenerated_pages: Option<bool>,
    max_scratch_tokens: Option<usize>,
    wipe_scratch_after_use: Option<bool>,
}

impl ParsedConfig {
    fn from_text(input: &str) -> Result<Self> {
        let mut parsed = Self::default();
        for raw_line in input.lines() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                return Err(CfrError::InvalidConfig("invalid config schema line"));
            };
            parsed.accept_field(key.trim(), value.trim())?;
        }
        Ok(parsed)
    }

    fn accept_field(&mut self, key: &str, value: &str) -> Result<()> {
        match key {
            MAGIC => set_once(
                &mut self.schema_version,
                parse_u32(value, "invalid config schema version")?,
                "duplicate config schema version",
            )?,
            "page_tokens" => set_once(
                &mut self.page_tokens,
                parse_usize(value, "invalid page_tokens")?,
                "duplicate page_tokens",
            )?,
            "head_dim" => set_once(
                &mut self.head_dim,
                parse_usize(value, "invalid head_dim")?,
                "duplicate head_dim",
            )?,
            "hot_cache_bytes" => set_once(
                &mut self.hot_cache_bytes,
                parse_usize(value, "invalid hot_cache_bytes")?,
                "duplicate hot_cache_bytes",
            )?,
            "scale" => set_once(
                &mut self.scale,
                parse_f32(value, "invalid scale")?,
                "duplicate scale",
            )?,
            "admit_regenerated_pages" => set_once(
                &mut self.admit_regenerated_pages,
                parse_bool(value)?,
                "duplicate admit_regenerated_pages",
            )?,
            "max_scratch_tokens" => set_once(
                &mut self.max_scratch_tokens,
                parse_usize(value, "invalid max_scratch_tokens")?,
                "duplicate max_scratch_tokens",
            )?,
            "wipe_scratch_after_use" => set_once(
                &mut self.wipe_scratch_after_use,
                parse_bool(value)?,
                "duplicate wipe_scratch_after_use",
            )?,
            _ => return Err(CfrError::InvalidConfig("unknown config schema field")),
        }
        Ok(())
    }
}

fn set_once<T>(slot: &mut Option<T>, value: T, error: &'static str) -> Result<()> {
    if slot.is_some() {
        return Err(CfrError::InvalidConfig(error));
    }
    *slot = Some(value);
    Ok(())
}

fn parse_u32(value: &str, error: &'static str) -> Result<u32> {
    value.parse().map_err(|_| CfrError::InvalidConfig(error))
}

fn parse_usize(value: &str, error: &'static str) -> Result<usize> {
    value.parse().map_err(|_| CfrError::InvalidConfig(error))
}

fn parse_f32(value: &str, error: &'static str) -> Result<f32> {
    let parsed: f32 = value.parse().map_err(|_| CfrError::InvalidConfig(error))?;
    if !parsed.is_finite() || parsed <= 0.0 {
        return Err(CfrError::InvalidConfig("scale must be positive and finite"));
    }
    Ok(parsed)
}

fn parse_bool(value: &str) -> Result<bool> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(CfrError::InvalidConfig("invalid boolean config value")),
    }
}

fn require_u32(value: Option<u32>, error: &'static str) -> Result<u32> {
    value.ok_or(CfrError::InvalidConfig(error))
}

fn require_usize(value: Option<usize>, error: &'static str) -> Result<usize> {
    value.ok_or(CfrError::InvalidConfig(error))
}

fn require_f32(value: Option<f32>, error: &'static str) -> Result<f32> {
    value.ok_or(CfrError::InvalidConfig(error))
}

fn require_bool(value: Option<bool>, error: &'static str) -> Result<bool> {
    value.ok_or(CfrError::InvalidConfig(error))
}
