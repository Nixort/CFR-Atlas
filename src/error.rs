// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 6 july 2026

//! Error types returned by the `CFR-Atlas` core.
//!
//! This module keeps configuration, dimension, page, backend, capacity and
//! numeric errors explicit for production integrations.

use crate::page::PageKey;
use std::{error::Error, fmt};

/// Error type returned by `CFR-Atlas` operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CfrError {
    /// Static invalid configuration.
    InvalidConfig(&'static str),
    /// Runtime dimension mismatch.
    Dimension {
        /// Name of the checked value.
        name: &'static str,
        /// Expected length.
        expected: usize,
        /// Actual length.
        got: usize,
    },
    /// A size computation overflowed before allocation or indexing.
    CapacityOverflow {
        /// Name of the computation that overflowed.
        name: &'static str,
    },
    /// Invalid page identity or page shape.
    InvalidPage {
        /// Page that failed validation.
        key: PageKey,
        /// Human-readable reason.
        message: &'static str,
    },
    /// Invalid token ledger operation.
    InvalidLedger(&'static str),
    /// Invalid attention topology or head mapping.
    InvalidTopology(&'static str),
    /// Backend regeneration failed.
    Regenerator(String),
    /// Numeric failure in attention reduction.
    Numeric(&'static str),
}

impl fmt::Display for CfrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid CFR-Atlas config: {msg}"),
            Self::Dimension {
                name,
                expected,
                got,
            } => {
                write!(
                    f,
                    "dimension mismatch for {name}: expected {expected}, got {got}"
                )
            }
            Self::CapacityOverflow { name } => {
                write!(f, "capacity overflow while computing {name}")
            }
            Self::InvalidPage { key, message } => write!(f, "invalid page {key:?}: {message}"),
            Self::InvalidLedger(msg) => write!(f, "invalid token ledger: {msg}"),
            Self::InvalidTopology(msg) => write!(f, "invalid attention topology: {msg}"),
            Self::Regenerator(msg) => write!(f, "KV regeneration failed: {msg}"),
            Self::Numeric(msg) => write!(f, "numeric failure: {msg}"),
        }
    }
}

impl Error for CfrError {}

/// `CFR-Atlas` result alias.
pub type Result<T> = std::result::Result<T, CfrError>;
