// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! Token ledger for deterministic backend replay.
//!
//! `CFR-Atlas` does not own tokenization, but `Phase 2` needs a stable replay log
//! that backend adapters can use to regenerate exact K/V pages.

use crate::{CfrError, Result};
use std::ops::Range;

/// Stable token identifier used by backend adapters.
pub type TokenId = u32;

/// One replayable token record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenRecord {
    /// Token id produced by the embedding runtime's tokenizer.
    pub token_id: TokenId,
    /// Absolute model position used for `RoPE`, `ALiBi` or plain positional replay.
    pub position: u64,
}

impl TokenRecord {
    /// Creates a token record.
    #[must_use]
    pub const fn new(token_id: TokenId, position: u64) -> Self {
        Self { token_id, position }
    }
}

/// Append-only token ledger used by backend adapters.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct TokenLedger {
    records: Vec<TokenRecord>,
}

impl TokenLedger {
    /// Creates an empty token ledger.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }

    /// Creates a ledger with positions equal to token indices.
    pub fn from_token_ids<I>(token_ids: I) -> Result<Self>
    where
        I: IntoIterator<Item = TokenId>,
    {
        let mut ledger = Self::new();
        for token_id in token_ids {
            ledger.push(token_id)?;
        }
        Ok(ledger)
    }

    /// Appends a token at the next sequential position.
    pub fn push(&mut self, token_id: TokenId) -> Result<usize> {
        let position = if let Some(last) = self.records.last() {
            last.position
                .checked_add(1)
                .ok_or(CfrError::InvalidLedger("token position overflow"))?
        } else {
            0
        };
        self.records.push(TokenRecord::new(token_id, position));
        Ok(self.records.len() - 1)
    }

    /// Appends a token with an explicit absolute position.
    pub fn push_with_position(&mut self, token_id: TokenId, position: u64) -> Result<usize> {
        if let Some(last) = self.records.last() {
            if position <= last.position {
                return Err(CfrError::InvalidLedger(
                    "token positions must be strictly increasing",
                ));
            }
        }
        self.records.push(TokenRecord::new(token_id, position));
        Ok(self.records.len() - 1)
    }

    /// Number of token records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the ledger contains no tokens.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Returns all token records.
    #[must_use]
    pub fn records(&self) -> &[TokenRecord] {
        &self.records
    }

    /// Returns a single token record.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<TokenRecord> {
        self.records.get(index).copied()
    }

    /// Returns a validated token-record range.
    pub fn range(&self, range: Range<usize>) -> Result<&[TokenRecord]> {
        if range.start >= range.end {
            return Err(CfrError::InvalidLedger("token range must be non-empty"));
        }
        if range.end > self.records.len() {
            return Err(CfrError::InvalidLedger("token range exceeds ledger length"));
        }
        Ok(&self.records[range])
    }

    /// Returns token ids as a compact vector.
    #[must_use]
    pub fn token_ids(&self) -> Vec<TokenId> {
        self.records.iter().map(|record| record.token_id).collect()
    }

    /// Returns absolute token positions as a compact vector.
    #[must_use]
    pub fn positions(&self) -> Vec<u64> {
        self.records.iter().map(|record| record.position).collect()
    }
}
