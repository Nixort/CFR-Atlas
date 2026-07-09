// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! Attention topology and query-to-`KV` head mapping.
//!
//! `Phase 2` adapters must handle `MHA`, `MQA` and `GQA` without changing the virtual
//! page identity used by the `CFR` core.

use crate::{CfrError, Result};

/// High-level topology kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionTopologyKind {
    /// Multi-head attention: each query head owns a matching K/V head.
    Mha,
    /// Multi-query attention: all query heads share one K/V head.
    Mqa,
    /// Grouped-query attention: query heads are partitioned over fewer K/V heads.
    Gqa,
}

/// Attention-head topology for `MHA`, `MQA` and `GQA`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionTopology {
    kind: AttentionTopologyKind,
    query_heads: u32,
    kv_heads: u32,
}

impl AttentionTopology {
    /// Creates a multi-head attention topology.
    pub const fn mha(heads: u32) -> Result<Self> {
        Self::new(AttentionTopologyKind::Mha, heads, heads)
    }

    /// Creates a multi-query attention topology.
    pub const fn mqa(query_heads: u32) -> Result<Self> {
        Self::new(AttentionTopologyKind::Mqa, query_heads, 1)
    }

    /// Creates a grouped-query attention topology.
    pub const fn gqa(query_heads: u32, kv_heads: u32) -> Result<Self> {
        Self::new(AttentionTopologyKind::Gqa, query_heads, kv_heads)
    }

    const fn new(kind: AttentionTopologyKind, query_heads: u32, kv_heads: u32) -> Result<Self> {
        if query_heads == 0 {
            return Err(CfrError::InvalidTopology("query_heads must be non-zero"));
        }
        if kv_heads == 0 {
            return Err(CfrError::InvalidTopology("kv_heads must be non-zero"));
        }
        if kv_heads > query_heads {
            return Err(CfrError::InvalidTopology("kv_heads must be <= query_heads"));
        }
        match kind {
            AttentionTopologyKind::Mha if query_heads != kv_heads => {
                return Err(CfrError::InvalidTopology(
                    "MHA requires query_heads == kv_heads",
                ));
            }
            AttentionTopologyKind::Mqa if kv_heads != 1 => {
                return Err(CfrError::InvalidTopology("MQA requires kv_heads == 1"));
            }
            AttentionTopologyKind::Gqa if query_heads % kv_heads != 0 => {
                return Err(CfrError::InvalidTopology(
                    "GQA requires query_heads to be divisible by kv_heads",
                ));
            }
            _ => {}
        }
        Ok(Self {
            kind,
            query_heads,
            kv_heads,
        })
    }

    /// Returns the topology kind.
    #[must_use]
    pub const fn kind(&self) -> AttentionTopologyKind {
        self.kind
    }

    /// Number of query heads.
    #[must_use]
    pub const fn query_heads(&self) -> u32 {
        self.query_heads
    }

    /// Number of K/V heads.
    #[must_use]
    pub const fn kv_heads(&self) -> u32 {
        self.kv_heads
    }

    /// Number of query heads served by one K/V head.
    #[must_use]
    pub const fn group_size(&self) -> u32 {
        self.query_heads / self.kv_heads
    }

    /// Maps a query head to its K/V head and group bounds.
    pub const fn map_query_head(&self, query_head: u32) -> Result<HeadMapping> {
        if query_head >= self.query_heads {
            return Err(CfrError::InvalidTopology("query head is out of range"));
        }
        let group_size = self.group_size();
        let kv_head = match self.kind {
            AttentionTopologyKind::Mha => query_head,
            AttentionTopologyKind::Mqa => 0,
            AttentionTopologyKind::Gqa => query_head / group_size,
        };
        let group_start = kv_head * group_size;
        let group_end = group_start + group_size;
        Ok(HeadMapping {
            topology: self.kind,
            query_head,
            kv_head,
            group_start,
            group_end,
        })
    }
}

/// Result of mapping a query head to the physical K/V head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadMapping {
    /// Topology kind used to produce this mapping.
    pub topology: AttentionTopologyKind,
    /// Query head requested by the model runtime.
    pub query_head: u32,
    /// K/V head that stores or regenerates the page.
    pub kv_head: u32,
    /// Inclusive start of the query-head group served by this K/V head.
    pub group_start: u32,
    /// Exclusive end of the query-head group served by this K/V head.
    pub group_end: u32,
}
