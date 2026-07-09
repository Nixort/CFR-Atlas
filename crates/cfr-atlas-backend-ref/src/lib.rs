// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! Reference backend adapter for `CFR-Atlas` `Phase 2`.
//!
//! This crate is intentionally small and deterministic. It is not a real LLM
//! backend; it is a production-shaped adapter boundary that demonstrates token
//! ledger replay, `MHA`/`MQA`/`GQA` head mapping, positional preservation, dtype
//! policy and stored-`KV` conformance checks.

use cfr_atlas::prelude::*;
use std::ops::Range;

/// Deterministic reference backend used for adapter conformance tests.
#[derive(Debug, Clone)]
pub struct ReferenceBackend {
    ledger: TokenLedger,
    topology: AttentionTopology,
    position_encoding: PositionEncoding,
    dtype_policy: DTypePolicy,
    layers: u32,
    head_dim: usize,
}

impl ReferenceBackend {
    /// Creates a new reference backend.
    pub fn new(
        ledger: TokenLedger,
        topology: AttentionTopology,
        position_encoding: PositionEncoding,
        dtype_policy: DTypePolicy,
        layers: u32,
        head_dim: usize,
    ) -> Result<Self> {
        if ledger.is_empty() {
            return Err(CfrError::InvalidLedger("reference backend requires tokens"));
        }
        if layers == 0 {
            return Err(CfrError::InvalidConfig("layers must be non-zero"));
        }
        if head_dim == 0 {
            return Err(CfrError::InvalidConfig("head_dim must be non-zero"));
        }
        dtype_policy.validate()?;
        Ok(Self {
            ledger,
            topology,
            position_encoding,
            dtype_policy,
            layers,
            head_dim,
        })
    }

    /// Returns the token ledger.
    #[must_use]
    pub const fn ledger(&self) -> &TokenLedger {
        &self.ledger
    }

    /// Returns the attention topology.
    #[must_use]
    pub const fn topology(&self) -> AttentionTopology {
        self.topology
    }

    /// Returns the position encoding policy.
    #[must_use]
    pub const fn position_encoding(&self) -> &PositionEncoding {
        &self.position_encoding
    }

    /// Returns the dtype policy.
    #[must_use]
    pub const fn dtype_policy(&self) -> DTypePolicy {
        self.dtype_policy
    }

    /// Returns the number of transformer layers modeled by the adapter.
    #[must_use]
    pub const fn layers(&self) -> u32 {
        self.layers
    }

    /// Returns the head dimension.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Maps a query head to the K/V head used by `CFR` page identity.
    pub const fn map_query_head(&self, query_head: u32) -> Result<HeadMapping> {
        self.topology.map_query_head(query_head)
    }

    /// Materializes a stored K/V page through the baseline path.
    ///
    /// The reference backend deliberately uses the same primitive row generator
    /// as regeneration. Real backend adapters should wire this method to their
    /// classic stored-`KV` path during conformance testing.
    pub fn stored_page(
        &self,
        key: PageKey,
        token_range: Range<usize>,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<()> {
        self.generate_page(key, token_range, self.head_dim, k_out, v_out)
    }

    /// Runs stored-`KV` vs regenerated-KV conformance for one page.
    pub fn conformance_report(
        &self,
        key: PageKey,
        token_range: Range<usize>,
        tolerance: f32,
    ) -> Result<PageConformance> {
        self.validate_range_key(key, &token_range)?;
        let tokens = checked_range_len("reference backend range length", &token_range)?;
        let expected = checked_matrix_len("reference stored page", tokens, self.head_dim)?;
        let mut stored_k = vec![0.0; expected];
        let mut stored_v = vec![0.0; expected];
        if let Err(err) = self.stored_page(key, token_range.clone(), &mut stored_k, &mut stored_v) {
            wipe_f32(&mut stored_k);
            wipe_f32(&mut stored_v);
            return Err(err);
        }
        let report = compare_regenerated_page(
            self,
            key,
            token_range,
            self.head_dim,
            &stored_k,
            &stored_v,
            tolerance,
        );
        wipe_f32(&mut stored_k);
        wipe_f32(&mut stored_v);
        report
    }

    fn generate_page(
        &self,
        key: PageKey,
        token_range: Range<usize>,
        head_dim: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<()> {
        if let Err(err) = self.validate_request(key, token_range.clone(), head_dim, k_out, v_out) {
            wipe_f32(k_out);
            wipe_f32(v_out);
            return Err(err);
        }
        let records = match self.ledger.range(token_range) {
            Ok(records) => records,
            Err(err) => {
                wipe_f32(k_out);
                wipe_f32(v_out);
                return Err(err);
            }
        };
        for (row_index, record) in records.iter().enumerate() {
            let row_range = match checked_row_range(
                "reference backend row range",
                row_index,
                head_dim,
                k_out.len(),
            ) {
                Ok(row_range) => row_range,
                Err(err) => {
                    wipe_f32(k_out);
                    wipe_f32(v_out);
                    return Err(err);
                }
            };
            if let Err(err) = self.generate_row(
                key,
                *record,
                &mut k_out[row_range.clone()],
                &mut v_out[row_range],
            ) {
                wipe_f32(k_out);
                wipe_f32(v_out);
                return Err(err);
            }
        }
        Ok(())
    }

    fn validate_request(
        &self,
        key: PageKey,
        token_range: Range<usize>,
        head_dim: usize,
        k_out: &[f32],
        v_out: &[f32],
    ) -> Result<()> {
        self.validate_range_key(key, &token_range)?;
        if key.layer >= self.layers {
            return Err(CfrError::InvalidPage {
                key,
                message: "layer is out of range",
            });
        }
        if key.head >= self.topology.kv_heads() {
            return Err(CfrError::InvalidPage {
                key,
                message: "KV head is out of range",
            });
        }
        if head_dim != self.head_dim {
            return Err(CfrError::Dimension {
                name: "head_dim",
                expected: self.head_dim,
                got: head_dim,
            });
        }
        let tokens = checked_range_len("reference backend range length", &token_range)?;
        let expected =
            checked_matrix_len("reference backend output length", tokens, self.head_dim)?;
        expect_len("backend K output", expected, k_out.len())?;
        expect_len("backend V output", expected, v_out.len())?;
        Ok(())
    }

    fn validate_range_key(&self, key: PageKey, token_range: &Range<usize>) -> Result<()> {
        if token_range.start != key.start_token {
            return Err(CfrError::InvalidPage {
                key,
                message: "range start must equal key.start_token",
            });
        }
        if token_range.start >= token_range.end {
            return Err(CfrError::InvalidPage {
                key,
                message: "token range must be non-empty",
            });
        }
        if token_range.end > self.ledger.len() {
            return Err(CfrError::InvalidLedger("token range exceeds ledger length"));
        }
        Ok(())
    }

    fn generate_row(
        &self,
        key: PageKey,
        record: TokenRecord,
        k_row: &mut [f32],
        v_row: &mut [f32],
    ) -> Result<()> {
        let token = u32_to_f32_checked("token id must fit exact f32", record.token_id)?;
        let position = u64_to_f32_checked("token position must fit f32", record.position)?;
        let layer = u32_to_f32_checked("layer index must fit exact f32", key.layer)?;
        let head = u32_to_f32_checked("head index must fit exact f32", key.head)?;

        for (dim, (k_dst, v_dst)) in k_row.iter_mut().zip(v_row.iter_mut()).enumerate() {
            let dim_f32 = usize_to_f32_checked("dimension index must fit exact f32", dim)?;
            let phase = token.mul_add(
                0.013_671_875,
                position.mul_add(
                    0.001_953_125,
                    layer.mul_add(
                        0.173_828_13,
                        head.mul_add(0.287_109_38, dim_f32 * 0.039_062_5),
                    ),
                ),
            );
            *k_dst = phase.sin().mul_add(0.75, (dim_f32 + 1.0).recip());
            *v_dst = phase.cos().mul_add(0.50, (token + 3.0).recip());
        }

        self.position_encoding.apply_key(record.position, k_row)?;
        self.dtype_policy.round_slice_in_place(k_row);
        self.dtype_policy.round_slice_in_place(v_row);
        Ok(())
    }
}

impl KvRegenerator for ReferenceBackend {
    fn regenerate_page(
        &self,
        key: PageKey,
        token_range: Range<usize>,
        head_dim: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<()> {
        self.generate_page(key, token_range, head_dim, k_out, v_out)
    }
}
