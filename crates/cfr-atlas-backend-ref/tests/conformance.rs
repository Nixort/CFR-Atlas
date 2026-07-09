// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! Reference backend conformance tests for `CFR-Atlas` `Phase 2`.

use cfr_atlas::prelude::*;
use cfr_atlas_backend_ref::ReferenceBackend;

fn backend(
    position_encoding: PositionEncoding,
    dtype_policy: DTypePolicy,
) -> Result<ReferenceBackend> {
    let ledger = TokenLedger::from_token_ids((0..128).map(|i| 1000 + i))?;
    ReferenceBackend::new(
        ledger,
        AttentionTopology::gqa(8, 2)?,
        position_encoding,
        dtype_policy,
        4,
        16,
    )
}

#[test]
fn stored_kv_matches_regenerated_kv_for_rope_bf16() -> Result<()> {
    let rope = PositionEncoding::Rope(RopeConfig::new(10_000.0, 16)?);
    let backend = backend(rope, DTypePolicy::bf16())?;
    let report = backend.conformance_report(PageKey::new(2, 1, 32), 32..64, 0.0)?;
    assert!(report.passed());
    assert!(report.max_abs_k <= f32::EPSILON);
    assert!(report.max_abs_v <= f32::EPSILON);
    Ok(())
}

#[test]
fn reference_backend_runs_through_cfr_atlas() -> Result<()> {
    let backend = backend(PositionEncoding::None, DTypePolicy::f32())?;
    let config = Config::builder(32, backend.head_dim())
        .hot_cache_bytes(0)
        .build()?;
    let mut atlas = CfrAtlas::new(config)?;
    let query = vec![0.125; backend.head_dim()];
    let mut output = vec![0.0; backend.head_dim()];
    atlas.attend_exact(
        &backend,
        AttentionRequest::new(0, 0, &query, 96),
        &mut output,
    )?;
    assert!(output.iter().all(|value| value.is_finite()));
    assert_eq!(atlas.stats().cold_regenerations, 3);
    Ok(())
}

#[test]
fn gqa_mapping_is_visible_to_adapter() -> Result<()> {
    let backend = backend(PositionEncoding::None, DTypePolicy::f16())?;
    let mapping = backend.map_query_head(7)?;
    assert_eq!(mapping.kv_head, 1);
    assert_eq!(mapping.group_start, 4);
    assert_eq!(mapping.group_end, 8);
    Ok(())
}
