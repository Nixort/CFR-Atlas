// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! Adapter core tests for ledger, topology, position and dtype policies.

use cfr_atlas::prelude::*;

#[test]
fn token_ledger_preserves_ids_and_positions() -> Result<()> {
    let mut ledger = TokenLedger::new();
    ledger.push(10)?;
    ledger.push(11)?;
    ledger.push_with_position(12, 10)?;

    assert_eq!(ledger.len(), 3);
    assert_eq!(ledger.token_ids(), vec![10, 11, 12]);
    assert_eq!(ledger.positions(), vec![0, 1, 10]);
    assert_eq!(ledger.range(1..3)?[0], TokenRecord::new(11, 1));
    Ok(())
}

#[test]
fn topology_maps_mha_mqa_and_gqa() -> Result<()> {
    assert_eq!(AttentionTopology::mha(4)?.map_query_head(3)?.kv_head, 3);
    assert_eq!(AttentionTopology::mqa(8)?.map_query_head(7)?.kv_head, 0);

    let mapping = AttentionTopology::gqa(16, 4)?.map_query_head(10)?;
    assert_eq!(mapping.kv_head, 2);
    assert_eq!(mapping.group_start, 8);
    assert_eq!(mapping.group_end, 12);
    Ok(())
}

#[test]
fn rope_changes_key_rows_deterministically() -> Result<()> {
    let rope = RopeConfig::new(10_000.0, 4)?;
    let mut row_a = vec![1.0, 0.0, 0.5, -0.25];
    let mut row_b = row_a.clone();
    rope.apply_key(7, &mut row_a)?;
    rope.apply_key(7, &mut row_b)?;
    assert_eq!(row_a, row_b);
    assert_ne!(row_a, vec![1.0, 0.0, 0.5, -0.25]);
    Ok(())
}

#[test]
fn dtype_policy_rounding_is_deterministic() {
    let values = [0.1, -0.2, 1.337, 65504.0];
    for policy in [DTypePolicy::f32(), DTypePolicy::bf16(), DTypePolicy::f16()] {
        let rounded_once: Vec<f32> = values
            .iter()
            .map(|value| policy.round_f32(*value))
            .collect();
        let rounded_twice: Vec<f32> = rounded_once
            .iter()
            .map(|value| policy.round_f32(*value))
            .collect();
        assert_eq!(rounded_once, rounded_twice);
    }
}

#[test]
fn alibi_bias_is_causal_and_head_specific() -> Result<()> {
    let alibi = PositionEncoding::Alibi(AlibiConfig::new(vec![0.5, 0.25])?);
    let head0_bias = alibi.alibi_bias(0, 10, 6)?;
    let head1_bias = alibi.alibi_bias(1, 10, 6)?;
    assert!((head0_bias + 2.0).abs() <= f32::EPSILON);
    assert!((head1_bias + 1.0).abs() <= f32::EPSILON);
    Ok(())
}

#[test]
fn token_ledger_push_after_explicit_position_stays_monotonic() -> Result<()> {
    let mut ledger = TokenLedger::new();
    ledger.push_with_position(1, 10)?;
    ledger.push(2)?;
    assert_eq!(ledger.positions(), vec![10, 11]);
    Ok(())
}

#[test]
fn alibi_rejects_future_key_position() -> Result<()> {
    let alibi = PositionEncoding::Alibi(AlibiConfig::new(vec![0.5])?);
    let result = alibi.alibi_bias(0, 4, 6);
    assert!(matches!(result, Err(CfrError::InvalidConfig(_))));
    Ok(())
}

#[test]
fn rope_rejects_positions_that_are_not_exact_in_f64() -> Result<()> {
    let rope = RopeConfig::new(10_000.0, 4)?;
    let mut row = vec![1.0, 0.0, 0.5, -0.25];
    let result = rope.apply_key(9_007_199_254_740_992, &mut row);
    assert!(matches!(result, Err(CfrError::Numeric(_))));
    Ok(())
}
