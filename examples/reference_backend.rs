// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! Reference backend example.
//!
//! This example exercises the adapter boundary: token ledger, `GQA` mapping, `RoPE`,
//! dtype policy, conformance and `CFR` folded attention.

use cfr_atlas::prelude::*;
use cfr_atlas_backend_ref::ReferenceBackend;

fn main() -> Result<()> {
    let ledger = TokenLedger::from_token_ids((0..256).map(|i| 32_000 + i))?;
    let topology = AttentionTopology::gqa(16, 4)?;
    let rope = PositionEncoding::Rope(RopeConfig::new(10_000.0, 32)?);
    let dtype = DTypePolicy::bf16();
    let backend = ReferenceBackend::new(ledger, topology, rope, dtype, 8, 32)?;

    let mapping = backend.map_query_head(11)?;
    println!("query_head=11 maps_to_kv_head={}", mapping.kv_head);

    let report = backend.conformance_report(PageKey::new(3, mapping.kv_head, 64), 64..128, 0.0)?;
    println!("conformance={report:?}");

    let hot_cache_bytes = checked_kv_bytes(
        "reference backend example hot-cache bytes",
        64,
        backend.head_dim(),
    )?;

    let config = Config::builder(64, backend.head_dim())
        .hot_cache_bytes(hot_cache_bytes)
        .admit_regenerated_pages(true)
        .build()?;
    let mut atlas = CfrAtlas::new(config)?;
    let query = vec![0.0625; backend.head_dim()];
    let mut output = vec![0.0; backend.head_dim()];

    atlas.attend_exact_with_policy(
        &backend,
        &KeepRecent { recent_tokens: 64 },
        AttentionRequest::new(3, mapping.kv_head, &query, 192),
        &mut output,
    )?;

    println!("first_output_values={:?}", &output[..8]);
    println!("stats={:?}", atlas.stats());
    Ok(())
}
