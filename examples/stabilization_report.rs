// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! Stabilization report example.
//!
//! This example prints the versioned schema, public API review, `no_std` readiness,
//! supply-chain status and release benchmark estimate used by the stabilization docs.

use cfr_atlas::prelude::*;

fn main() -> Result<()> {
    let config = Config::builder(512, 64)
        .hot_cache_bytes(256 << 20)
        .admit_regenerated_pages(true)
        .build()?;
    let versioned = VersionedConfig::new(config)?;
    let encoded = versioned.encode();
    let decoded = VersionedConfig::decode(&encoded)?;
    let report = stabilization_report();
    let estimate = estimate_benchmark_memory(BenchScenario::new(65_536, 64, 512, 0))?;

    println!("schema_version={}", decoded.schema_version());
    println!("msrv={}", report.msrv);
    println!(
        "runtime_dependencies={}",
        report.supply_chain.runtime_dependency_count
    );
    println!("no_std_readiness={:?}", report.no_std.readiness);
    println!(
        "alloc_only_core_feasible={}",
        report.no_std.alloc_only_core_feasible
    );
    println!(
        "public_api_reviewed={}",
        report.public_api.root_reexports_reviewed
    );
    println!("baseline_kv_bytes={}", estimate.baseline_kv_bytes);
    println!(
        "cfr_resident_budget_bytes={}",
        estimate.cfr_resident_budget_bytes
    );
    println!(
        "estimated_memory_reduction={:.2}x",
        estimate.estimated_memory_reduction
    );

    Ok(())
}
