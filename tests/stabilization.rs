// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

use cfr_atlas::{
    estimate_benchmark_memory, stabilization_benchmark_scenarios, stabilization_report,
    BenchScenario, CfrError, Config, NoStdBlocker, NoStdReadiness, StabilizationStatus,
    VersionedConfig, CONFIG_SCHEMA_VERSION, MSRV,
};

#[test]
fn versioned_config_round_trips() -> cfr_atlas::Result<()> {
    let config = Config::builder(512, 64)
        .hot_cache_bytes(128 << 20)
        .admit_regenerated_pages(true)
        .max_scratch_tokens(512)
        .wipe_scratch_after_use(true)
        .build()?;
    let versioned = VersionedConfig::new(config.clone())?;
    let encoded = versioned.encode();
    let decoded = VersionedConfig::decode(&encoded)?;

    assert_eq!(decoded.schema_version(), CONFIG_SCHEMA_VERSION);
    assert_eq!(decoded.config(), &config);
    assert!(encoded.contains("cfr_atlas_config_schema=1"));
    assert!(encoded.contains("page_tokens=512"));
    Ok(())
}

#[test]
fn versioned_config_rejects_unknown_schema() -> cfr_atlas::Result<()> {
    let config = Config::new(128, 32, 0)?;
    let encoded = VersionedConfig::new(config)?.encode();
    let unsupported = encoded.replace("cfr_atlas_config_schema=1", "cfr_atlas_config_schema=2");
    let error = match VersionedConfig::decode(&unsupported) {
        Ok(_) => return Err(CfrError::InvalidConfig("schema version was accepted")),
        Err(error) => error,
    };
    assert!(matches!(error, CfrError::InvalidConfig(_)));
    Ok(())
}

#[test]
fn versioned_config_rejects_missing_required_field() -> cfr_atlas::Result<()> {
    let config = Config::new(128, 32, 0)?;
    let encoded = VersionedConfig::new(config)?.encode();
    let missing_head_dim = encoded
        .lines()
        .filter(|line| !line.starts_with("head_dim="))
        .collect::<Vec<_>>()
        .join("\n");
    let error = match VersionedConfig::decode(&missing_head_dim) {
        Ok(_) => return Err(CfrError::InvalidConfig("missing field was accepted")),
        Err(error) => error,
    };
    assert!(matches!(error, CfrError::InvalidConfig(_)));
    Ok(())
}

#[test]
fn stabilization_report_matches_release_policy() {
    let report = stabilization_report();
    assert_eq!(report.msrv, MSRV);
    assert_eq!(report.config_schema_version, CONFIG_SCHEMA_VERSION);
    assert!(report.public_api.root_reexports_reviewed);
    assert!(report.public_api.missing_docs_denied);
    assert!(report.public_api.unsafe_code_forbidden);
    assert_eq!(report.no_std.readiness, NoStdReadiness::RequiresStd);
    assert!(report.no_std.alloc_only_core_feasible);
    assert!(report
        .no_std
        .blockers
        .contains(&NoStdBlocker::HotCacheAllocation));
    assert_eq!(report.supply_chain.runtime_dependency_count, 0);
    assert!(report.supply_chain.cargo_deny_policy_shipped);
    assert_eq!(
        report.claims.status,
        StabilizationStatus::RequiresExternalReview
    );
}

#[test]
fn benchmark_memory_estimate_is_deterministic() -> cfr_atlas::Result<()> {
    let scenario = BenchScenario::new(65_536, 64, 512, 0);
    let estimate = estimate_benchmark_memory(scenario)?;
    assert_eq!(estimate.baseline_kv_bytes, 33_554_432);
    assert_eq!(estimate.cfr_scratch_bytes, 262_144);
    assert_eq!(estimate.cfr_resident_budget_bytes, 262_144);
    assert!((estimate.estimated_memory_reduction - 128.0).abs() <= f32::EPSILON);
    Ok(())
}

#[test]
fn stabilization_benchmark_matrix_is_non_empty_and_ordered() {
    let scenarios = stabilization_benchmark_scenarios();
    assert!(!scenarios.is_empty());
    assert_eq!(scenarios[0], BenchScenario::new(4_096, 64, 128, 0));
    assert_eq!(scenarios[1], BenchScenario::new(4_096, 64, 128, 64 << 20));
}

#[test]
fn versioned_config_rejects_duplicate_field() -> cfr_atlas::Result<()> {
    let config = Config::new(128, 32, 0)?;
    let encoded = VersionedConfig::new(config)?.encode();
    let duplicated = format!("{encoded}page_tokens=256\n");
    let error = match VersionedConfig::decode(&duplicated) {
        Ok(_) => return Err(CfrError::InvalidConfig("duplicate field was accepted")),
        Err(error) => error,
    };
    assert!(matches!(error, CfrError::InvalidConfig(_)));
    Ok(())
}

#[test]
fn benchmark_memory_estimate_rejects_invalid_shapes() {
    let zero_context = BenchScenario::new(0, 64, 64, 0);
    let zero_head_dim = BenchScenario::new(128, 0, 64, 0);
    let zero_page = BenchScenario::new(128, 64, 0, 0);
    let oversized_page = BenchScenario::new(128, 64, 256, 0);

    assert!(estimate_benchmark_memory(zero_context).is_err());
    assert!(estimate_benchmark_memory(zero_head_dim).is_err());
    assert!(estimate_benchmark_memory(zero_page).is_err());
    assert!(estimate_benchmark_memory(oversized_page).is_err());
}
