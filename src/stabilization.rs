// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! `Phase 5` stabilization metadata and release-readiness checks.
//!
//! These values are deliberately small and deterministic. They let embedders and
//! release automation inspect the crate's public promises without parsing `README`
//! text or `CI` configuration.

use crate::schema::CONFIG_SCHEMA_VERSION;

/// Current public API review version.
pub const PUBLIC_API_REVIEW_VERSION: u32 = 1;

/// Minimum supported `Rust` version for the stable workspace.
pub const MSRV: &str = "1.75";

/// Current stabilization status for one release gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StabilizationStatus {
    /// The gate is complete for the current skeleton release.
    Complete,
    /// The gate is intentionally documented but still requires downstream proof.
    RequiresExternalReview,
}

/// Summary of the public `API` review.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublicApiReview {
    /// Review document version.
    pub review_version: u32,
    /// Whether the core public API is intentionally re-exported from `lib.rs`.
    pub root_reexports_reviewed: bool,
    /// Whether public documentation is denied by lint rather than manually checked.
    pub missing_docs_denied: bool,
    /// Whether `unsafe` code is forbidden by lint in the core crate.
    pub unsafe_code_forbidden: bool,
}

/// `no_std` readiness level for the current crate architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoStdReadiness {
    /// The current workspace requires `std`.
    RequiresStd,
    /// A future split could keep a smaller core at `alloc` level.
    AllocOnlyCandidate,
    /// The workspace is already `no_std` compatible.
    Ready,
}

/// Blocking area for a future `no_std` or `alloc`-only split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoStdBlocker {
    /// The hot cache currently uses allocation-backed maps.
    HotCacheAllocation,
    /// Worker execution currently uses standard-library threads.
    ThreadExecutor,
    /// Validation and examples intentionally use allocation and `std` utilities.
    ValidationHarness,
}

const NO_STD_BLOCKERS: &[NoStdBlocker] = &[
    NoStdBlocker::HotCacheAllocation,
    NoStdBlocker::ThreadExecutor,
    NoStdBlocker::ValidationHarness,
];

/// `no_std` feasibility result for the current crate architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoStdFeasibilityReport {
    /// Current readiness level.
    pub readiness: NoStdReadiness,
    /// Whether the core could plausibly become `alloc`-only after isolating `std` modules.
    pub alloc_only_core_feasible: bool,
    /// Reviewable list of current blockers.
    pub blockers: &'static [NoStdBlocker],
}

/// Supply-chain posture for the dependency-light release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SupplyChainReview {
    /// Runtime dependency count for the main crate.
    pub runtime_dependency_count: usize,
    /// Whether a `cargo-deny` policy and helper script are shipped.
    pub cargo_deny_policy_shipped: bool,
    /// Whether release scripts can generate checksum manifests.
    pub checksum_manifest_script_shipped: bool,
    /// Whether release scripts support detached signing with the caller's `GPG` key.
    pub signing_script_shipped: bool,
}

/// External claim-review status for exactness and memory accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClaimReview {
    /// Whether exactness assumptions are documented in a reviewable claims file.
    pub exactness_claim_documented: bool,
    /// Whether memory-accounting assumptions are documented in a reviewable claims file.
    pub memory_claim_documented: bool,
    /// Current external review status.
    pub status: StabilizationStatus,
}

/// Full `Phase 5` stabilization report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StabilizationReport {
    /// Minimum supported `Rust` version.
    pub msrv: &'static str,
    /// Versioned config schema currently emitted by the crate.
    pub config_schema_version: u32,
    /// Public `API` review gate.
    pub public_api: PublicApiReview,
    /// `no_std` feasibility gate.
    pub no_std: NoStdFeasibilityReport,
    /// Supply-chain and release-artifact gate.
    pub supply_chain: SupplyChainReview,
    /// External-review gate for exactness and memory claims.
    pub claims: ClaimReview,
}

/// Returns the current `Phase 5` stabilization report.
#[must_use]
pub const fn stabilization_report() -> StabilizationReport {
    StabilizationReport {
        msrv: MSRV,
        config_schema_version: CONFIG_SCHEMA_VERSION,
        public_api: PublicApiReview {
            review_version: PUBLIC_API_REVIEW_VERSION,
            root_reexports_reviewed: true,
            missing_docs_denied: true,
            unsafe_code_forbidden: true,
        },
        no_std: NoStdFeasibilityReport {
            readiness: NoStdReadiness::RequiresStd,
            alloc_only_core_feasible: true,
            blockers: NO_STD_BLOCKERS,
        },
        supply_chain: SupplyChainReview {
            runtime_dependency_count: 0,
            cargo_deny_policy_shipped: true,
            checksum_manifest_script_shipped: true,
            signing_script_shipped: true,
        },
        claims: ClaimReview {
            exactness_claim_documented: true,
            memory_claim_documented: true,
            status: StabilizationStatus::RequiresExternalReview,
        },
    }
}
