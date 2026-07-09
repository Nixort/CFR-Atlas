// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! `CFR-Atlas` public crate root and production integration surface.
//!
//! This file exposes the safe Rust `API` used by runtimes that want exact
//! `CPU`-first `KV` memory virtualization through deterministic regeneration.

mod atlas;
mod attention;
mod bench;
pub(crate) mod cache;
mod config;
mod conformance;
mod dtype;
mod error;
mod executor;
mod kernel;
pub mod layout;
mod page;
mod pipeline;
mod policy;
mod position;
mod regenerator;
mod schema;
mod stabilization;
mod stats;
mod token;
mod topology;
mod tuning;
mod validation;

pub use atlas::{AttentionRequest, CfrAtlas};
pub use attention::FoldedAttention;
pub use bench::{
    estimate_benchmark_memory, stabilization_benchmark_scenarios, BenchEstimate, BenchScenario,
};
pub use cache::{HotCache, PageView};
pub use config::{Config, ConfigBuilder};
pub use conformance::{assert_regenerated_page, compare_regenerated_page, PageConformance};
pub use dtype::{parse_storage_dtype, AccumulatorDType, DTypePolicy, StorageDType};
pub use error::{CfrError, Result};
pub use executor::{ThreadPoolConfig, ThreadPoolExecutor};
pub use kernel::{dot_auto_vectorized, dot_scalar, DotProductKernel};
pub use layout::{
    checked_add, checked_kv_bytes, checked_matrix_len, checked_mul, checked_range_len,
    checked_row_range, expect_all_finite, expect_len, f64_to_f32_checked, max_abs_diff_finite,
    prepare_zeroed, u32_to_f32_checked, u64_to_f32_checked, u64_to_f64_checked,
    usize_to_f32_checked, usize_to_f64_checked, usize_to_u64_saturating, wipe_f32,
};
pub use page::{PageKey, PageRange};
pub use pipeline::{ColdPageBuffer, DoubleBufferedPipeline};
pub use policy::{
    KeepRecent, NeverAdmit, ResidencyContext, ResidencyDecision, ResidencyPolicy,
    TelemetryResidencyPolicy,
};
pub use position::{AlibiConfig, PositionEncoding, RopeConfig};
pub use regenerator::KvRegenerator;
pub use schema::{VersionedConfig, CONFIG_SCHEMA_VERSION};
pub use stabilization::{
    stabilization_report, ClaimReview, NoStdBlocker, NoStdFeasibilityReport, NoStdReadiness,
    PublicApiReview, StabilizationReport, StabilizationStatus, SupplyChainReview, MSRV,
    PUBLIC_API_REVIEW_VERSION,
};
pub use stats::{CfrCounters, CfrStatsSnapshot};
pub use token::{TokenId, TokenLedger, TokenRecord};
pub use topology::{AttentionTopology, AttentionTopologyKind, HeadMapping};
pub use tuning::{PageSizeTuner, PageTuningCandidate, PageTuningInput, PageTuningResult};
pub use validation::{
    deterministic_query, phase4_regression_corpus, regression_corpus, validate_decode_loop,
    validate_decode_step, DeterministicLogitProjector, LogitProjector, MemoryTelemetry, PromptCase,
    PromptShape, StepValidationRequest, ValidationPlan, ValidationReport, ValidationStepReport,
};

/// Convenient imports for applications that embed `CFR-Atlas`.
pub mod prelude {
    pub use crate::{
        assert_regenerated_page, checked_add, checked_kv_bytes, checked_matrix_len, checked_mul,
        checked_range_len, checked_row_range, compare_regenerated_page, deterministic_query,
        dot_auto_vectorized, dot_scalar, estimate_benchmark_memory, expect_all_finite, expect_len,
        f64_to_f32_checked, max_abs_diff_finite, parse_storage_dtype, phase4_regression_corpus,
        prepare_zeroed, regression_corpus, stabilization_benchmark_scenarios, stabilization_report,
        u32_to_f32_checked, u64_to_f32_checked, u64_to_f64_checked, usize_to_f32_checked,
        usize_to_f64_checked, usize_to_u64_saturating, validate_decode_loop, validate_decode_step,
        wipe_f32, AccumulatorDType, AlibiConfig, AttentionRequest, AttentionTopology,
        AttentionTopologyKind, BenchEstimate, BenchScenario, CfrAtlas, CfrCounters, CfrError,
        CfrStatsSnapshot, ClaimReview, Config, ConfigBuilder, DTypePolicy,
        DeterministicLogitProjector, DotProductKernel, DoubleBufferedPipeline, FoldedAttention,
        HeadMapping, HotCache, KeepRecent, KvRegenerator, LogitProjector, MemoryTelemetry,
        NeverAdmit, NoStdBlocker, NoStdFeasibilityReport, NoStdReadiness, PageConformance, PageKey,
        PageRange, PageSizeTuner, PageTuningCandidate, PageTuningInput, PageTuningResult, PageView,
        PositionEncoding, PromptCase, PromptShape, PublicApiReview, ResidencyContext,
        ResidencyDecision, ResidencyPolicy, Result, RopeConfig, StabilizationReport,
        StabilizationStatus, StepValidationRequest, StorageDType, SupplyChainReview,
        TelemetryResidencyPolicy, ThreadPoolConfig, ThreadPoolExecutor, TokenId, TokenLedger,
        TokenRecord, ValidationPlan, ValidationReport, ValidationStepReport, VersionedConfig,
        CONFIG_SCHEMA_VERSION, MSRV, PUBLIC_API_REVIEW_VERSION,
    };
}
