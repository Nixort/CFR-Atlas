// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! `Phase 4` long-context validation helpers.
//!
//! This module is a lightweight validation harness for integrations that need
//! to compare `CFR` folded attention with a full-`KV` baseline at attention and
//! logit level. It deliberately stays backend-agnostic: callers provide a
//! deterministic [`KvRegenerator`] and a [`LogitProjector`] matching their model
//! head or a reference projection used by conformance tests.

use crate::layout::{
    checked_add, checked_kv_bytes, checked_matrix_len, checked_mul, expect_all_finite, expect_len,
    f64_to_f32_checked, max_abs_diff_finite, u32_to_f32_checked, usize_to_f32_checked,
    usize_to_f64_checked, wipe_f32,
};
use crate::{
    AttentionRequest, AttentionTopology, CfrAtlas, CfrError, Config, FoldedAttention, HeadMapping,
    KvRegenerator, PageKey, ResidencyPolicy, Result, TokenId, TokenLedger,
};

/// Regression prompt shape used by the `Phase 4` validation corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptShape {
    /// Short instruction-like prompt.
    Short,
    /// Long patterned context used to stress paging and memory telemetry.
    Long,
    /// Repeated-token prompt that stresses cache reuse and repeated positions.
    Repeated,
    /// Code-like prompt with structured tokens.
    Code,
    /// Dialogue-like prompt with alternating speaker/token regions.
    Dialogue,
}

impl PromptShape {
    /// Stable lowercase label for logs and reports.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Short => "short",
            Self::Long => "long",
            Self::Repeated => "repeated",
            Self::Code => "code",
            Self::Dialogue => "dialogue",
        }
    }
}

/// Tokenized prompt case used by validation loops.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptCase {
    shape: PromptShape,
    token_ids: Vec<TokenId>,
}

impl PromptCase {
    /// Creates a prompt case from already-tokenized ids.
    pub fn new(shape: PromptShape, token_ids: Vec<TokenId>) -> Result<Self> {
        if token_ids.is_empty() {
            return Err(CfrError::InvalidLedger(
                "validation prompt must not be empty",
            ));
        }
        Ok(Self { shape, token_ids })
    }

    /// Returns the prompt shape label.
    #[must_use]
    pub const fn shape(&self) -> PromptShape {
        self.shape
    }

    /// Returns the token ids backing this prompt.
    #[must_use]
    pub fn token_ids(&self) -> &[TokenId] {
        &self.token_ids
    }

    /// Returns the number of tokens in the prompt.
    #[must_use]
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    /// Returns whether the prompt contains no tokens.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }

    /// Converts the prompt into a sequential [`TokenLedger`].
    pub fn to_ledger(&self) -> Result<TokenLedger> {
        TokenLedger::from_token_ids(self.token_ids.iter().copied())
    }
}

/// Returns the built-in long-context regression prompt corpus.
pub fn regression_corpus() -> Result<Vec<PromptCase>> {
    Ok(vec![
        PromptCase::new(PromptShape::Short, short_prompt_tokens())?,
        PromptCase::new(PromptShape::Long, patterned_tokens(2_048, 17, 23)?)?,
        PromptCase::new(PromptShape::Repeated, repeated_prompt_tokens(512, 42)?)?,
        PromptCase::new(PromptShape::Code, code_prompt_tokens()?)?,
        PromptCase::new(PromptShape::Dialogue, dialogue_prompt_tokens()?)?,
    ])
}

/// Returns the built-in long-context regression prompt corpus.
///
/// This compatibility alias preserves older callers while the examples and
/// tests use [`regression_corpus`].
pub fn phase4_regression_corpus() -> Result<Vec<PromptCase>> {
    regression_corpus()
}

/// Projection boundary from attention output to logits.
pub trait LogitProjector {
    /// Number of logits emitted by this projector.
    #[must_use]
    fn vocab_size(&self) -> usize;

    /// Projects one attention output vector into a row of logits.
    fn project_logits(&self, hidden: &[f32], logits_out: &mut [f32]) -> Result<()>;
}

/// Deterministic validation projector used when a real model head is unavailable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeterministicLogitProjector {
    vocab_size: usize,
    scale: f32,
}

impl DeterministicLogitProjector {
    /// Creates a deterministic projector with a finite positive scale.
    pub fn new(vocab_size: usize, scale: f32) -> Result<Self> {
        if vocab_size == 0 {
            return Err(CfrError::InvalidConfig("vocab_size must be non-zero"));
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(CfrError::InvalidConfig(
                "projector scale must be positive and finite",
            ));
        }
        Ok(Self { vocab_size, scale })
    }

    /// Returns the projection scale.
    #[must_use]
    pub const fn scale(&self) -> f32 {
        self.scale
    }
}

impl DeterministicLogitProjector {
    fn project_logits_into_tmp(&self, hidden: &[f32], tmp: &mut [f32]) -> Result<()> {
        for (token_index, logit) in tmp.iter_mut().enumerate() {
            let token_f64 = usize_to_f64_checked("validation token index", token_index)?;
            let mut acc = 0.0f64;
            for (dim, value) in hidden.iter().enumerate() {
                let dim_f64 = usize_to_f64_checked("validation hidden dim", dim)?;
                let phase = token_f64.mul_add(0.000_977, dim_f64 * 0.017_578_125);
                let weight = phase.sin() * f64::from(self.scale);
                acc = f64::from(*value).mul_add(weight, acc);
            }
            *logit = f64_to_f32_checked("validation logit", acc)?;
        }
        Ok(())
    }
}

impl LogitProjector for DeterministicLogitProjector {
    #[must_use]
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn project_logits(&self, hidden: &[f32], logits_out: &mut [f32]) -> Result<()> {
        expect_len("validation logits", self.vocab_size, logits_out.len())?;
        expect_all_finite("validation hidden contains non-finite value", hidden)?;
        let mut tmp = vec![0.0f32; logits_out.len()];
        if let Err(err) = self.project_logits_into_tmp(hidden, &mut tmp) {
            wipe_f32(&mut tmp);
            return Err(err);
        }
        logits_out.copy_from_slice(&tmp);
        wipe_f32(&mut tmp);
        Ok(())
    }
}

/// Long-context memory telemetry for one validation step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryTelemetry {
    /// Resident bytes required by a full-`KV` baseline for the configured span.
    pub baseline_kv_bytes: usize,
    /// Scratch bytes required by one `CFR` page.
    pub cfr_scratch_bytes: usize,
    /// Hot-cache bytes resident after the step.
    pub cfr_hot_cache_bytes: usize,
    /// Total `CFR` resident bytes counted by the validation harness.
    pub cfr_resident_bytes: usize,
    /// Estimated full-`KV` bytes divided by `CFR` resident bytes.
    pub estimated_memory_reduction: f32,
}

impl MemoryTelemetry {
    fn new(
        config: &Config,
        topology: AttentionTopology,
        layers: u32,
        context_tokens: usize,
        hot_cache_bytes: usize,
    ) -> Result<Self> {
        let layer_count = u32_to_usize("validation layers", layers)?;
        let kv_heads = u32_to_usize("validation kv_heads", topology.kv_heads())?;
        let one_head_bytes = checked_kv_bytes(
            "validation baseline one head",
            context_tokens,
            config.head_dim,
        )?;
        let all_layer_bytes =
            checked_mul("validation baseline layers", one_head_bytes, layer_count)?;
        let baseline_kv_bytes =
            checked_mul("validation baseline kv heads", all_layer_bytes, kv_heads)?;
        let cfr_scratch_bytes = checked_kv_bytes(
            "validation CFR scratch bytes",
            config.page_tokens,
            config.head_dim,
        )?;
        let cfr_resident_bytes = checked_add(
            "validation CFR resident bytes",
            cfr_scratch_bytes,
            hot_cache_bytes,
        )?;
        let estimated_memory_reduction = if cfr_resident_bytes == 0 {
            0.0
        } else {
            let baseline = usize_to_f64_checked("validation baseline bytes", baseline_kv_bytes)?;
            let resident = usize_to_f64_checked("validation resident bytes", cfr_resident_bytes)?;
            f64_to_f32_checked("validation memory reduction", baseline / resident)?
        };
        Ok(Self {
            baseline_kv_bytes,
            cfr_scratch_bytes,
            cfr_hot_cache_bytes: hot_cache_bytes,
            cfr_resident_bytes,
            estimated_memory_reduction,
        })
    }
}

/// One decoded validation step after comparing baseline and `CFR` outputs.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationStepReport {
    /// Layer validated by this step.
    pub layer: u32,
    /// Query head requested by the decode loop.
    pub query_head: u32,
    /// Query-to-`KV` mapping used by `MHA`, `MQA` or `GQA`.
    pub mapping: HeadMapping,
    /// Number of visible context tokens.
    pub context_tokens: usize,
    /// Maximum absolute attention-output difference.
    pub max_abs_output: f32,
    /// Maximum absolute logit difference.
    pub max_abs_logits: f32,
    /// Memory telemetry captured after the step.
    pub memory: MemoryTelemetry,
}

/// Full decode-loop validation report.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationReport {
    /// Prompt shape validated by this report.
    pub prompt_shape: PromptShape,
    /// Per-step reports.
    pub steps: Vec<ValidationStepReport>,
}

impl ValidationReport {
    /// Returns the worst attention-output difference in the report.
    #[must_use]
    pub fn worst_output_diff(&self) -> f32 {
        self.steps
            .iter()
            .map(|step| step.max_abs_output)
            .fold(0.0, f32::max)
    }

    /// Returns the worst logit difference in the report.
    #[must_use]
    pub fn worst_logit_diff(&self) -> f32 {
        self.steps
            .iter()
            .map(|step| step.max_abs_logits)
            .fold(0.0, f32::max)
    }

    /// Returns the best observed memory-reduction estimate.
    #[must_use]
    pub fn best_memory_reduction(&self) -> f32 {
        self.steps
            .iter()
            .map(|step| step.memory.estimated_memory_reduction)
            .fold(0.0, f32::max)
    }
}

/// Decode-loop validation plan.
#[derive(Debug, Clone, Copy)]
pub struct ValidationPlan<'a> {
    /// Prompt case to validate.
    pub prompt: &'a PromptCase,
    /// Attention topology used to map query heads to `KV` heads.
    pub topology: AttentionTopology,
    /// Layers sampled by the validation loop.
    pub layers: &'a [u32],
    /// Query heads sampled by the validation loop.
    pub query_heads: &'a [u32],
    /// Decode-step indices sampled from the prompt.
    pub decode_steps: &'a [usize],
    /// Allowed attention-output difference.
    pub output_tolerance: f32,
    /// Allowed logit difference.
    pub logit_tolerance: f32,
}

impl<'a> ValidationPlan<'a> {
    /// Creates a validation plan with zero drift tolerance.
    #[must_use]
    pub const fn new(
        prompt: &'a PromptCase,
        topology: AttentionTopology,
        layers: &'a [u32],
        query_heads: &'a [u32],
        decode_steps: &'a [usize],
    ) -> Self {
        Self {
            prompt,
            topology,
            layers,
            query_heads,
            decode_steps,
            output_tolerance: 0.0,
            logit_tolerance: 0.0,
        }
    }

    /// Sets validation tolerances.
    #[must_use]
    pub const fn with_tolerances(mut self, output_tolerance: f32, logit_tolerance: f32) -> Self {
        self.output_tolerance = output_tolerance;
        self.logit_tolerance = logit_tolerance;
        self
    }

    fn validate(&self) -> Result<()> {
        if self.prompt.is_empty() {
            return Err(CfrError::InvalidLedger(
                "validation prompt must not be empty",
            ));
        }
        if self.layers.is_empty() {
            return Err(CfrError::InvalidConfig(
                "validation layers must not be empty",
            ));
        }
        if self.query_heads.is_empty() {
            return Err(CfrError::InvalidConfig(
                "validation query_heads must not be empty",
            ));
        }
        if self.decode_steps.is_empty() {
            return Err(CfrError::InvalidConfig(
                "validation decode_steps must not be empty",
            ));
        }
        if !self.output_tolerance.is_finite() || self.output_tolerance < 0.0 {
            return Err(CfrError::InvalidConfig(
                "output_tolerance must be finite and non-negative",
            ));
        }
        if !self.logit_tolerance.is_finite() || self.logit_tolerance < 0.0 {
            return Err(CfrError::InvalidConfig(
                "logit_tolerance must be finite and non-negative",
            ));
        }
        for step in self.decode_steps {
            if *step >= self.prompt.len() {
                return Err(CfrError::InvalidLedger("decode step exceeds prompt length"));
            }
        }
        Ok(())
    }
}

/// Runs one full-`KV` vs `CFR` comparison step and validates drift bounds.
pub fn validate_decode_step<R, P, L>(
    atlas: &mut CfrAtlas,
    regenerator: &R,
    policy: &P,
    projector: &L,
    request: StepValidationRequest<'_>,
) -> Result<ValidationStepReport>
where
    R: KvRegenerator,
    P: ResidencyPolicy,
    L: LogitProjector,
{
    request.validate(atlas.config(), projector)?;
    let mapping = request.topology.map_query_head(request.query_head)?;
    let mut full_output = baseline_attention_output(
        atlas.config(),
        regenerator,
        request.layer,
        mapping.kv_head,
        request.query,
        request.context_tokens,
    )?;

    let mut atlas_output = vec![0.0f32; atlas.config().head_dim];
    if let Err(err) = atlas.attend_exact_with_policy(
        regenerator,
        policy,
        AttentionRequest::new(
            request.layer,
            mapping.kv_head,
            request.query,
            request.context_tokens,
        ),
        &mut atlas_output,
    ) {
        wipe_f32(&mut full_output);
        wipe_f32(&mut atlas_output);
        return Err(err);
    }

    let output_diff =
        match max_abs_diff_finite("validation attention output", &full_output, &atlas_output) {
            Ok(diff) => diff,
            Err(err) => {
                wipe_f32(&mut full_output);
                wipe_f32(&mut atlas_output);
                return Err(err);
            }
        };
    if output_diff > request.output_tolerance {
        wipe_f32(&mut full_output);
        wipe_f32(&mut atlas_output);
        return Err(CfrError::Numeric(
            "attention output drift exceeds tolerance",
        ));
    }

    let mut full_logits = vec![0.0f32; projector.vocab_size()];
    let mut atlas_logits = vec![0.0f32; projector.vocab_size()];
    if let Err(err) = projector.project_logits(&full_output, &mut full_logits) {
        wipe_f32(&mut full_output);
        wipe_f32(&mut atlas_output);
        wipe_f32(&mut full_logits);
        wipe_f32(&mut atlas_logits);
        return Err(err);
    }
    if let Err(err) = projector.project_logits(&atlas_output, &mut atlas_logits) {
        wipe_f32(&mut full_output);
        wipe_f32(&mut atlas_output);
        wipe_f32(&mut full_logits);
        wipe_f32(&mut atlas_logits);
        return Err(err);
    }

    let logit_diff = match max_abs_diff_finite("validation logits", &full_logits, &atlas_logits) {
        Ok(diff) => diff,
        Err(err) => {
            wipe_f32(&mut full_output);
            wipe_f32(&mut atlas_output);
            wipe_f32(&mut full_logits);
            wipe_f32(&mut atlas_logits);
            return Err(err);
        }
    };
    wipe_f32(&mut full_output);
    wipe_f32(&mut atlas_output);
    wipe_f32(&mut full_logits);
    wipe_f32(&mut atlas_logits);

    if logit_diff > request.logit_tolerance {
        return Err(CfrError::Numeric("logit drift exceeds tolerance"));
    }

    let memory = MemoryTelemetry::new(
        atlas.config(),
        request.topology,
        request.layers,
        request.context_tokens,
        atlas.stats().hot_cache_bytes,
    )?;

    Ok(ValidationStepReport {
        layer: request.layer,
        query_head: request.query_head,
        mapping,
        context_tokens: request.context_tokens,
        max_abs_output: output_diff,
        max_abs_logits: logit_diff,
        memory,
    })
}

/// Input object for a single validation step.
#[derive(Debug, Clone, Copy)]
pub struct StepValidationRequest<'a> {
    /// Number of model layers represented in memory telemetry.
    pub layers: u32,
    /// Attention topology used by the step.
    pub topology: AttentionTopology,
    /// Layer validated by the step.
    pub layer: u32,
    /// Query head validated by the step.
    pub query_head: u32,
    /// Query vector used by baseline and `CFR` paths.
    pub query: &'a [f32],
    /// Number of visible causal context tokens.
    pub context_tokens: usize,
    /// Maximum accepted attention-output difference.
    pub output_tolerance: f32,
    /// Maximum accepted logit difference.
    pub logit_tolerance: f32,
}

impl<'a> StepValidationRequest<'a> {
    /// Creates a single-step validation request.
    #[must_use]
    pub const fn new(
        layers: u32,
        topology: AttentionTopology,
        layer: u32,
        query_head: u32,
        query: &'a [f32],
        context_tokens: usize,
    ) -> Self {
        Self {
            layers,
            topology,
            layer,
            query_head,
            query,
            context_tokens,
            output_tolerance: 0.0,
            logit_tolerance: 0.0,
        }
    }

    /// Sets accepted drift tolerances.
    #[must_use]
    pub const fn with_tolerances(mut self, output_tolerance: f32, logit_tolerance: f32) -> Self {
        self.output_tolerance = output_tolerance;
        self.logit_tolerance = logit_tolerance;
        self
    }

    fn validate<L: LogitProjector>(&self, config: &Config, projector: &L) -> Result<()> {
        if self.layers == 0 {
            return Err(CfrError::InvalidConfig(
                "validation layers must be non-zero",
            ));
        }
        if self.layer >= self.layers {
            return Err(CfrError::InvalidConfig("validation layer is out of range"));
        }
        if self.context_tokens == 0 {
            return Err(CfrError::InvalidConfig(
                "validation context_tokens must be non-zero",
            ));
        }
        if projector.vocab_size() == 0 {
            return Err(CfrError::InvalidConfig(
                "validation projector vocab_size must be non-zero",
            ));
        }
        if !self.output_tolerance.is_finite() || self.output_tolerance < 0.0 {
            return Err(CfrError::InvalidConfig(
                "output_tolerance must be finite and non-negative",
            ));
        }
        if !self.logit_tolerance.is_finite() || self.logit_tolerance < 0.0 {
            return Err(CfrError::InvalidConfig(
                "logit_tolerance must be finite and non-negative",
            ));
        }
        expect_len("validation query", config.head_dim, self.query.len())?;
        Ok(())
    }
}

/// Runs a sampled decode-loop validation over layers, heads and prompt positions.
pub fn validate_decode_loop<R, P, L>(
    atlas: &mut CfrAtlas,
    regenerator: &R,
    policy: &P,
    projector: &L,
    plan: &ValidationPlan<'_>,
) -> Result<ValidationReport>
where
    R: KvRegenerator,
    P: ResidencyPolicy,
    L: LogitProjector,
{
    plan.validate()?;
    let layer_count = max_layer_count(plan.layers)?;
    let mut steps = Vec::new();

    for decode_step in plan.decode_steps {
        let context_tokens = checked_add("validation context tokens", *decode_step, 1)?;
        for layer in plan.layers {
            for query_head in plan.query_heads {
                let mut query = deterministic_query(
                    plan.prompt,
                    *decode_step,
                    *layer,
                    *query_head,
                    atlas.config().head_dim,
                )?;
                let request = StepValidationRequest::new(
                    layer_count,
                    plan.topology,
                    *layer,
                    *query_head,
                    &query,
                    context_tokens,
                )
                .with_tolerances(plan.output_tolerance, plan.logit_tolerance);
                let step = validate_decode_step(atlas, regenerator, policy, projector, request);
                wipe_f32(&mut query);
                steps.push(step?);
            }
        }
    }

    Ok(ValidationReport {
        prompt_shape: plan.prompt.shape(),
        steps,
    })
}

/// Builds a deterministic query vector for validation decode steps.
pub fn deterministic_query(
    prompt: &PromptCase,
    decode_step: usize,
    layer: u32,
    query_head: u32,
    head_dim: usize,
) -> Result<Vec<f32>> {
    if decode_step >= prompt.len() {
        return Err(CfrError::InvalidLedger("decode step exceeds prompt length"));
    }
    if head_dim == 0 {
        return Err(CfrError::InvalidConfig("head_dim must be non-zero"));
    }
    let token = u32_to_f32_checked("validation query token", prompt.token_ids()[decode_step])?;
    let step = usize_to_f32_checked("validation query step", decode_step)?;
    let layer = u32_to_f32_checked("validation query layer", layer)?;
    let query_head = u32_to_f32_checked("validation query head", query_head)?;
    let mut query = Vec::with_capacity(head_dim);
    for dim in 0..head_dim {
        let dim = match usize_to_f32_checked("validation query dim", dim) {
            Ok(dim) => dim,
            Err(err) => {
                wipe_f32(&mut query);
                return Err(err);
            }
        };
        let phase = token.mul_add(
            0.004_882_812_5,
            step.mul_add(
                0.000_976_562_5,
                layer.mul_add(0.031_25, query_head.mul_add(0.015_625, dim * 0.007_812_5)),
            ),
        );
        query.push(phase.sin().mul_add(0.75, phase.cos() * 0.25));
    }
    Ok(query)
}

fn baseline_attention_output<R: KvRegenerator>(
    config: &Config,
    regenerator: &R,
    layer: u32,
    kv_head: u32,
    query: &[f32],
    context_tokens: usize,
) -> Result<Vec<f32>> {
    let key = PageKey::new(layer, kv_head, 0);
    let len = checked_matrix_len("validation baseline page", context_tokens, config.head_dim)?;
    let mut k = vec![0.0f32; len];
    let mut v = vec![0.0f32; len];
    if let Err(err) =
        regenerator.regenerate_page(key, 0..context_tokens, config.head_dim, &mut k, &mut v)
    {
        wipe_f32(&mut k);
        wipe_f32(&mut v);
        return Err(err);
    }

    let mut folded = match FoldedAttention::new(config.head_dim, config.scale) {
        Ok(folded) => folded,
        Err(err) => {
            wipe_f32(&mut k);
            wipe_f32(&mut v);
            return Err(err);
        }
    };
    if let Err(err) = folded.consume_page(query, &k, &v, context_tokens) {
        wipe_f32(&mut k);
        wipe_f32(&mut v);
        return Err(err);
    }
    let mut out = vec![0.0f32; config.head_dim];
    let result = folded.finish_into(&mut out);
    wipe_f32(&mut k);
    wipe_f32(&mut v);
    match result {
        Ok(()) => Ok(out),
        Err(err) => {
            wipe_f32(&mut out);
            Err(err)
        }
    }
}

fn max_layer_count(layers: &[u32]) -> Result<u32> {
    let Some(max_layer) = layers.iter().copied().max() else {
        return Err(CfrError::InvalidConfig(
            "validation layers must not be empty",
        ));
    };
    max_layer.checked_add(1).ok_or(CfrError::CapacityOverflow {
        name: "validation layer count",
    })
}

fn short_prompt_tokens() -> Vec<TokenId> {
    vec![101, 734, 2_048, 19, 87, 4_096, 102]
}

fn repeated_prompt_tokens(len: usize, token_id: TokenId) -> Result<Vec<TokenId>> {
    if len == 0 {
        return Err(CfrError::InvalidLedger(
            "repeated prompt length must be non-zero",
        ));
    }
    Ok(vec![token_id; len])
}

fn code_prompt_tokens() -> Result<Vec<TokenId>> {
    patterned_tokens(384, 31, 7)
}

fn dialogue_prompt_tokens() -> Result<Vec<TokenId>> {
    let mut tokens = Vec::with_capacity(448);
    for turn in 0..56usize {
        let speaker = if turn % 2 == 0 { 11 } else { 13 };
        tokens.push(speaker);
        let body = patterned_tokens(7, turn.saturating_add(3), 29)?;
        tokens.extend(body);
    }
    Ok(tokens)
}

fn patterned_tokens(len: usize, stride: usize, offset: usize) -> Result<Vec<TokenId>> {
    if len == 0 {
        return Err(CfrError::InvalidLedger(
            "patterned prompt length must be non-zero",
        ));
    }
    let mut tokens = Vec::with_capacity(len);
    for index in 0..len {
        let stepped = checked_mul("patterned token stride", index, stride)?;
        let mixed = checked_add("patterned token offset", stepped, offset)?;
        let folded = mixed % 32_000;
        let token = u32::try_from(folded)
            .map_err(|_| CfrError::Numeric("patterned token id does not fit u32"))?;
        tokens.push(token.saturating_add(1));
    }
    Ok(tokens)
}

fn u32_to_usize(name: &'static str, value: u32) -> Result<usize> {
    usize::try_from(value).map_err(|_| CfrError::Numeric(name))
}
