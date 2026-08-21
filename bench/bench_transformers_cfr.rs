//! Real-model CFR-Atlas attention benchmark over exported Transformers K/V pages.
//!
//! `export_transformers_qwen_kv.py` creates the ignored input artifacts from a
//! real Qwen2.5 forward pass and independently replayed causal prefixes. This
//! binary intentionally has no Python, `PyTorch`, model-weight, or network
//! dependency: it verifies that the real-model K/V rows can be consumed by the
//! public CFR-Atlas contract and emits raw CSV for report generation.

use cfr_atlas::{
    AttentionRequest, CfrAtlas, CfrError, Config, KeepRecent, KvRegenerator, PageKey, Result,
};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const DEFAULT_ARTIFACT_DIR: &str = "bench/data/transformers_qwen2_5_0_5b";
const MEASURED_RUNS: usize = 5;

#[derive(Debug, Clone, Copy)]
struct Metadata {
    context_tokens: usize,
    page_tokens: usize,
    layer: u32,
    kv_head: u32,
    head_dim: usize,
    scale: f32,
}

#[derive(Debug)]
struct ExactPages {
    metadata: Metadata,
    k: Vec<f32>,
    v: Vec<f32>,
}

impl ExactPages {
    fn load(
        directory: &Path,
        k_name: &str,
        v_name: &str,
    ) -> std::result::Result<Self, Box<dyn Error>> {
        let metadata = Metadata::load(directory.join("manifest.json").as_path())?;
        let expected = metadata
            .context_tokens
            .checked_mul(metadata.head_dim)
            .ok_or("artifact K/V shape overflow")?;
        let k = read_f32(directory.join(k_name).as_path())?;
        let v = read_f32(directory.join(v_name).as_path())?;
        if k.len() != expected || v.len() != expected {
            return Err(format!(
                "unexpected K/V artifact length: expected {expected}, got K={} V={}",
                k.len(),
                v.len()
            )
            .into());
        }
        Ok(Self { metadata, k, v })
    }
}

impl KvRegenerator for ExactPages {
    fn regenerate_page(
        &self,
        key: PageKey,
        token_range: std::ops::Range<usize>,
        head_dim: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<()> {
        if key.layer != self.metadata.layer || key.head != self.metadata.kv_head {
            return Err(CfrError::Regenerator(
                "page identity does not match exported model head".to_owned(),
            ));
        }
        if head_dim != self.metadata.head_dim {
            return Err(CfrError::Regenerator(
                "page head dimension does not match exported model".to_owned(),
            ));
        }
        if token_range.start >= token_range.end || token_range.end > self.metadata.context_tokens {
            return Err(CfrError::Regenerator(
                "page token range is outside exported context".to_owned(),
            ));
        }
        let start = token_range
            .start
            .checked_mul(head_dim)
            .ok_or(CfrError::CapacityOverflow {
                name: "page start offset",
            })?;
        let end = token_range
            .end
            .checked_mul(head_dim)
            .ok_or(CfrError::CapacityOverflow {
                name: "page end offset",
            })?;
        let expected = end - start;
        if k_out.len() != expected || v_out.len() != expected {
            return Err(CfrError::Dimension {
                name: "real-model page output",
                expected,
                got: k_out.len().min(v_out.len()),
            });
        }
        k_out.copy_from_slice(&self.k[start..end]);
        v_out.copy_from_slice(&self.v[start..end]);
        Ok(())
    }
}

impl Metadata {
    fn load(path: &Path) -> std::result::Result<Self, Box<dyn Error>> {
        let raw = fs::read_to_string(path)?;
        Ok(Self {
            context_tokens: json_usize(&raw, "context_tokens")?,
            page_tokens: json_usize(&raw, "page_tokens")?,
            layer: u32::try_from(json_usize(&raw, "layer")?)?,
            kv_head: u32::try_from(json_usize(&raw, "kv_head")?)?,
            head_dim: json_usize(&raw, "head_dim")?,
            scale: json_f32(&raw, "attention_scale")?,
        })
    }
}

fn json_value<'a>(raw: &'a str, key: &str) -> std::result::Result<&'a str, Box<dyn Error>> {
    let marker = format!("\"{key}\":");
    let value = raw
        .split_once(marker.as_str())
        .ok_or_else(|| format!("manifest key is missing: {key}"))?
        .1
        .trim_start();
    Ok(value
        .split(|character: char| character == ',' || character == '\n' || character == '}')
        .next()
        .ok_or_else(|| format!("manifest value is missing: {key}"))?
        .trim())
}

fn json_usize(raw: &str, key: &str) -> std::result::Result<usize, Box<dyn Error>> {
    Ok(json_value(raw, key)?.parse()?)
}

fn json_f32(raw: &str, key: &str) -> std::result::Result<f32, Box<dyn Error>> {
    Ok(json_value(raw, key)?.parse()?)
}

fn read_f32(path: &Path) -> std::result::Result<Vec<f32>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "f32 artifact has non-multiple-of-four length: {}",
            path.display()
        )
        .into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn direct_attention(query: &[f32], k: &[f32], v: &[f32], metadata: Metadata) -> Vec<f32> {
    let mut scores = Vec::with_capacity(metadata.context_tokens);
    for token in 0..metadata.context_tokens {
        let offset = token * metadata.head_dim;
        let score = query
            .iter()
            .zip(&k[offset..offset + metadata.head_dim])
            .map(|(query_value, key_value)| query_value * key_value)
            .sum::<f32>()
            * metadata.scale;
        scores.push(score);
    }
    let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let normalizer = scores
        .iter()
        .map(|score| (score - maximum).exp())
        .sum::<f32>();
    let mut output = vec![0.0; metadata.head_dim];
    for (token, score) in scores.into_iter().enumerate() {
        let weight = (score - maximum).exp() / normalizer;
        let offset = token * metadata.head_dim;
        for (output_value, value) in output
            .iter_mut()
            .zip(&v[offset..offset + metadata.head_dim])
        {
            *output_value += weight * value;
        }
    }
    output
}

fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left_value, right_value)| (left_value - right_value).abs())
        .fold(0.0, f32::max)
}

fn make_atlas(metadata: Metadata, hot_cache_bytes: usize, admit: bool) -> Result<CfrAtlas> {
    CfrAtlas::new(
        Config::builder(metadata.page_tokens, metadata.head_dim)
            .hot_cache_bytes(hot_cache_bytes)
            .scale(metadata.scale)
            .admit_regenerated_pages(admit)
            .max_scratch_tokens(metadata.page_tokens)
            .wipe_scratch_after_use(false)
            .build()?,
    )
}

fn run_cold(
    regenerator: &ExactPages,
    query: &[f32],
) -> Result<(Vec<f32>, cfr_atlas::CfrStatsSnapshot, u128)> {
    let metadata = regenerator.metadata;
    let mut atlas = make_atlas(metadata, 0, false)?;
    let mut output = vec![0.0; metadata.head_dim];
    let started = Instant::now();
    atlas.attend_exact(
        regenerator,
        AttentionRequest::new(
            metadata.layer,
            metadata.kv_head,
            query,
            metadata.context_tokens,
        ),
        &mut output,
    )?;
    Ok((output, atlas.stats(), started.elapsed().as_micros()))
}

fn run_hot(
    regenerator: &ExactPages,
    query: &[f32],
) -> Result<(Vec<f32>, cfr_atlas::CfrStatsSnapshot, u128)> {
    let metadata = regenerator.metadata;
    let page_bytes = metadata
        .page_tokens
        .checked_mul(metadata.head_dim)
        .and_then(|values| values.checked_mul(8))
        .ok_or(CfrError::CapacityOverflow {
            name: "hot page bytes",
        })?;
    let mut atlas = make_atlas(metadata, page_bytes, true)?;
    let policy = KeepRecent {
        recent_tokens: metadata.page_tokens.saturating_sub(1),
    };
    let mut output = vec![0.0; metadata.head_dim];
    atlas.attend_exact_with_policy(
        regenerator,
        &policy,
        AttentionRequest::new(
            metadata.layer,
            metadata.kv_head,
            query,
            metadata.context_tokens,
        ),
        &mut output,
    )?;
    atlas.reset_counters();
    let started = Instant::now();
    atlas.attend_exact_with_policy(
        regenerator,
        &policy,
        AttentionRequest::new(
            metadata.layer,
            metadata.kv_head,
            query,
            metadata.context_tokens,
        ),
        &mut output,
    )?;
    Ok((output, atlas.stats(), started.elapsed().as_micros()))
}

fn main() -> std::result::Result<(), Box<dyn Error>> {
    let artifact_dir = std::env::args()
        .nth(1)
        .map_or_else(|| PathBuf::from(DEFAULT_ARTIFACT_DIR), PathBuf::from);
    let baseline = ExactPages::load(
        &artifact_dir,
        "baseline_k_f32le.bin",
        "baseline_v_f32le.bin",
    )?;
    let replayed = ExactPages::load(
        &artifact_dir,
        "replayed_k_f32le.bin",
        "replayed_v_f32le.bin",
    )?;
    let metadata = baseline.metadata;
    if replayed.metadata.context_tokens != metadata.context_tokens
        || replayed.metadata.page_tokens != metadata.page_tokens
        || replayed.metadata.head_dim != metadata.head_dim
    {
        return Err("baseline and replayed artifacts have incompatible metadata".into());
    }
    let query = read_f32(&artifact_dir.join("query_f32le.bin"))?;
    let python_baseline = read_f32(&artifact_dir.join("baseline_attention_f32le.bin"))?;
    if query.len() != metadata.head_dim || python_baseline.len() != metadata.head_dim {
        return Err("query or Python attention artifact has wrong head dimension".into());
    }
    let direct = direct_attention(&query, &baseline.k, &baseline.v, metadata);
    let direct_python_diff = max_abs_diff(&direct, &python_baseline);
    println!("scenario,run,elapsed_us,output_max_abs_diff,direct_python_max_abs_diff,hot_hits,cold_regenerations,hot_cache_bytes,hot_cache_pages");
    println!(
        "direct_full_kv,0,0,{direct_python_diff:.9},{direct_python_diff:.9},0,0,{},0",
        metadata.context_tokens * metadata.head_dim * 8
    );
    for run in 1..=MEASURED_RUNS {
        let (output, stats, elapsed) = run_cold(&replayed, &query)?;
        println!(
            "cfr_cold,{run},{elapsed},{:.9},{direct_python_diff:.9},{},{},{},{}",
            max_abs_diff(&output, &direct),
            stats.hot_hits,
            stats.cold_regenerations,
            stats.hot_cache_bytes,
            stats.hot_cache_pages
        );
    }
    for run in 1..=MEASURED_RUNS {
        let (output, stats, elapsed) = run_hot(&replayed, &query)?;
        println!(
            "cfr_hot_recent_page,{run},{elapsed},{:.9},{direct_python_diff:.9},{},{},{},{}",
            max_abs_diff(&output, &direct),
            stats.hot_hits,
            stats.cold_regenerations,
            stats.hot_cache_bytes,
            stats.hot_cache_pages
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_pages_overwrite_requested_rows() {
        let metadata = Metadata {
            context_tokens: 2,
            page_tokens: 1,
            layer: 0,
            kv_head: 0,
            head_dim: 2,
            scale: 0.5,
        };
        let pages = ExactPages {
            metadata,
            k: vec![1.0, 2.0, 3.0, 4.0],
            v: vec![5.0, 6.0, 7.0, 8.0],
        };
        let mut k_out = vec![f32::NAN; 2];
        let mut v_out = vec![f32::NAN; 2];
        pages
            .regenerate_page(PageKey::new(0, 0, 1), 1..2, 2, &mut k_out, &mut v_out)
            .expect("synthetic in-memory page must be valid");
        assert_eq!(k_out, vec![3.0, 4.0]);
        assert_eq!(v_out, vec![7.0, 8.0]);
    }

    #[test]
    fn manifest_parser_reads_numeric_values() {
        let manifest = r#"{
            "context_tokens": 512,
            "page_tokens": 64,
            "layer": 0,
            "kv_head": 1,
            "head_dim": 128,
            "attention_scale": 0.08838835
        }"#;
        let metadata = Metadata {
            context_tokens: json_usize(manifest, "context_tokens").expect("context token value"),
            page_tokens: json_usize(manifest, "page_tokens").expect("page token value"),
            layer: u32::try_from(json_usize(manifest, "layer").expect("layer value"))
                .expect("u32 layer"),
            kv_head: u32::try_from(json_usize(manifest, "kv_head").expect("head value"))
                .expect("u32 head"),
            head_dim: json_usize(manifest, "head_dim").expect("head dimension value"),
            scale: json_f32(manifest, "attention_scale").expect("attention scale value"),
        };
        assert_eq!(metadata.context_tokens, 512);
        assert_eq!(metadata.page_tokens, 64);
        assert_eq!(metadata.kv_head, 1);
        assert_eq!(metadata.head_dim, 128);
        assert!((metadata.scale - 0.088_388_35).abs() < f32::EPSILON);
    }
}
