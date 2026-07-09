// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! Phase 1 property-style tests for page boundaries and cache invariants.
//!
//! The project intentionally keeps the core dependency-light, so these tests use
//! deterministic exhaustive grids and a tiny pseudo-random generator instead of
//! an external property-test crate.

use cfr_atlas::prelude::*;

fn fill_page(key: PageKey, tokens: usize, head_dim: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    let len = checked_matrix_len("property invariant page length", tokens, head_dim)?;
    let mut k = vec![0.0; len];
    let mut v = vec![0.0; len];
    for row in 0..tokens {
        let row_range = checked_row_range("property invariant row", row, head_dim, len)?;
        for (dim, offset) in row_range.enumerate() {
            let layer = u32_to_f32_checked("property layer", key.layer)?;
            let head = u32_to_f32_checked("property head", key.head)?;
            let token = usize_to_f32_checked("property token", key.start_token + row)?;
            let dim = usize_to_f32_checked("property dim", dim)?;
            k[offset] = layer.mul_add(0.25, head.mul_add(0.125, token.mul_add(0.01, dim)));
            v[offset] = layer.mul_add(0.5, head.mul_add(0.25, token.mul_add(0.02, dim)));
        }
    }
    Ok((k, v))
}

#[test]
fn page_range_boundary_properties() {
    for start in 0..24usize {
        let key = PageKey::new(1, 2, start);
        for range_start in 0..24usize {
            for range_end in 0..24usize {
                let result = PageRange::new(key, range_start..range_end);
                let should_accept = range_start == start && range_end > range_start;
                if should_accept {
                    assert!(result.is_ok());
                } else {
                    assert!(result.is_err());
                }
                if let Ok(range) = result {
                    assert_eq!(range.key, key);
                    assert_eq!(range.tokens.start, start);
                    assert_eq!(range.len(), range_end - range_start);
                    assert!(!range.is_empty());
                }
            }
        }
    }
}

#[test]
fn checked_row_range_properties() -> Result<()> {
    for width in 1..32usize {
        for rows in 1..32usize {
            let total = checked_matrix_len("property invariant matrix", rows, width)?;
            for row in 0..rows {
                let range = checked_row_range("property invariant row range", row, width, total)?;
                assert_eq!(range.start, row * width);
                assert_eq!(range.end, range.start + width);
                assert!(range.end <= total);
            }
            assert!(
                checked_row_range("property invariant out of bounds", rows, width, total).is_err()
            );
        }
    }
    Ok(())
}

#[test]
fn hot_cache_global_budget_property() -> Result<()> {
    let head_dim = 8;
    let page_bytes = checked_kv_bytes("property invariant page bytes", 4, head_dim)?;
    let budget = page_bytes * 3;
    let mut cache = HotCache::new(budget);

    for index in 0..48usize {
        let key = PageKey::new(0, 0, index * 4);
        let (k, v) = fill_page(key, 4, head_dim)?;
        let _inserted = cache.insert(key, 4, head_dim, &k, &v)?;
        assert!(cache.used_bytes() <= cache.max_bytes());
        assert!(cache.len() <= 3);
    }

    Ok(())
}

#[test]
fn hot_cache_layer_budget_property() -> Result<()> {
    let head_dim = 4;
    let page_bytes = checked_kv_bytes("property invariant layer page bytes", 2, head_dim)?;
    let mut cache = HotCache::new(page_bytes * 8);
    let _evicted = cache.set_layer_budget(7, page_bytes * 2);

    for index in 0..16usize {
        let key = PageKey::new(7, 0, index * 2);
        let (k, v) = fill_page(key, 2, head_dim)?;
        assert!(cache.insert(key, 2, head_dim, &k, &v)?);
        assert!(cache.layer_used_bytes(7) <= page_bytes * 2);
        assert!(cache.used_bytes() <= cache.max_bytes());
    }

    Ok(())
}

#[test]
fn config_validation_grid_property() {
    for page_tokens in 0..10usize {
        for head_dim in 0..10usize {
            for max_scratch_tokens in 0..10usize {
                let result = Config::builder(page_tokens, head_dim)
                    .max_scratch_tokens(max_scratch_tokens)
                    .build();
                let should_accept =
                    page_tokens > 0 && head_dim > 0 && max_scratch_tokens >= page_tokens;
                if should_accept {
                    assert!(result.is_ok());
                } else {
                    assert!(result.is_err());
                }
            }
        }
    }
}

#[test]
fn pseudo_random_cache_invariants_property() -> Result<()> {
    let mut state = 0xC0DE_A7A5_u64;
    let head_dim = 8;
    let page_bytes = checked_kv_bytes("property pseudo page bytes", 1, head_dim)?;
    let mut cache = HotCache::new(page_bytes * 11);

    for _ in 0..256 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let layer = u32::try_from(state & 0x3)
            .map_err(|_| CfrError::Numeric("property random layer conversion"))?;
        let head = u32::try_from((state >> 8) & 0x3)
            .map_err(|_| CfrError::Numeric("property random head conversion"))?;
        let token = usize::try_from((state >> 16) & 0xff)
            .map_err(|_| CfrError::Numeric("property random token conversion"))?;
        let key = PageKey::new(layer, head, token);
        let (k, v) = fill_page(key, 1, head_dim)?;
        let _ = cache.insert(key, 1, head_dim, &k, &v)?;
        if state & 1 == 0 {
            let _ = cache.get(&key);
        }
        if state & 0x10 != 0 {
            let _ = cache.remove(&key);
        }
        assert!(cache.used_bytes() <= cache.max_bytes());
        for layer in 0..4u32 {
            if let Some(layer_budget) = cache.layer_budget(layer) {
                assert!(cache.layer_used_bytes(layer) <= layer_budget);
            }
        }
    }

    Ok(())
}

#[test]
fn hot_cache_rejects_non_finite_page_values() -> Result<()> {
    let mut cache = HotCache::new(1024);
    let key = PageKey::new(0, 0, 0);
    let k = [f32::NAN, 0.0, 1.0, 2.0];
    let v = [0.0, 1.0, 2.0, 3.0];

    let error = match cache.insert(key, 2, 2, &k, &v) {
        Ok(_) => return Err(CfrError::Numeric("NaN K was accepted")),
        Err(error) => error,
    };
    assert!(matches!(error, CfrError::Numeric(_)));
    assert_eq!(cache.used_bytes(), 0);
    assert!(cache.is_empty());
    Ok(())
}

#[test]
fn manual_invalid_page_range_len_is_saturating() {
    let range = PageRange {
        key: PageKey::new(0, 0, 8),
        tokens: core::ops::Range { start: 8, end: 4 },
    };
    assert_eq!(range.len(), 0);
    assert!(range.is_empty());
}

#[test]
fn hot_cache_failed_replacement_keeps_existing_page() -> Result<()> {
    let mut cache = HotCache::new(1024);
    let key = PageKey::new(0, 0, 0);
    let good_k = [1.0f32, 2.0, 3.0, 4.0];
    let good_v = [5.0f32, 6.0, 7.0, 8.0];
    assert!(cache.insert(key, 2, 2, &good_k, &good_v)?);

    let bad_k = [f32::INFINITY, 0.0, 1.0, 2.0];
    let bad_v = [0.0, 1.0, 2.0, 3.0];
    assert!(cache.insert(key, 2, 2, &bad_k, &bad_v).is_err());

    let view = cache.get(&key).ok_or(CfrError::InvalidPage {
        key,
        message: "valid page was removed by failed replacement",
    })?;
    assert!(max_abs_diff_finite("kept K page", view.k, &good_k)? <= f32::EPSILON);
    assert!(max_abs_diff_finite("kept V page", view.v, &good_v)? <= f32::EPSILON);
    Ok(())
}
