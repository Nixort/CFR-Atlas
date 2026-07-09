// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

#![no_main]

use cfr_atlas::prelude::*;
use libfuzzer_sys::fuzz_target;

fn read_u32(data: &[u8], offset: usize) -> u32 {
    let mut bytes = [0u8; 4];
    if let Some(slice) = data.get(offset..offset + 4) {
        bytes.copy_from_slice(slice);
    }
    u32::from_le_bytes(bytes)
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 24 {
        return;
    }

    let page_tokens = usize_from_u32_or_zero(read_u32(data, 0)) % 4096;
    let head_dim = usize_from_u32_or_zero(read_u32(data, 4)) % 2048;
    let max_scratch_tokens = usize_from_u32_or_zero(read_u32(data, 8)) % 4096;
    let hot_cache_bytes = usize_from_u32_or_zero(read_u32(data, 12)) % (1 << 24);
    let start = usize_from_u32_or_zero(read_u32(data, 16)) % 4096;
    let len = usize_from_u32_or_zero(read_u32(data, 20)) % 4096;
    let key = PageKey::new(0, 0, start);

    let _ = Config::builder(page_tokens, head_dim)
        .hot_cache_bytes(hot_cache_bytes)
        .max_scratch_tokens(max_scratch_tokens)
        .build();

    let _ = PageRange::new(key, start..start.saturating_add(len));

    if head_dim > 0 && len > 0 {
        let mut cache = HotCache::new(hot_cache_bytes);
        if let Ok(values) = checked_matrix_len("fuzz matrix", len, head_dim) {
            if values <= 16_384 {
                let k = vec![0.0; values];
                let v = vec![0.0; values];
                let _ = cache.insert(key, len, head_dim, &k, &v);
                let _ = cache.get(&key);
                let _ = cache.remove(&key);
            }
        }
    }
});

fn usize_from_u32_or_zero(value: u32) -> usize {
    usize::try_from(value).map_or(0, |value| value)
}
