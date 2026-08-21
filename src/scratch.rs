// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas scratch-page storage.

//! Reusable K/V scratch storage for regenerated cold pages.
//!
//! The owner keeps allocation capacity across page requests. A
//! [`crate::KvRegenerator`] must overwrite every element in the requested
//! output ranges; this module therefore grows buffers when necessary but does
//! not redundantly zero the active range before regeneration. Callers decide
//! whether to wipe the complete live buffers after use.

use crate::layout::wipe_f32;

#[derive(Debug, Default)]
pub struct ScratchBuffers {
    k: Vec<f32>,
    v: Vec<f32>,
}

impl ScratchBuffers {
    /// Exposes two mutable output slices of exactly `len` elements.
    ///
    /// New capacity is initialized safely by `Vec::resize`. Existing elements
    /// are intentionally left untouched because `KvRegenerator` owns complete
    /// initialization of the requested K/V page.
    pub fn outputs(&mut self, len: usize) -> (&mut [f32], &mut [f32]) {
        if self.k.len() < len {
            self.k.resize(len, 0.0);
        }
        if self.v.len() < len {
            self.v.resize(len, 0.0);
        }
        (&mut self.k[..len], &mut self.v[..len])
    }

    /// Borrows two initialized K/V page slices after regeneration.
    pub fn page(&self, len: usize) -> (&[f32], &[f32]) {
        (&self.k[..len], &self.v[..len])
    }

    /// Wipes every allocated live scratch element.
    pub fn wipe(&mut self) {
        wipe_f32(&mut self.k);
        wipe_f32(&mut self.v);
    }
}

impl Drop for ScratchBuffers {
    fn drop(&mut self) {
        self.wipe();
    }
}
