// Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
// You can find the license file in the project root.
//
// CFR-Atlas
// The code was written for CFR-Atlas.
// 9 july 2026

//! Optional thread execution helpers for cold-page work.

use crate::{CfrError, Result};
use std::thread;

/// Configuration for optional cold-page worker threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadPoolConfig {
    /// Maximum number of worker threads used by one batch.
    pub threads: usize,
}

impl ThreadPoolConfig {
    /// Creates a thread configuration.
    pub const fn new(threads: usize) -> Result<Self> {
        if threads == 0 {
            return Err(CfrError::InvalidConfig("thread count must be non-zero"));
        }
        Ok(Self { threads })
    }

    /// Uses the platform parallelism hint, falling back to one thread.
    #[must_use]
    pub fn auto() -> Self {
        let threads = thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        Self { threads }
    }
}

/// Small dependency-free batch executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadPoolExecutor {
    config: ThreadPoolConfig,
}

impl ThreadPoolExecutor {
    /// Creates a new executor.
    #[must_use]
    pub const fn new(config: ThreadPoolConfig) -> Self {
        Self { config }
    }

    /// Returns the executor configuration.
    #[must_use]
    pub const fn config(&self) -> ThreadPoolConfig {
        self.config
    }

    /// Runs jobs in bounded batches and preserves output order.
    pub fn run<T, F>(&self, jobs: Vec<F>) -> Result<Vec<T>>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T> + Send + 'static,
    {
        let mut outputs = Vec::with_capacity(jobs.len());
        let mut iter = jobs.into_iter();

        loop {
            let batch: Vec<_> = iter.by_ref().take(self.config.threads).collect();
            if batch.is_empty() {
                break;
            }
            let mut handles = Vec::with_capacity(batch.len());
            for job in batch {
                handles.push(thread::spawn(job));
            }
            let mut batch_outputs = Vec::with_capacity(handles.len());
            let mut first_error = None;
            for handle in handles {
                match handle.join() {
                    Ok(Ok(value)) => batch_outputs.push(value),
                    Ok(Err(err)) => {
                        if first_error.is_none() {
                            first_error = Some(err);
                        }
                    }
                    Err(_) => {
                        if first_error.is_none() {
                            first_error = Some(CfrError::Regenerator(
                                "cold-page worker panicked".to_owned(),
                            ));
                        }
                    }
                }
            }
            if let Some(err) = first_error {
                return Err(err);
            }
            outputs.extend(batch_outputs);
        }

        Ok(outputs)
    }
}
