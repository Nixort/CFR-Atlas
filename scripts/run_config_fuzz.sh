#!/usr/bin/env sh
set -eu

# Run the CFR-Atlas Phase 1 fuzz target with the nightly toolchain required by
# cargo-fuzz/libFuzzer sanitizer instrumentation.

if ! command -v rustup >/dev/null 2>&1; then
    echo "rustup is required to select the nightly toolchain for cargo-fuzz." >&2
    exit 1
fi

if ! rustup toolchain list | grep -q '^nightly'; then
    echo "Installing the nightly Rust toolchain required by cargo-fuzz..." >&2
    rustup toolchain install nightly
fi

cargo +nightly fuzz run config_page_validation "$@"
