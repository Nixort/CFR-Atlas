<!--
Copyright Nixort & Itan Winter <https://github.com/Nixort/CFR-Atlas> 2026.

License: MIT
You can find the license file in the project root.

CFR-Atlas
The code was written for CFR-Atlas.
9 july 2026
-->

# Fuzzing

CFR-Atlas keeps the normal workspace on stable Rust, pinned by the root
`rust-toolchain.toml`. The fuzz target is different: `cargo-fuzz` uses
libFuzzer and sanitizer instrumentation, which require a nightly compiler.

Run the Phase 1 fuzz target with:

```sh
rustup toolchain install nightly
cargo +nightly fuzz run config_page_validation
```

Or use the helper script from the repository root:

```sh
./scripts/run_config_fuzz.sh
```

The fuzz package is excluded from the normal workspace so stable commands stay
clean:

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
```
