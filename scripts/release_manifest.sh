#!/usr/bin/env sh
set -eu

out_dir=${1:-target/release-artifacts}
mkdir -p "$out_dir"

cargo metadata --format-version 1 > "$out_dir/cargo-metadata.json"
cargo tree --workspace > "$out_dir/cargo-tree.txt"

files="Cargo.toml README.md LICENSE rust-toolchain.toml docs/ROADMAP.md docs/STABILIZATION.md docs/CLAIMS.md"
if [ -f Cargo.lock ]; then
    files="$files Cargo.lock"
fi

sha256sum $files > "$out_dir/SHA256SUMS"
printf '%s\n' "wrote $out_dir/cargo-metadata.json"
printf '%s\n' "wrote $out_dir/cargo-tree.txt"
printf '%s\n' "wrote $out_dir/SHA256SUMS"
