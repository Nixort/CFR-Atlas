#!/usr/bin/env sh
set -eu

artifact_dir=${1:-target/release-artifacts}
key=${GPG_SIGNING_KEY:?set GPG_SIGNING_KEY to the signing key id}

gpg --batch --yes --local-user "$key" --detach-sign --armor "$artifact_dir/SHA256SUMS"
printf '%s\n' "wrote $artifact_dir/SHA256SUMS.asc"
