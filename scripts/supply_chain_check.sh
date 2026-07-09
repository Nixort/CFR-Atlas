#!/usr/bin/env sh
set -eu

cargo deny check advisories bans licenses sources
