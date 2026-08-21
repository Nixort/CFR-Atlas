#!/usr/bin/env python3
"""Fetch the public Tiny Shakespeare corpus used by the CFR-Atlas benchmark."""

from __future__ import annotations

import hashlib
import pathlib
import sys
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEFAULT_DESTINATION = pathlib.Path("bench/data/tinyshakespeare.txt")


def main() -> int:
    destination = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DESTINATION
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=30) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError("downloaded corpus is empty")
    destination.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    print(f"source_url={URL}")
    print(f"bytes={len(payload)}")
    print(f"sha256={digest}")
    print(f"path={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
