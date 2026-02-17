#!/usr/bin/env bash
set -euo pipefail

# Cross-compilation test script
# Tests against little-endian and big-endian architectures.
#
# If a glibc error pops up, run `cargo clean` first.
# See https://github.com/cross-rs/cross/issues/724

TARGETS=(
    # Big-endian
    "powerpc64-unknown-linux-gnu"
    "s390x-unknown-linux-gnu"
)

for target in "${TARGETS[@]}"; do
    echo "=== Testing $target ==="
    cross test --target "$target"
done

echo "=== All targets passed ==="
