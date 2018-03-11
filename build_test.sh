#!/bin/sh

set -x

# build with various flags, and confirms no warning is shown.

RUSTFLAGS="-C target-feature=+sse" cargo build
RUSTFLAGS="-C target-feature=+sse2" cargo build
RUSTFLAGS="-C target-feature=+sse3" cargo build
RUSTFLAGS="-C target-feature=+ssse3" cargo build
RUSTFLAGS="-C target-feature=+sse4.1" cargo build
RUSTFLAGS="-C target-feature=+sse4.2" cargo build
RUSTFLAGS="-C target-feature=+avx" cargo build
RUSTFLAGS="-C target-feature=+avx2" cargo build
RUSTFLAGS="-C target-feature=+bmi2" cargo build
