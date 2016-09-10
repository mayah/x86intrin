# x86intrin
x86 intrinsics for rust

[![Crates.io Status](https://img.shields.io/crates/v/x86intrin.svg)](https://crates.io/crates/x86intrin)

This crate will implement C-like x86 intrinsics with the similar names
to what intel uses (removing prefix underscore; e.g. `__m128i -> m128i`,
`_mm_set_epi32 -> mm_set_epi32`).

I actually need various integer SIMD arithmetics, so such functions will
be implemeneted with priority.

# Current Status

Currently most of SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, and AVX are implemented.
Some of the functions cannot be implemented since rust is not exposing necessary
functions.

AVX2 implementation is ongoing.

After all done, I'd like to contribute to rust libraries to support missing functions.

# Note

To build with `cargo`, you need to set `target-cpu` or `target-feature` in `RUSTFLAGS`.

For example:
```
$ RUSTFLAGS="-C target-cpu=native" cargo build
$ RUSTFLAGS="-C target-feature=+sse3" cargo build
```
