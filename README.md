# x86intrin
x86 intrinsics for rust

This crate will implement C-like x86 intrinsics with the similar names
to what intel uses (removing prefix underscore; e.g. `__m128i -> m128i`,
`_mm_set_epi32 -> mm_set_epi32`).

I actually need various integer SIMD arithmetics, so such functions will
be implmeneted with priority.

## Current Status

Currently most of SSE, SSE2, SSE3, SSSE3, SSE4.1, and SSE4.2 are implemented.
Some of the functions cannot be implemented since rust is not exposing necessary
functions.

AVX and AVX2 implementation are ongoing.

After all done, I'd like to contribute to rust libraries to support missing functions.
