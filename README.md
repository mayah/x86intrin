# x86intrin
x86 intrinsics for rust

This crate will implement C-like x86 intrinsics with the similar names
to what intel uses (removing prefix underscore; e.g. __m128i -> m128i,
_mm_set_epi32 -> mm_set_epi32).

I actually need various integer SIMD arithmetics, so such functions will
be implmeneted with priority.
