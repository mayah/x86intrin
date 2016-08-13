use super::*;
use super::{simd_shuffle8};

// vaddpd
// __m256d _mm256_add_pd (__m256d a, __m256d b)
// vaddps
// __m256 _mm256_add_ps (__m256 a, __m256 b)
// vaddsubpd
// __m256d _mm256_addsub_pd (__m256d a, __m256d b)
// vaddsubps
// __m256 _mm256_addsub_ps (__m256 a, __m256 b)
// vandpd
// __m256d _mm256_and_pd (__m256d a, __m256d b)
// vandps
// __m256 _mm256_and_ps (__m256 a, __m256 b)
// vandnpd
// __m256d _mm256_andnot_pd (__m256d a, __m256d b)
// vandnps
// __m256 _mm256_andnot_ps (__m256 a, __m256 b)
// vblendpd
// __m256d _mm256_blend_pd (__m256d a, __m256d b, const int imm8)
// vblendps
// __m256 _mm256_blend_ps (__m256 a, __m256 b, const int imm8)
// vblendvpd
// __m256d _mm256_blendv_pd (__m256d a, __m256d b, __m256d mask)
// vblendvps
// __m256 _mm256_blendv_ps (__m256 a, __m256 b, __m256 mask)
// vbroadcastf128
// __m256d _mm256_broadcast_pd (__m128d const * mem_addr)
// vbroadcastf128
// __m256 _mm256_broadcast_ps (__m128 const * mem_addr)
// vbroadcastsd
// __m256d _mm256_broadcast_sd (double const * mem_addr)
// vbroadcastss
// __m128 _mm_broadcast_ss (float const * mem_addr)
// vbroadcastss
// __m256 _mm256_broadcast_ss (float const * mem_addr)
// __m256 _mm256_castpd_ps (__m256d a)
// __m256i _mm256_castpd_si256 (__m256d a)
// __m256d _mm256_castpd128_pd256 (__m128d a)
// __m128d _mm256_castpd256_pd128 (__m256d a)
// __m256d _mm256_castps_pd (__m256 a)
// __m256i _mm256_castps_si256 (__m256 a)
// __m256 _mm256_castps128_ps256 (__m128 a)
// __m128 _mm256_castps256_ps128 (__m256 a)
// __m256i _mm256_castsi128_si256 (__m128i a)
// __m256d _mm256_castsi256_pd (__m256i a)
// __m256 _mm256_castsi256_ps (__m256i a)
// __m128i _mm256_castsi256_si128 (__m256i a)
// vroundpd
// __m256d _mm256_ceil_pd (__m256d a)
// vroundps
// __m256 _mm256_ceil_ps (__m256 a)
// vcmppd
// __m128d _mm_cmp_pd (__m128d a, __m128d b, const int imm8)
// vcmppd
// __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
// vcmpps
// __m128 _mm_cmp_ps (__m128 a, __m128 b, const int imm8)
// vcmpps
// __m256 _mm256_cmp_ps (__m256 a, __m256 b, const int imm8)
// vcmpsd
// __m128d _mm_cmp_sd (__m128d a, __m128d b, const int imm8)
// vcmpss
// __m128 _mm_cmp_ss (__m128 a, __m128 b, const int imm8)
// vcvtdq2pd
// __m256d _mm256_cvtepi32_pd (__m128i a)
// vcvtdq2ps
// __m256 _mm256_cvtepi32_ps (__m256i a)
// vcvtpd2dq
// __m128i _mm256_cvtpd_epi32 (__m256d a)
// vcvtpd2ps
// __m128 _mm256_cvtpd_ps (__m256d a)
// vcvtps2dq
// __m256i _mm256_cvtps_epi32 (__m256 a)
// vcvtps2pd
// __m256d _mm256_cvtps_pd (__m128 a)
// vcvttpd2dq
// __m128i _mm256_cvttpd_epi32 (__m256d a)
// vcvttps2dq
// __m256i _mm256_cvttps_epi32 (__m256 a)
// vdivpd
// __m256d _mm256_div_pd (__m256d a, __m256d b)
// vdivps
// __m256 _mm256_div_ps (__m256 a, __m256 b)
// vdpps
// __m256 _mm256_dp_ps (__m256 a, __m256 b, const int imm8)
// ...
// __int16 _mm256_extract_epi16 (__m256i a, const int index)
// ...
// __int32 _mm256_extract_epi32 (__m256i a, const int index)
// ...
// __int64 _mm256_extract_epi64 (__m256i a, const int index)
// ...
// __int8 _mm256_extract_epi8 (__m256i a, const int index)
// vextractf128
// __m128d _mm256_extractf128_pd (__m256d a, const int imm8)
// vextractf128
// __m128 _mm256_extractf128_ps (__m256 a, const int imm8)
// vextractf128
// __m128i _mm256_extractf128_si256 (__m256i a, const int imm8)
// vroundpd
// __m256d _mm256_floor_pd (__m256d a)
// vroundps
// __m256 _mm256_floor_ps (__m256 a)
// vhaddpd
// __m256d _mm256_hadd_pd (__m256d a, __m256d b)
// vhaddps
// __m256 _mm256_hadd_ps (__m256 a, __m256 b)
// vhsubpd
// __m256d _mm256_hsub_pd (__m256d a, __m256d b)
// vhsubps
// __m256 _mm256_hsub_ps (__m256 a, __m256 b)
// ...
// __m256i _mm256_insert_epi16 (__m256i a, __int16 i, const int index)
// ...
// __m256i _mm256_insert_epi32 (__m256i a, __int32 i, const int index)
// ...
// __m256i _mm256_insert_epi64 (__m256i a, __int64 i, const int index)
// ...
// __m256i _mm256_insert_epi8 (__m256i a, __int8 i, const int index)
// vinsertf128
// __m256d _mm256_insertf128_pd (__m256d a, __m128d b, int imm8)
// vinsertf128
// __m256 _mm256_insertf128_ps (__m256 a, __m128 b, int imm8)
// vinsertf128
// __m256i _mm256_insertf128_si256 (__m256i a, __m128i b, int imm8)
// vlddqu
// __m256i _mm256_lddqu_si256 (__m256i const * mem_addr)
// vmovapd
// __m256d _mm256_load_pd (double const * mem_addr)
// vmovaps
// __m256 _mm256_load_ps (float const * mem_addr)
// vmovdqa
// __m256i _mm256_load_si256 (__m256i const * mem_addr)
// vmovupd
// __m256d _mm256_loadu_pd (double const * mem_addr)
// vmovups
// __m256 _mm256_loadu_ps (float const * mem_addr)
// vmovdqu
// __m256i _mm256_loadu_si256 (__m256i const * mem_addr)
// ...
// __m256 _mm256_loadu2_m128 (float const* hiaddr, float const* loaddr)
// ...
// __m256d _mm256_loadu2_m128d (double const* hiaddr, double const* loaddr)
// ...
// __m256i _mm256_loadu2_m128i (__m128i const* hiaddr, __m128i const* loaddr)
// vmaskmovpd
// __m128d _mm_maskload_pd (double const * mem_addr, __m128i mask)
// vmaskmovpd
// __m256d _mm256_maskload_pd (double const * mem_addr, __m256i mask)
// vmaskmovps
// __m128 _mm_maskload_ps (float const * mem_addr, __m128i mask)
// vmaskmovps
// __m256 _mm256_maskload_ps (float const * mem_addr, __m256i mask)
// vmaskmovpd
// void _mm_maskstore_pd (double * mem_addr, __m128i mask, __m128d a)
// vmaskmovpd
// void _mm256_maskstore_pd (double * mem_addr, __m256i mask, __m256d a)
// vmaskmovps
// void _mm_maskstore_ps (float * mem_addr, __m128i mask, __m128 a)
// vmaskmovps
// void _mm256_maskstore_ps (float * mem_addr, __m256i mask, __m256 a)
// vmaxpd
// __m256d _mm256_max_pd (__m256d a, __m256d b)
// vmaxps
// __m256 _mm256_max_ps (__m256 a, __m256 b)
// vminpd
// __m256d _mm256_min_pd (__m256d a, __m256d b)
// vminps
// __m256 _mm256_min_ps (__m256 a, __m256 b)
// vmovddup
// __m256d _mm256_movedup_pd (__m256d a)
// vmovshdup
// __m256 _mm256_movehdup_ps (__m256 a)
// vmovsldup
// __m256 _mm256_moveldup_ps (__m256 a)
// vmovmskpd
// int _mm256_movemask_pd (__m256d a)
// vmovmskps
// int _mm256_movemask_ps (__m256 a)
// vmulpd
// __m256d _mm256_mul_pd (__m256d a, __m256d b)
// vmulps
// __m256 _mm256_mul_ps (__m256 a, __m256 b)
// vorpd
// __m256d _mm256_or_pd (__m256d a, __m256d b)
// vorps
// __m256 _mm256_or_ps (__m256 a, __m256 b)
// vpermilpd
// __m128d _mm_permute_pd (__m128d a, int imm8)
// vpermilpd
// __m256d _mm256_permute_pd (__m256d a, int imm8)
// vpermilps
// __m128 _mm_permute_ps (__m128 a, int imm8)
// vpermilps
// __m256 _mm256_permute_ps (__m256 a, int imm8)
// vperm2f128
// __m256d _mm256_permute2f128_pd (__m256d a, __m256d b, int imm8)
// vperm2f128
// __m256 _mm256_permute2f128_ps (__m256 a, __m256 b, int imm8)
// vperm2f128
// __m256i _mm256_permute2f128_si256 (__m256i a, __m256i b, int imm8)
// vpermilpd
// __m128d _mm_permutevar_pd (__m128d a, __m128i b)
// vpermilpd
// __m256d _mm256_permutevar_pd (__m256d a, __m256i b)
// vpermilps
// __m128 _mm_permutevar_ps (__m128 a, __m128i b)
// vpermilps
// __m256 _mm256_permutevar_ps (__m256 a, __m256i b)
// vrcpps
// __m256 _mm256_rcp_ps (__m256 a)
// vroundpd
// __m256d _mm256_round_pd (__m256d a, int rounding)
// vroundps
// __m256 _mm256_round_ps (__m256 a, int rounding)
// vrsqrtps
// __m256 _mm256_rsqrt_ps (__m256 a)

// ...
// __m256i _mm256_set_epi16 (short e15, short e14, short e13, short e12, short e11, short e10, short e9, short e8, short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0)
#[inline]
pub fn mm256_set_epi16(e15: i16, e14: i16, e13: i16, e12: i16, e11: i16, e10: i16, e9: i16, e8: i16,
                       e7: i16, e6: i16, e5: i16, e4: i16, e3: i16, e2: i16, e1: i16, e0: i16) -> m256i {
    i16x16(e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15).as_m256i()
}

// ...
// __m256i _mm256_set_epi32 (int e7, int e6, int e5, int e4, int e3, int e2, int e1, int e0)
#[inline]
pub fn mm256_set_epi32(e7: i32, e6: i32, e5: i32, e4: i32, e3: i32, e2: i32, e1: i32, e0: i32) -> m256i {
    i32x8(e0, e1, e2, e3, e4, e5, e6, e7).as_m256i()
}

// ...
// __m256i _mm256_set_epi64x (__int64 e3, __int64 e2, __int64 e1, __int64 e0)
#[inline]
pub fn mm256_set_epi64x(e3: i64, e2: i64, e1: i64, e0: i64) -> m256i {
    i64x4(e0, e1, e2, e3).as_m256i()
}

// ...
// __m256i _mm256_set_epi8 (char e31, char e30, char e29, char e28, char e27, char e26, char e25, char e24, char e23, char e22, char e21, char e20, char e19, char e18, char e17, char e16, char e15, char e14, char e13, char e12, char e11, char e10, char e9, char e8, char e7, char e6, char e5, char e4, char e3, char e2, char e1, char e0)
#[inline]
pub fn mm256_set_epi8(e31: i8, e30: i8, e29: i8, e28: i8, e27: i8, e26: i8, e25: i8, e24: i8,
                      e23: i8, e22: i8, e21: i8, e20: i8, e19: i8, e18: i8, e17: i8, e16: i8,
                      e15: i8, e14: i8, e13: i8, e12: i8, e11: i8, e10: i8, e9: i8, e8: i8,
                      e7: i8, e6: i8, e5: i8, e4: i8, e3: i8, e2: i8, e1: i8, e0: i8) -> m256i {
    i8x32(e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15,
          e16, e17, e18, e19, e20, e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, e31).as_m256i()
}

// vinsertf128
// __m256 _mm256_set_m128 (__m128 hi, __m128 lo)
#[inline]
pub fn mm256_set_m128(hi: m128, lo: m128) -> m256 {
    unsafe { simd_shuffle8(lo, hi, [0, 1, 2, 3, 4, 5, 6, 7]) }
}

// vinsertf128
// __m256d _mm256_set_m128d (__m128d hi, __m128d lo)
#[inline]
pub fn mm256_set_m128d(hi: m128d, lo: m128d) -> m256d {
    mm256_set_m128(hi.as_m128(), lo.as_m128()).as_m256d()
}

// vinsertf128
// __m256i _mm256_set_m128i (__m128i hi, __m128i lo)
#[inline]
pub fn mm256_set_m128i(hi: m128i, lo: m128i) -> m256i {
    mm256_set_m128(hi.as_m128(), lo.as_m128()).as_m256i()
}

// ...
// __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
#[inline]
pub fn mm256_set_pd(e3: f64, e2: f64, e1: f64, e0: f64) -> m256d {
    f64x4(e0, e1, e2, e3).as_m256d()
}

// ...
// __m256 _mm256_set_ps (float e7, float e6, float e5, float e4, float e3, float e2, float e1, float e0)
#[inline]
pub fn mm256_set_ps(e7: f32, e6: f32, e5: f32, e4: f32, e3: f32, e2: f32, e1: f32, e0: f32) -> m256 {
    f32x8(e0, e1, e2, e3, e4, e5, e6, e7).as_m256()
}

// ...
// __m256i _mm256_set1_epi16 (short a)
#[inline]
pub fn mm256_set1_epi16(a: i16) -> m256i {
    i16x16(a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a).as_m256i()
}

// ...
// __m256i _mm256_set1_epi32 (int a)
#[inline]
pub fn mm256_set1_epi32(a: i32) -> m256i {
    i32x8(a, a, a, a, a, a, a, a).as_m256i()
}

// ...
// __m256i _mm256_set1_epi64x (long long a)
#[inline]
pub fn mm256_set1_epi64x(a: i64) -> m256i {
    i64x4(a, a, a, a).as_m256i()
}

// ...
// __m256i _mm256_set1_epi8 (char a)
#[inline]
pub fn mm256_set1_epi8(a: i8) -> m256i {
    i8x32(a, a, a, a, a, a, a, a,
          a, a, a, a, a, a, a, a,
          a, a, a, a, a, a, a, a,
          a, a, a, a, a, a, a, a).as_m256i()
}

// ...
// __m256d _mm256_set1_pd (double a)
#[inline]
pub fn mm256_set1_pd(a: f64) -> m256d {
    f64x4(a, a, a, a).as_m256d()
}

// ...
// __m256 _mm256_set1_ps (float a)
#[inline]
pub fn mm256_set1_ps(a: f32) -> m256 {
    f32x8(a, a, a, a, a, a, a, a).as_m256()
}

// ...
// __m256i _mm256_setr_epi16 (short e15, short e14, short e13, short e12, short e11, short e10, short e9, short e8, short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0)
#[inline]
pub fn mm256_setr_epi16(e15: i16, e14: i16, e13: i16, e12: i16, e11: i16, e10: i16, e9: i16, e8: i16,
                       e7: i16, e6: i16, e5: i16, e4: i16, e3: i16, e2: i16, e1: i16, e0: i16) -> m256i {
    i16x16(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0).as_m256i()
}

// ...
// __m256i _mm256_setr_epi32 (int e7, int e6, int e5, int e4, int e3, int e2, int e1, int e0)
#[inline]
pub fn mm256_setr_epi32(e7: i32, e6: i32, e5: i32, e4: i32, e3: i32, e2: i32, e1: i32, e0: i32) -> m256i {
    i32x8(e7, e6, e5, e4, e3, e2, e1, e0).as_m256i()
}

// ...
// __m256i _mm256_setr_epi64x (__int64 e3, __int64 e2, __int64 e1, __int64 e0)
#[inline]
pub fn mm256_setr_epi64x(e3: i64, e2: i64, e1: i64, e0: i64) -> m256i {
    i64x4(e3, e2, e1, e0).as_m256i()
}

// ...
// __m256i _mm256_setr_epi8 (char e31, char e30, char e29, char e28, char e27, char e26, char e25, char e24, char e23, char e22, char e21, char e20, char e19, char e18, char e17, char e16, char e15, char e14, char e13, char e12, char e11, char e10, char e9, char e8, char e7, char e6, char e5, char e4, char e3, char e2, char e1, char e0)
#[inline]
pub fn mm256_setr_epi8(e31: i8, e30: i8, e29: i8, e28: i8, e27: i8, e26: i8, e25: i8, e24: i8,
                       e23: i8, e22: i8, e21: i8, e20: i8, e19: i8, e18: i8, e17: i8, e16: i8,
                       e15: i8, e14: i8, e13: i8, e12: i8, e11: i8, e10: i8, e9: i8, e8: i8,
                       e7: i8, e6: i8, e5: i8, e4: i8, e3: i8, e2: i8, e1: i8, e0: i8) -> m256i {
    i8x32(e31, e30, e29, e28, e27, e26, e25, e24,
          e23, e22, e21, e20, e19, e18, e17, e16,
          e15, e14, e13, e12, e11, e10, e9, e8,
          e7, e6, e5, e4, e3, e2, e1, e0).as_m256i()
}

// vinsertf128
// __m256 _mm256_setr_m128 (__m128 lo, __m128 hi)
#[inline]
pub fn mm256_setr_m128(lo: m128, hi: m128) -> m256 {
    mm256_set_m128(hi.as_m128(), lo.as_m128())
}

// vinsertf128
// __m256d _mm256_setr_m128d (__m128d lo, __m128d hi)
#[inline]
pub fn mm256_setr_m128d(lo: m128d, hi: m128d) -> m256d {
    mm256_set_m128(hi.as_m128(), lo.as_m128()).as_m256d()
}

// vinsertf128
// __m256i _mm256_setr_m128i (__m128i lo, __m128i hi)
#[inline]
pub fn mm256_setr_m128i(lo: m128i, hi: m128i) -> m256i {
    mm256_set_m128(hi.as_m128(), lo.as_m128()).as_m256i()
}

// ...
// __m256d _mm256_setr_pd (double e3, double e2, double e1, double e0)
#[inline]
pub fn mm256_setr_pd(e3: f64, e2: f64, e1: f64, e0: f64) -> m256d {
    f64x4(e3, e2, e1, e0).as_m256d()
}

// ...
// __m256 _mm256_setr_ps (float e7, float e6, float e5, float e4, float e3, float e2, float e1, float e0)
#[inline]
pub fn mm256_setr_ps(e7: f32, e6: f32, e5: f32, e4: f32, e3: f32, e2: f32, e1: f32, e0: f32) -> m256 {
    f32x8(e7, e6, e5, e4, e3, e2, e1, e0).as_m256()
}

// vxorpd
// __m256d _mm256_setzero_pd (void)
#[inline]
pub fn mm256_setzero_pd() -> m256d {
    f64x4(0.0, 0.0, 0.0, 0.0).as_m256d()
}

// vxorps
// __m256 _mm256_setzero_ps (void)
#[inline]
pub fn mm256_setzero_ps() -> m256 {
    f32x8(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).as_m256()
}

// vpxor
// __m256i _mm256_setzero_si256 (void)
#[inline]
pub fn mm256_setzero_si256() -> m256i {
    i64x4(0, 0, 0, 0).as_m256i()
}

// vshufpd
// __m256d _mm256_shuffle_pd (__m256d a, __m256d b, const int imm8)
// vshufps
// __m256 _mm256_shuffle_ps (__m256 a, __m256 b, const int imm8)
// vsqrtpd
// __m256d _mm256_sqrt_pd (__m256d a)
// vsqrtps
// __m256 _mm256_sqrt_ps (__m256 a)
// vmovapd
// void _mm256_store_pd (double * mem_addr, __m256d a)
// vmovaps
// void _mm256_store_ps (float * mem_addr, __m256 a)
// vmovdqa
// void _mm256_store_si256 (__m256i * mem_addr, __m256i a)
// vmovupd
// void _mm256_storeu_pd (double * mem_addr, __m256d a)
// vmovups
// void _mm256_storeu_ps (float * mem_addr, __m256 a)
// vmovdqu
// void _mm256_storeu_si256 (__m256i * mem_addr, __m256i a)
// ...
// void _mm256_storeu2_m128 (float* hiaddr, float* loaddr, __m256 a)
// ...
// void _mm256_storeu2_m128d (double* hiaddr, double* loaddr, __m256d a)
// ...
// void _mm256_storeu2_m128i (__m128i* hiaddr, __m128i* loaddr, __m256i a)
// vmovntpd
// void _mm256_stream_pd (double * mem_addr, __m256d a)
// vmovntps
// void _mm256_stream_ps (float * mem_addr, __m256 a)
// vmovntdq
// void _mm256_stream_si256 (__m256i * mem_addr, __m256i a)
// vsubpd
// __m256d _mm256_sub_pd (__m256d a, __m256d b)
// vsubps
// __m256 _mm256_sub_ps (__m256 a, __m256 b)
// vtestpd
// int _mm_testc_pd (__m128d a, __m128d b)
// vtestpd
// int _mm256_testc_pd (__m256d a, __m256d b)
// vtestps
// int _mm_testc_ps (__m128 a, __m128 b)
// vtestps
// int _mm256_testc_ps (__m256 a, __m256 b)
// vptest
// int _mm256_testc_si256 (__m256i a, __m256i b)
// vtestpd
// int _mm_testnzc_pd (__m128d a, __m128d b)
// vtestpd
// int _mm256_testnzc_pd (__m256d a, __m256d b)
// vtestps
// int _mm_testnzc_ps (__m128 a, __m128 b)
// vtestps
// int _mm256_testnzc_ps (__m256 a, __m256 b)
// vptest
// int _mm256_testnzc_si256 (__m256i a, __m256i b)
// vtestpd
// int _mm_testz_pd (__m128d a, __m128d b)
// vtestpd
// int _mm256_testz_pd (__m256d a, __m256d b)
// vtestps
// int _mm_testz_ps (__m128 a, __m128 b)
// vtestps
// int _mm256_testz_ps (__m256 a, __m256 b)
// vptest
// int _mm256_testz_si256 (__m256i a, __m256i b)
// __m128d _mm_undefined_pd (void)
// __m256d _mm256_undefined_pd (void)
// __m128 _mm_undefined_ps (void)
// __m256 _mm256_undefined_ps (void)
// __m128i _mm_undefined_si128 (void)
// __m256i _mm256_undefined_si256 (void)
// vunpckhpd
// __m256d _mm256_unpackhi_pd (__m256d a, __m256d b)
// vunpckhps
// __m256 _mm256_unpackhi_ps (__m256 a, __m256 b)
// vunpcklpd
// __m256d _mm256_unpacklo_pd (__m256d a, __m256d b)
// vunpcklps
// __m256 _mm256_unpacklo_ps (__m256 a, __m256 b)
// vxorpd
// __m256d _mm256_xor_pd (__m256d a, __m256d b)
// vxorps
// __m256 _mm256_xor_ps (__m256 a, __m256 b)
// vzeroall
// void _mm256_zeroall (void)
// vzeroupper
// void _mm256_zeroupper (void)

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test_mm256_set_int() {
        assert_eq!(mm256_set_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                   17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32).as_i8x32().as_array(),
                   [32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
                    16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(mm256_set_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16).as_i16x16().as_array(),
                   [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8).as_i32x8().as_array(), [8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(mm256_set_epi64x(1, 2, 3, 4).as_i64x4().as_array(), [4, 3, 2, 1]);

        assert_eq!(mm256_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                   17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32).as_i8x32().as_array(),
                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]);
        assert_eq!(mm256_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16).as_i16x16().as_array(),
                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        assert_eq!(mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8).as_i32x8().as_array(), [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(mm256_setr_epi64x(1, 2, 3, 4).as_i64x4().as_array(), [1, 2, 3, 4]);

        assert_eq!(mm256_set1_epi8(1).as_i8x32().as_array(),
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(mm256_set1_epi16(1).as_i16x16().as_array(),
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(mm256_set1_epi32(1).as_i32x8().as_array(), [1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(mm256_set1_epi64x(1).as_i64x4().as_array(), [1, 1, 1, 1]);

        assert_eq!(mm256_setzero_si256().as_i64x4().as_array(), [0, 0, 0, 0]);

        let lo = mm_set_epi64x(2, 1);
        let hi = mm_set_epi64x(4, 3);
        assert_eq!(mm256_set_m128i(hi, lo).as_i64x4().as_array(), [1, 2, 3, 4]);
        assert_eq!(mm256_setr_m128i(lo, hi).as_i64x4().as_array(), [1, 2, 3, 4]);
    }

    #[test]
    fn test_mm256_set_float() {
        assert_eq!(mm256_set_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0).as_f32x8().as_array(),
                   [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        assert_eq!(mm256_set_pd(1.0, 2.0, 3.0, 4.0).as_f64x4().as_array(),
                   [4.0, 3.0, 2.0, 1.0]);

        assert_eq!(mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0).as_f32x8().as_array(),
                   [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(mm256_setr_pd(1.0, 2.0, 3.0, 4.0).as_f64x4().as_array(),
                   [1.0, 2.0, 3.0, 4.0]);

        assert_eq!(mm256_set1_pd(1.0).as_f64x4().as_array(), [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(mm256_set1_ps(1.0).as_f32x8().as_array(), [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        assert_eq!(mm256_setzero_pd().as_f64x4().as_array(), [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(mm256_setzero_ps().as_f32x8().as_array(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let lo32 = mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let hi32 = mm_setr_ps(5.0, 6.0, 7.0, 8.0);
        let lo64 = mm_setr_pd(1.0, 2.0);
        let hi64 = mm_setr_pd(3.0, 4.0);
        assert_eq!(mm256_set_m128(hi32, lo32).as_f32x8().as_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(mm256_setr_m128(lo32, hi32).as_f32x8().as_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        assert_eq!(mm256_set_m128d(hi64, lo64).as_f64x4().as_array(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(mm256_setr_m128d(lo64, hi64).as_f64x4().as_array(), [1.0, 2.0, 3.0, 4.0]);

    }
}
