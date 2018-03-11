#![allow(improper_ctypes)]  // TODO(mayah): Remove this flag

use super::*;
use super::{simd_mul,
            simd_eq,
            simd_shuffle2, simd_shuffle4, simd_shuffle8};

extern {
    #[link_name = "llvm.x86.sse41.pblendvb"]
    fn sse41_pblendvb(a: i8x16, b: i8x16, c: i8x16) -> i8x16;
    #[link_name = "llvm.x86.sse41.blendvpd"]
    fn sse41_blendvpd(a: m128d, b: m128d, c: m128d) -> m128d;
    #[link_name = "llvm.x86.sse41.blendvps"]
    fn sse41_blendvps(a: m128, b: m128, c: m128) -> m128;

    #[link_name = "llvm.x86.sse41.round.ss"]
    fn sse41_round_ss(a: m128, b: m128, c: i32) -> m128;
    #[link_name = "llvm.x86.sse41.round.ps"]
    fn sse41_round_ps(a: m128, b: i32) -> m128;
    #[link_name = "llvm.x86.sse41.round.sd"]
    fn sse41_round_sd(a: m128d, b: m128d, c: i32) -> m128d;
    #[link_name = "llvm.x86.sse41.round.pd"]
    fn sse41_round_pd(a: m128d, b: i32) -> m128d;

    #[link_name = "llvm.x86.sse41.insertps"]
    fn sse41_insertps(a: m128, b: m128, c: u8) -> m128;

    #[link_name = "llvm.x86.sse41.pmovsxbd"]
    pub fn sse41_pmovsxbd(a: i8x16) -> i32x4;
    #[link_name = "llvm.x86.sse41.pmovsxbq"]
    pub fn sse41_pmovsxbq(a: i8x16) -> i64x2;
    #[link_name = "llvm.x86.sse41.pmovsxbw"]
    pub fn sse41_pmovsxbw(a: i8x16) -> i16x8;
    #[link_name = "llvm.x86.sse41.pmovsxdq"]
    pub fn sse41_pmovsxdq(a: i32x4) -> i64x2;
    #[link_name = "llvm.x86.sse41.pmovsxwd"]
    pub fn sse41_pmovsxwd(a: i16x8) -> i32x4;
    #[link_name = "llvm.x86.sse41.pmovsxwq"]
    pub fn sse41_pmovsxwq(a: i16x8) -> i64x2;
    #[link_name = "llvm.x86.sse41.pmovzxbd"]
    pub fn sse41_pmovzxbd(a: i8x16) -> i32x4;
    #[link_name = "llvm.x86.sse41.pmovzxbq"]
    pub fn sse41_pmovzxbq(a: i8x16) -> i64x2;
    #[link_name = "llvm.x86.sse41.pmovzxbw"]
    pub fn sse41_pmovzxbw(a: i8x16) -> i16x8;
    #[link_name = "llvm.x86.sse41.pmovzxdq"]
    pub fn sse41_pmovzxdq(a: i32x4) -> i64x2;
    #[link_name = "llvm.x86.sse41.pmovzxwd"]
    pub fn sse41_pmovzxwd(a: i16x8) -> i32x4;
    #[link_name = "llvm.x86.sse41.pmovzxwq"]
    pub fn sse41_pmovzxwq(a: i16x8) -> i64x2;

    #[link_name = "llvm.x86.sse41.ptestc"]
    fn sse41_ptestc(a: i64x2, b: i64x2) -> i32;
    #[link_name = "llvm.x86.sse41.ptestnzc"]
    fn sse41_ptestnzc(a: i64x2, b: i64x2) -> i32;
    #[link_name = "llvm.x86.sse41.ptestz"]
    fn sse41_ptestz(a: i64x2, b: i64x2) -> i32;
}

extern "platform-intrinsic" {
    fn x86_mm_dp_pd(x: m128d, y: m128d, z: i32) -> m128d;
    fn x86_mm_dp_ps(x: m128, y: m128, z: i32) -> m128;

    fn x86_mm_max_epi32(x: i32x4, y: i32x4) -> i32x4;
    fn x86_mm_max_epi8(x: i8x16, y: i8x16) -> i8x16;
    fn x86_mm_max_epu16(x: u16x8, y: u16x8) -> u16x8;
    fn x86_mm_max_epu32(x: u32x4, y: u32x4) -> u32x4;
    fn x86_mm_min_epi32(x: i32x4, y: i32x4) -> i32x4;
    fn x86_mm_min_epi8(x: i8x16, y: i8x16) -> i8x16;
    fn x86_mm_min_epu16(x: u16x8, y: u16x8) -> u16x8;
    fn x86_mm_min_epu32(x: u32x4, y: u32x4) -> u32x4;

    fn x86_mm_minpos_epu16(x: u16x8) -> u16x8;
    fn x86_mm_mpsadbw_epu8(x: u8x16, y: u8x16, z: i32) -> u16x8;
    fn x86_mm_mul_epi32(x: i32x4, y: i32x4) -> i64x2;
    fn x86_mm_packus_epi32(x: i32x4, y: i32x4) -> u16x8;
}

/* SSE4 Rounding macros. */
//#define _MM_FROUND_TO_NEAREST_INT    0x00
pub const MM_FROUND_TO_NEAREST_INT: i32 = 0x00;
//#define _MM_FROUND_TO_NEG_INF        0x01
pub const MM_FROUND_TO_NEG_INF: i32 = 0x01;
//#define _MM_FROUND_TO_POS_INF        0x02
pub const MM_FROUND_TO_POS_INF: i32 = 0x02;
//#define _MM_FROUND_TO_ZERO           0x03
pub const MM_FROUND_TO_ZERO: i32 = 0x03;
//#define _MM_FROUND_CUR_DIRECTION     0x04
pub const MM_FROUND_CUR_DIRECTION: i32 = 0x04;

//#define _MM_FROUND_RAISE_EXC         0x00
pub const MM_FROUND_RAISE_EXC: i32 = 0x00;
//#define _MM_FROUND_NO_EXC            0x08
pub const MM_FROUND_NO_EXC: i32 = 0x08;

//#define _MM_FROUND_NINT      (_MM_FROUND_RAISE_EXC | _MM_FROUND_TO_NEAREST_INT)
pub const MM_FROUND_NINT: i32 = MM_FROUND_RAISE_EXC | MM_FROUND_TO_NEAREST_INT;
//#define _MM_FROUND_FLOOR     (_MM_FROUND_RAISE_EXC | _MM_FROUND_TO_NEG_INF)
pub const MM_FROUND_FLOOR: i32 = MM_FROUND_RAISE_EXC | MM_FROUND_TO_NEG_INF;
//#define _MM_FROUND_CEIL      (_MM_FROUND_RAISE_EXC | _MM_FROUND_TO_POS_INF)
pub const MM_FROUND_CEIL: i32 = MM_FROUND_RAISE_EXC | MM_FROUND_TO_POS_INF;
//#define _MM_FROUND_TRUNC     (_MM_FROUND_RAISE_EXC | _MM_FROUND_TO_ZERO)
pub const MM_FROUND_TRUNC: i32 = MM_FROUND_RAISE_EXC | MM_FROUND_TO_ZERO;
//#define _MM_FROUND_RINT      (_MM_FROUND_RAISE_EXC | _MM_FROUND_CUR_DIRECTION)
pub const MM_FROUND_RINT: i32 = MM_FROUND_RAISE_EXC | MM_FROUND_CUR_DIRECTION;
//#define _MM_FROUND_NEARBYINT (_MM_FROUND_NO_EXC | _MM_FROUND_CUR_DIRECTION)
pub const MM_FROUND_NEARBYINT: i32 = MM_FROUND_NO_EXC | MM_FROUND_CUR_DIRECTION;

// pblendw
// __m128i _mm_blend_epi16 (__m128i a, __m128i b, const int imm8)
#[inline]
pub fn mm_blend_epi16(a: m128i, b: m128i, imm8: i32) -> m128i {
    let x: i16x8 = blend_shuffle8!(a.as_i16x8(), b.as_i16x8(), imm8);
    x.as_m128i()
}

// blendpd
// __m128d _mm_blend_pd (__m128d a, __m128d b, const int imm8)
#[inline]
pub fn mm_blend_pd(a: m128d, b: m128d, imm8: i32) -> m128d {
    unsafe {
        match imm8 & 0x3 {
            0 => simd_shuffle2(a, b, [0, 1]),
            1 => simd_shuffle2(a, b, [2, 1]),
            2 => simd_shuffle2(a, b, [0, 3]),
            3 => simd_shuffle2(a, b, [2, 3]),
            _ => unreachable!()
        }
    }
}

// blendps
// __m128 _mm_blend_ps (__m128 a, __m128 b, const int imm8)
#[inline]
pub fn mm_blend_ps(a: m128, b: m128, imm8: i32) -> m128 {
    blend_shuffle4!(a, b, imm8)
}

// pblendvb
// __m128i _mm_blendv_epi8 (__m128i a, __m128i b, __m128i mask)
#[inline]
pub fn mm_blendv_epi8(a: m128i, b: m128i, mask: m128i) -> m128i {
    unsafe { sse41_pblendvb(a.as_i8x16(), b.as_i8x16(), mask.as_i8x16()).as_m128i() }
}

// blendvpd
// __m128d _mm_blendv_pd (__m128d a, __m128d b, __m128d mask)
#[inline]
pub fn mm_blendv_pd(a: m128d, b: m128d, mask: m128d) -> m128d {
    unsafe { sse41_blendvpd(a, b, mask) }
}

// blendvps
// __m128 _mm_blendv_ps (__m128 a, __m128 b, __m128 mask)
#[inline]
pub fn mm_blendv_ps(a: m128, b: m128, mask: m128) -> m128 {
    unsafe { sse41_blendvps(a, b, mask) }
}

// roundpd
// __m128d _mm_ceil_pd (__m128d a)
#[inline]
pub fn mm_ceil_pd(a: m128d) -> m128d {
    mm_round_pd(a, MM_FROUND_CEIL)
}

// roundps
// __m128 _mm_ceil_ps (__m128 a)
#[inline]
pub fn mm_ceil_ps(a: m128) -> m128 {
    mm_round_ps(a, MM_FROUND_CEIL)
}

// roundsd
// __m128d _mm_ceil_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_ceil_sd(a: m128d, b: m128d) -> m128d {
    mm_round_sd(a, b, MM_FROUND_CEIL)
}

// roundss
// __m128 _mm_ceil_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_ceil_ss(a: m128, b: m128) -> m128 {
    mm_round_ss(a, b, MM_FROUND_CEIL)
}

// pcmpeqq
// __m128i _mm_cmpeq_epi64 (__m128i a, __m128i b)
#[inline]
pub fn mm_cmpeq_epi64(a: m128i, b: m128i) -> m128i {
    let x: i64x2 = unsafe { simd_eq(a.as_i64x2(), b.as_i64x2()) };
    x.as_m128i()
}

// pmovsxwd
// __m128i _mm_cvtepi16_epi32 (__m128i a)
#[inline]
pub fn mm_cvtepi16_epi32(a: m128i) -> m128i {
    // return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v8hi)__V, (__v8hi)__V, 0, 1, 2, 3), __v4si);
    unsafe { sse41_pmovsxwd(a.as_i16x8()).as_m128i() }
}

// pmovsxwq
// __m128i _mm_cvtepi16_epi64 (__m128i a)
#[inline]
pub fn mm_cvtepi16_epi64(a: m128i) -> m128i {
    // return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v8hi)__V, (__v8hi)__V, 0, 1), __v2di);
    unsafe { sse41_pmovsxwq(a.as_i16x8()).as_m128i() }
}

// pmovsxdq
// __m128i _mm_cvtepi32_epi64 (__m128i a)
#[inline]
pub fn mm_cvtepi32_epi64(a: m128i) -> m128i {
    // return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v4si)__V, (__v4si)__V, 0, 1), __v2di);
    unsafe { sse41_pmovsxdq(a.as_i32x4()).as_m128i() }
}

// pmovsxbw
// __m128i _mm_cvtepi8_epi16 (__m128i a)
#[inline]
pub fn mm_cvtepi8_epi16(a: m128i) -> m128i {
    // return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v16qs)__V, (__v16qs)__V, 0, 1, 2, 3, 4, 5, 6, 7), __v8hi);
    unsafe { sse41_pmovsxbw(a.as_i8x16()).as_m128i() }
}

// pmovsxbd
// __m128i _mm_cvtepi8_epi32 (__m128i a)
#[inline]
pub fn mm_cvtepi8_epi32(a: m128i) -> m128i {
    // return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v16qs)__V, (__v16qs)__V, 0, 1, 2, 3), __v4si);
    unsafe { sse41_pmovsxbd(a.as_i8x16()).as_m128i() }
}

// pmovsxbq
// __m128i _mm_cvtepi8_epi64 (__m128i a)
#[inline]
pub fn mm_cvtepi8_epi64(a: m128i) -> m128i {
    // return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v16qs)__V, (__v16qs)__V, 0, 1), __v2di);
    unsafe { sse41_pmovsxbq(a.as_i8x16()).as_m128i() }
}

// pmovzxwd
// __m128i _mm_cvtepu16_epi32 (__m128i a)
#[inline]
pub fn mm_cvtepu16_epi32(a: m128i) -> m128i {
    // return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v8hu)__V, (__v8hu)__V, 0, 1, 2, 3), __v4si);
    unsafe { sse41_pmovzxwd(a.as_i16x8()).as_m128i() }
}

// pmovzxwq
// __m128i _mm_cvtepu16_epi64 (__m128i a)
#[inline]
pub fn mm_cvtepu16_epi64(a: m128i) -> m128i {
    // return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v8hu)__V, (__v8hu)__V, 0, 1), __v2di);
    unsafe { sse41_pmovzxwq(a.as_i16x8()).as_m128i() }
}

// pmovzxdq
// __m128i _mm_cvtepu32_epi64 (__m128i a)
#[inline]
pub fn mm_cvtepu32_epi64(a: m128i) -> m128i {
    // return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v4su)__V, (__v4su)__V, 0, 1), __v2di);
    unsafe { sse41_pmovzxdq(a.as_i32x4()).as_m128i() }
}

// pmovzxbw
// __m128i _mm_cvtepu8_epi16 (__m128i a)
#[inline]
pub fn mm_cvtepu8_epi16(a: m128i) -> m128i {
    // return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v16qu)__V, (__v16qu)__V, 0, 1, 2, 3, 4, 5, 6, 7), __v8hi);
    unsafe { sse41_pmovzxbw(a.as_i8x16()).as_m128i() }
}

// pmovzxbd
// __m128i _mm_cvtepu8_epi32 (__m128i a)
#[inline]
pub fn mm_cvtepu8_epi32(a: m128i) -> m128i {
    // return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v16qu)__V, (__v16qu)__V, 0, 1, 2, 3), __v4si);
    unsafe { sse41_pmovzxbd(a.as_i8x16()).as_m128i() }
}

// pmovzxbq
// __m128i _mm_cvtepu8_epi64 (__m128i a)
#[inline]
pub fn mm_cvtepu8_epi64(a: m128i) -> m128i {
    unsafe { sse41_pmovzxbq(a.as_i8x16()).as_m128i() }
}

// dppd
// __m128d _mm_dp_pd (__m128d a, __m128d b, const int imm8)
#[inline]
#[allow(unused_variables)]
pub fn mm_dp_pd(a: m128d, b: m128d, imm8: i32) -> m128d {
    fn_imm8_arg2!(x86_mm_dp_pd, a, b, imm8)
}

// dpps
// __m128 _mm_dp_ps (__m128 a, __m128 b, const int imm8)
#[inline]
#[allow(unused_variables)]
pub fn mm_dp_ps(a: m128, b: m128, imm8: i32) -> m128 {
    fn_imm8_arg2!(x86_mm_dp_ps, a, b, imm8)
}

// pextrd
// int _mm_extract_epi32 (__m128i a, const int imm8)
#[inline]
pub fn mm_extract_epi32(a: m128i, imm8: i32) -> i32 {
    a.as_i32x4().extract(imm8 as usize)
}

// pextrq
// __int64 _mm_extract_epi64 (__m128i a, const int imm8)
#[inline]
pub fn mm_extract_epi64(a: m128i, imm8: i32) -> i64 {
    a.as_i64x2().extract(imm8 as usize)
}

// pextrb
// int _mm_extract_epi8 (__m128i a, const int imm8)
#[inline]
pub fn mm_extract_epi8(a: m128i, imm8: i32) -> i32 {
    a.as_i8x16().extract(imm8 as usize) as i32
}

// extractps
// int _mm_extract_ps (__m128 a, const int imm8)
#[inline]
pub fn mm_extract_ps(a: m128, imm8: i32) -> i32 {
    a.as_m128i().as_i32x4().extract(imm8 as usize)
}

// roundpd
// __m128d _mm_floor_pd (__m128d a)
#[inline]
pub fn mm_floor_pd(a: m128d) -> m128d {
    mm_round_pd(a, MM_FROUND_FLOOR)
}

// roundps
// __m128 _mm_floor_ps (__m128 a)
#[inline]
pub fn mm_floor_ps(a: m128) -> m128 {
    mm_round_ps(a, MM_FROUND_FLOOR)
}

// roundsd
// __m128d _mm_floor_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_floor_sd(a: m128d, b: m128d) -> m128d {
    mm_round_sd(a, b, MM_FROUND_FLOOR)
}

// roundss
// __m128 _mm_floor_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_floor_ss(a: m128, b: m128) -> m128 {
    mm_round_ss(a, b, MM_FROUND_FLOOR)
}

// pinsrd
// __m128i _mm_insert_epi32 (__m128i a, int i, const int imm8)
#[inline]
pub fn mm_insert_epi32(a: m128i, i: i32, imm8: i32) -> m128i {
    a.as_i32x4().insert(imm8 as usize, i).as_m128i()
}

// pinsrq
// __m128i _mm_insert_epi64 (__m128i a, __int64 i, const int imm8)
#[inline]
pub fn mm_insert_epi64(a: m128i, i: i64, imm8: i32) -> m128i {
    a.as_i64x2().insert(imm8 as usize, i).as_m128i()
}

// pinsrb
// __m128i _mm_insert_epi8 (__m128i a, int i, const int imm8)
#[inline]
pub fn mm_insert_epi8(a: m128i, i: i32, imm8: i32) -> m128i {
    a.as_i8x16().insert(imm8 as usize, i as i8).as_m128i()
}

// insertps
// __m128 _mm_insert_ps (__m128 a, __m128 b, const int imm8)
#[inline]
pub fn mm_insert_ps(a: m128, b: m128, imm8: i32) -> m128 {
    fn_imm8_arg2!(sse41_insertps, a, b, imm8)
}

// pmaxsd
// __m128i _mm_max_epi32 (__m128i a, __m128i b)
#[inline]
pub fn mm_max_epi32(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_max_epi32(a.as_i32x4(), b.as_i32x4()).as_m128i() }
}

// pmaxsb
// __m128i _mm_max_epi8 (__m128i a, __m128i b)
#[inline]
pub fn mm_max_epi8(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_max_epi8(a.as_i8x16(), b.as_i8x16()).as_m128i() }
}

// pmaxuw
// __m128i _mm_max_epu16 (__m128i a, __m128i b)
#[inline]
pub fn mm_max_epu16(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_max_epu16(a.as_u16x8(), b.as_u16x8()).as_m128i() }
}

// pmaxud
// __m128i _mm_max_epu32 (__m128i a, __m128i b)
#[inline]
pub fn mm_max_epu32(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_max_epu32(a.as_u32x4(), b.as_u32x4()).as_m128i() }
}

// pminsd
// __m128i _mm_min_epi32 (__m128i a, __m128i b)
#[inline]
pub fn mm_min_epi32(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_min_epi32(a.as_i32x4(), b.as_i32x4()).as_m128i() }
}

// pminsb
// __m128i _mm_min_epi8 (__m128i a, __m128i b)
#[inline]
pub fn mm_min_epi8(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_min_epi8(a.as_i8x16(), b.as_i8x16()).as_m128i() }
}

// pminuw
// __m128i _mm_min_epu16 (__m128i a, __m128i b)
#[inline]
pub fn mm_min_epu16(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_min_epu16(a.as_u16x8(), b.as_u16x8()).as_m128i() }
}

// pminud
// __m128i _mm_min_epu32 (__m128i a, __m128i b)
#[inline]
pub fn mm_min_epu32(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_min_epu32(a.as_u32x4(), b.as_u32x4()).as_m128i() }
}

// phminposuw
// __m128i _mm_minpos_epu16 (__m128i a)
#[inline]
pub fn mm_minpos_epu16(a: m128i) -> m128i {
    unsafe { x86_mm_minpos_epu16(a.as_u16x8()).as_m128i() }
}

// mpsadbw
// __m128i _mm_mpsadbw_epu8 (__m128i a, __m128i b, const int imm8)
#[inline]
pub fn mm_mpsadbw_epu8(a: m128i, b: m128i, imm8: i32) -> m128i {
    fn_imm8_arg2!(x86_mm_mpsadbw_epu8, a.as_u8x16(), b.as_u8x16(), imm8).as_m128i()
}

// pmuldq
// __m128i _mm_mul_epi32 (__m128i a, __m128i b)
#[inline]
pub fn mm_mul_epi32(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_mul_epi32(a.as_i32x4(), b.as_i32x4()).as_m128i() }
}

// pmulld
// __m128i _mm_mullo_epi32 (__m128i a, __m128i b)
#[inline]
pub fn mm_mullo_epi32(a: m128i, b: m128i) -> m128i {
    unsafe { simd_mul(a.as_i32x4(), b.as_i32x4()).as_m128i() }
}

// packusdw
// __m128i _mm_packus_epi32 (__m128i a, __m128i b)
#[inline]
pub fn mm_packus_epi32(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_packus_epi32(a.as_i32x4(), b.as_i32x4()).as_m128i() }
}

// roundpd
// __m128d _mm_round_pd (__m128d a, int rounding)
#[inline]
pub fn mm_round_pd(a: m128d, rounding: i32) -> m128d {
    fn_imm8_arg1!(sse41_round_pd, a, rounding)
}

// roundps
// __m128 _mm_round_ps (__m128 a, int rounding)
#[inline]
pub fn mm_round_ps(a: m128, rounding: i32) -> m128 {
    fn_imm8_arg1!(sse41_round_ps, a, rounding)
}

// roundsd
// __m128d _mm_round_sd (__m128d a, __m128d b, int rounding)
#[inline]
pub fn mm_round_sd(a: m128d, b: m128d, rounding: i32) -> m128d {
    fn_imm8_arg2!(sse41_round_sd, a, b, rounding)
}

// roundss
// __m128 _mm_round_ss (__m128 a, __m128 b, int rounding)
#[inline]
pub fn mm_round_ss(a: m128, b: m128, rounding: i32) -> m128 {
    fn_imm8_arg2!(sse41_round_ss, a, b, rounding)
}

// movntdqa
// __m128i _mm_stream_load_si128 (__m128i* mem_addr)
#[inline]
#[allow(unused_variables)]
pub fn mm_stream_load_si128(mem_addr: *const m128i) -> m128i {
    // TODO(mayah): Make this
    unimplemented!()
}

// ...
// int _mm_test_all_ones (__m128i a)
#[inline]
pub fn mm_test_all_ones(a: m128i) -> i32 {
    mm_testc_si128(a, mm_cmpeq_epi32(a, a))
}

// ptest
// int _mm_test_all_zeros (__m128i a, __m128i mask)
#[inline]
pub fn mm_test_all_zeros(a: m128i, mask: m128i) -> i32 {
    mm_testz_si128(a, mask)
}

// ptest
// int _mm_test_mix_ones_zeros (__m128i a, __m128i mask)
#[inline]
pub fn mm_test_mix_ones_zeros(a: m128i, mask: m128i) -> i32 {
    mm_testnzc_si128(a, mask)
}

// ptest
// int _mm_testc_si128 (__m128i a, __m128i b)
#[inline]
pub fn mm_testc_si128(a: m128i, b: m128i) -> i32 {
    unsafe { sse41_ptestc(a.as_i64x2(), b.as_i64x2()) }
}

// ptest
// int _mm_testnzc_si128 (__m128i a, __m128i b)
#[inline]
pub fn mm_testnzc_si128(a: m128i, b: m128i) -> i32 {
    unsafe { sse41_ptestnzc(a.as_i64x2(), b.as_i64x2()) }
}

// ptest
// int _mm_testz_si128 (__m128i a, __m128i b)
#[inline]
pub fn mm_testz_si128(a: m128i, b: m128i) -> i32 {
    unsafe { sse41_ptestz(a.as_i64x2(), b.as_i64x2()) }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test_mm_testc_si128() {
        let x = mm_setr_epi32(0x7, 0x7, 0x7, 0x7);
        let y = mm_setr_epi32(0x3, 0x3, 0x3, 0x3);
        let z = mm_setr_epi32(0x8, 0x8, 0x8, 0x8);

        assert_eq!(mm_testc_si128(x, y), 1);
        assert_eq!(mm_testc_si128(x, z), 0);
    }

    #[test]
    fn test_mm_testnzc_si128() {
        {
            let a = mm_setr_epi32(0, 0, 0, 0);
            let b = mm_setr_epi32(!0, !0, 0, 0);
            assert_eq!(mm_testnzc_si128(a, b), 0);
        }

        {
            let a = mm_setr_epi32(1, 0, 0, 0);
            let b = mm_setr_epi32(!0, !0, 0, 0);
            assert_eq!(mm_testnzc_si128(a, b), 1);
        }
    }

    #[test]
    fn test_mm_testz_si128() {
        let x = mm_setr_epi32(0x0, 0x0, 0x0, 0x0);
        let y = mm_setr_epi32(0x1, 0x1, 0x1, 0x1);
        let z = mm_setr_epi32(0x2, 0x2, 0x2, 0x2);

        assert_eq!(mm_testz_si128(x, x), 1);
        assert_eq!(mm_testz_si128(y, y), 0);
        assert_eq!(mm_testz_si128(z, z), 0);
        assert_eq!(mm_testz_si128(y, z), 1);
    }

    #[test]
    fn test_mm_test() {
        let zero = mm_setzero_si128();
        let one = mm_setr_epi32(!0, !0, !0, !0);
        let mix = mm_setr_epi32(0, !0, 0, !0);

        assert_eq!(mm_test_all_zeros(zero, zero), 1);
        assert_eq!(mm_test_all_zeros(one, one), 0);
        assert_eq!(mm_test_all_zeros(mix, one), 0);

        assert_eq!(mm_test_all_ones(zero), 0);
        assert_eq!(mm_test_all_ones(one), 1);
        assert_eq!(mm_test_all_ones(mix), 0);

        assert_eq!(mm_test_mix_ones_zeros(zero, zero), 0);
        assert_eq!(mm_test_mix_ones_zeros(one, one), 0);
        assert_eq!(mm_test_mix_ones_zeros(mix, one), 1);
    }

    #[test]
    fn test_mm_blend() {
        let a = mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let b = mm_setr_epi16(11, 12, 13, 14, 15, 16, 17, 18);

        assert_eq!(mm_blend_epi16(a, b, 0).as_i16x8().as_array(), [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(mm_blend_epi16(a, b, 0xFF).as_i16x8().as_array(), [11, 12, 13, 14, 15, 16, 17, 18]);
        assert_eq!(mm_blend_epi16(a, b, 0x11).as_i16x8().as_array(), [11, 2, 3, 4, 15, 6, 7, 8]);

        let apd = mm_setr_pd(1.0, 2.0);
        let bpd = mm_setr_pd(3.0, 4.0);

        assert_eq!(mm_blend_pd(apd, bpd, 0).as_f64x2().as_array(), [1.0, 2.0]);
        assert_eq!(mm_blend_pd(apd, bpd, 1).as_f64x2().as_array(), [3.0, 2.0]);
        assert_eq!(mm_blend_pd(apd, bpd, 2).as_f64x2().as_array(), [1.0, 4.0]);
        assert_eq!(mm_blend_pd(apd, bpd, 3).as_f64x2().as_array(), [3.0, 4.0]);

        let aps = mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let bps = mm_setr_ps(5.0, 6.0, 7.0, 8.0);

        assert_eq!(mm_blend_ps(aps, bps, 0).as_f32x4().as_array(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(mm_blend_ps(aps, bps, 15).as_f32x4().as_array(), [5.0, 6.0, 7.0, 8.0]);
        assert_eq!(mm_blend_ps(aps, bps, 3).as_f32x4().as_array(), [5.0, 6.0, 3.0, 4.0]);
        assert_eq!(mm_blend_ps(aps, bps, 7).as_f32x4().as_array(), [5.0, 6.0, 7.0, 4.0]);
    }

    #[test]
    fn test_mm_blendv() {
        let a8 = mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b8 = mm_setr_epi8(101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116);
        let m8 = mm_setr_epi8(0, 0, 0, 0, !0, !0, !0, !0, 0, 0, 0, 0, !0, !0, !0, !0);

        assert_eq!(mm_blendv_epi8(a8, b8, m8).as_i8x16().as_array(),
                   [1, 2, 3, 4, 105, 106, 107, 108, 9, 10, 11, 12, 113, 114, 115, 116]);

        let apd = mm_setr_pd(1.0, 2.0);
        let bpd = mm_setr_pd(3.0, 4.0);
        let mpd = i64x2(0, !0).as_m128d();

        assert_eq!(mm_blendv_pd(apd, bpd, mpd).as_f64x2().as_array(), [1.0, 4.0]);

        let aps = mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let bps = mm_setr_ps(5.0, 6.0, 7.0, 8.0);
        let mps = i32x4(0, !0, 0, !0).as_m128();

        assert_eq!(mm_blendv_ps(aps, bps, mps).as_f32x4().as_array(), [1.0, 6.0, 3.0, 8.0]);
    }

    #[test]
    fn test_ceil_floor_round() {
        let ps = mm_setr_ps(1.5, 2.5, 3.5, 4.5);
        let pd = mm_setr_pd(1.5, 2.5);
        let ps1 = mm_set1_ps(6.0);
        let pd1 = mm_set1_pd(6.0);

        assert_eq!(mm_ceil_pd(pd).as_f64x2().as_array(), [2.0, 3.0]);
        assert_eq!(mm_floor_pd(pd).as_f64x2().as_array(), [1.0, 2.0]);

        assert_eq!(mm_ceil_ps(ps).as_f32x4().as_array(), [2.0, 3.0, 4.0, 5.0]);
        assert_eq!(mm_floor_ps(ps).as_f32x4().as_array(), [1.0, 2.0, 3.0, 4.0]);

        assert_eq!(mm_ceil_sd(pd1, pd).as_f64x2().as_array(), [2.0, 6.0]);
        assert_eq!(mm_floor_sd(pd1, pd).as_f64x2().as_array(), [1.0, 6.0]);

        assert_eq!(mm_ceil_ss(ps1, ps).as_f32x4().as_array(), [2.0, 6.0, 6.0, 6.0]);
        assert_eq!(mm_floor_ss(ps1, ps).as_f32x4().as_array(), [1.0, 6.0, 6.0, 6.0]);
    }

    #[test]
    fn test_cmpeq_epi64() {
        let x = i64x2(1, 1).as_m128i();
        let y = i64x2(0, 1).as_m128i();
        assert_eq!(mm_cmpeq_epi64(x, y).as_i64x2().as_array(), [0, !0]);
    }

    #[test]
    fn test_convert() {
        let x8 = mm_setr_epi8(1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16);
        let x16 = mm_setr_epi16(1, -2, 3, -4, 5, -6, 7, -8);
        let x32 = mm_setr_epi32(1, -2, 3, -4);

        assert_eq!(mm_cvtepi16_epi32(x16).as_i32x4().as_array(), [1, -2, 3, -4]);
        assert_eq!(mm_cvtepi16_epi64(x16).as_i64x2().as_array(), [1, -2]);
        assert_eq!(mm_cvtepi32_epi64(x32).as_i64x2().as_array(), [1, -2]);
        assert_eq!(mm_cvtepi8_epi16(x8).as_i16x8().as_array(), [1, -2, 3, -4, 5, -6, 7, -8]);
        assert_eq!(mm_cvtepi8_epi32(x8).as_i32x4().as_array(), [1, -2, 3, -4]);
        assert_eq!(mm_cvtepi8_epi64(x8).as_i64x2().as_array(), [1, -2]);

        assert_eq!(mm_cvtepu16_epi32(x16).as_i32x4().as_array(), [1, -2 & 0xFFFF, 3, -4 & 0xFFFF]);
        assert_eq!(mm_cvtepu16_epi64(x16).as_i64x2().as_array(), [1, -2 & 0xFFFF]);
        assert_eq!(mm_cvtepu32_epi64(x32).as_i64x2().as_array(), [1, -2 & 0xFFFFFFFF]);
        assert_eq!(mm_cvtepu8_epi16(x8).as_i16x8().as_array(), [1, -2 & 0xFF, 3, -4 & 0xFF, 5, -6 & 0xFF, 7, -8 & 0xFF]);
        assert_eq!(mm_cvtepu8_epi32(x8).as_i32x4().as_array(), [1, -2 & 0xFF, 3, -4 & 0xFF]);
        assert_eq!(mm_cvtepu8_epi64(x8).as_i64x2().as_array(), [1, -2 & 0xFF]);
    }

    #[test]
    fn test_mm_dp_pd() {
        let a = mm_setr_pd(1.5, 10.25);
        let b = mm_setr_pd(-1.5, 3.125);
        assert_eq!(mm_dp_pd(a, b, 0x31).as_f64x2().as_array(), [-1.5 * 1.5 + 10.25 * 3.125, 0.0]);
    }

    #[test]
    fn test_mm_dp_ps() {
        let a = mm_setr_ps(1.5, 10.25, -11.0625, 81.0);
        let b = mm_setr_ps(-1.5, 3.125, -50.5, 100.0);
        assert_eq!(mm_dp_ps(a, b, 0x55).as_f32x4().as_array(), [556.406250, 0.000000, 556.406250, 0.000000]);
    }

    #[test]
    fn test_insert_epi() {
        assert_eq!(mm_insert_epi32(mm_setr_epi32(0, 11, 2222, 333333), -65536, 2).as_i32x4().as_array(),
                   [0, 11, -65536, 333333]);
        assert_eq!(mm_insert_epi64(i64x2(500000, 3200000).as_m128i(), 4294901750, 0).as_i64x2().as_array(),
                   [4294901750, 3200000]);
        assert_eq!(mm_insert_epi8(mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), -32, 7).as_i8x16().as_array(),
                   [0, 1, 2, 3, 4, 5, 6, -32, 8, 9, 10, 11, 12, 13, 14, 15]);
    }

    #[test]
    fn test_insert_ps() {
        let a = mm_setr_ps(1.0, -1.0, 1.5, 105.5);
        let b = mm_setr_ps(-5.0, 10.0, -325.0625, 81.125);
        assert_eq!(mm_insert_ps(a, b, 0xD9).as_f32x4().as_array(),
                   [0.0, 81.125, 1.5, 0.0]);
    }

    #[test]
    fn test_extract() {
        assert_eq!(mm_extract_epi32(mm_setr_epi32(-1, 2, 3, 4), 0), -1);
        assert_eq!(mm_extract_epi64(i64x2(-1, 2).as_m128i(), 0), -1);
        assert_eq!(mm_extract_epi8(mm_setr_epi8(-1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), 0), -1);
        assert_eq!(mm_extract_ps(mm_setr_ps(1.25, -5.125, 16.0, 3.5), 1), 0xc0a40000u32 as i32);
    }

    #[test]
    fn test_maxmin() {
        let a8 = mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b8 = mm_setr_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a16 = mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let b16 = mm_setr_epi16(-1, -2, -3, -4, 5, 6, 7, 8);
        let a32 = mm_setr_epi32(1, 2, 3, 4);
        let b32 = mm_setr_epi32(-1, -2, 3, 4);

        assert_eq!(mm_max_epu16(a16, b16).as_u16x8().as_array(), [-1i16 as u16, -2i16 as u16, -3i16 as u16, -4i16 as u16, 5, 6, 7, 8]);
        assert_eq!(mm_min_epu16(a16, b16).as_u16x8().as_array(), [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(mm_max_epi32(a32, b32).as_i32x4().as_array(), [1, 2, 3, 4]);
        assert_eq!(mm_min_epi32(a32, b32).as_i32x4().as_array(), [-1, -2, 3, 4]);
        assert_eq!(mm_max_epu32(a32, b32).as_u32x4().as_array(), [-1i32 as u32, -2i32 as u32, 3, 4]);
        assert_eq!(mm_min_epu32(a32, b32).as_u32x4().as_array(), [1, 2, 3, 4]);
        assert_eq!(mm_max_epi8(a8, b8).as_i8x16().as_array(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        assert_eq!(mm_min_epi8(a8, b8).as_i8x16().as_array(), [-1, -2, -3, -4, -5, -6, -7, -8, 9, 10, 11, 12, 13, 14, 15, 16]);
    }

    #[test]
    fn test_minpos() {
        let a = mm_setr_epi16(127, 10000, -1, 1, 2, 64, 314, 27);
        assert_eq!(mm_minpos_epu16(a).as_u16x8().as_array(), [1, 3, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_mpsadbw() {
        let a = u8x16(15, 60, 55, 31, 0, 1, 2, 4, 8, 16, 32, 64, 128, 255, 1, 17).as_m128i();
        let b = u8x16(2, 4, 8, 64, 255, 0, 1, 16, 32, 64, 128, 255, 75, 31, 42, 11).as_m128i();
        assert_eq!(mm_mpsadbw_epu8(a, b, 5).as_u16x8().as_array(), [269, 267, 264, 290, 342, 446, 653, 588]);
    }

    #[test]
    fn test_mul() {
        let a = mm_setr_epi32(65000, 0, 24000000, 0);
        let b = mm_setr_epi32(-320000, 0, 56400000, 0);
        assert_eq!(mm_mul_epi32(a, b).as_i64x2().as_array(), [-20800000000, 1353600000000000]);

        let a = mm_setr_epi32(65535, -512, 77910, 0);
        let b = mm_setr_epi32(2, 4431, -7969, 240000000);
        assert_eq!(mm_mullo_epi32(a, b).as_i32x4().as_array(), [131070, -2268672, -620864790, 0]);
    }

    #[test]
    fn test_packus() {
        let a = mm_setr_epi32(0, -1, 70000, 128);
        let b = mm_setr_epi32(-512, 5200, 32768, 65536);
        assert_eq!(mm_packus_epi32(a, b).as_u16x8().as_array(), [0, 0, 65535, 128, 0, 5200, 32768, 65535]);
    }
}
