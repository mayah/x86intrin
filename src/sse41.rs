use super::*;
use super::{simd_eq,
            simd_shuffle2, simd_shuffle4, simd_shuffle8};

extern {
    #[link_name = "llvm.x86.sse41.pblendvb"]
    fn sse41_pblendvb(a: i8x16, b: i8x16, c: i8x16) -> i8x16;
    #[link_name = "llvm.x86.sse41.blendvpd"]
    fn sse41_blendvpd(a: m128d, b: m128d, c: m128d) -> m128d;
    #[link_name = "llvm.x86.sse41.blendvps"]
    fn sse41_blendvps(a: m128, b: m128, c: m128) -> m128;

    // TODO(mayah): These functions are not published yet?
    // #[link_name = "llvm.x86.sse41.round.ss"]
    // fn sse41_round_ss(a: m128, b: m128, c: i32) -> m128;
    // #[link_name = "llvm.x86.sse41.round.ps"]
    // fn sse41_round_ps(a: m128, b: i32) -> m128;
    // #[link_name = "llvm.x86.sse41.round.sd"]
    // fn sse41_round_sd(a: m128d, b: m128d, c: i32) -> m128d;
    // #[link_name = "llvm.x86.sse41.round.pd"]
    // fn sse41_round_pd(a: m128d, b: i32) -> m128d;

    #[link_name = "llvm.x86.sse41.ptestc"]
    fn sse41_ptestc(a: i64x2, b: i64x2) -> i32;
    #[link_name = "llvm.x86.sse41.ptestnzc"]
    fn sse41_ptestnzc(a: i64x2, b: i64x2) -> i32;
    #[link_name = "llvm.x86.sse41.ptestz"]
    fn sse41_ptestz(a: i64x2, b: i64x2) -> i32;
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
    macro_rules! shuffle4 {
        ($e1: expr, $e2: expr, $e3: expr, $e4: expr, $e5: expr, $e6: expr, $e7: expr, $e8: expr) => {
            unsafe {
                let x: i16x8 = simd_shuffle8(a.as_i16x8(), b.as_i16x8(),
                                             [$e1, $e2, $e3, $e4, $e5, $e6, $e7, $e8]);
                x.as_m128i()
            }
        }
    }
    macro_rules! shuffle3 {
        ($e1: expr, $e2: expr, $e3: expr, $e4: expr, $e5: expr, $e6: expr) => {
            match (imm8 >> 6) & 3 {
                0 => shuffle4!($e1, $e2, $e3, $e4, $e5, $e6,  6,  7),
                1 => shuffle4!($e1, $e2, $e3, $e4, $e5, $e6, 14,  7),
                2 => shuffle4!($e1, $e2, $e3, $e4, $e5, $e6,  6, 15),
                3 => shuffle4!($e1, $e2, $e3, $e4, $e5, $e6, 14, 15),
                _ => unreachable!()
            }
        }
    }
    macro_rules! shuffle2 {
        ($e1: expr, $e2: expr, $e3: expr, $e4: expr) => {
            match (imm8 >> 4) & 3 {
                0 => shuffle3!($e1, $e2, $e3, $e4,  4,  5),
                1 => shuffle3!($e1, $e2, $e3, $e4, 12,  5),
                2 => shuffle3!($e1, $e2, $e3, $e4,  4, 13),
                3 => shuffle3!($e1, $e2, $e3, $e4, 12, 13),
                _ => unreachable!()
            }
        }
    }
    macro_rules! shuffle1 {
        ($e1: expr, $e2: expr) => {
            match (imm8 >> 2) & 0x3 {
                0 => shuffle2!($e1, $e2,  2,  3),
                1 => shuffle2!($e1, $e2, 10, 11),
                2 => shuffle2!($e1, $e2,  2,  3),
                3 => shuffle2!($e1, $e2, 10, 11),
                _ => unreachable!()
            }
        }
    }
    macro_rules! shuffle0 {
        () => {
            match (imm8 >> 0) & 0x3 {
                0 => shuffle1!(0, 1),
                1 => shuffle1!(8, 1),
                2 => shuffle1!(0, 9),
                3 => shuffle1!(8, 9),
                _ => unreachable!()
            }
        }
    }

    shuffle0!()
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
    unsafe {
        match imm8 & 0xF {
            0x0 => simd_shuffle4(a, b, [0, 1, 2, 3]),
            0x1 => simd_shuffle4(a, b, [4, 1, 2, 3]),
            0x2 => simd_shuffle4(a, b, [0, 5, 2, 3]),
            0x3 => simd_shuffle4(a, b, [4, 5, 2, 3]),
            0x4 => simd_shuffle4(a, b, [0, 1, 6, 3]),
            0x5 => simd_shuffle4(a, b, [4, 1, 6, 3]),
            0x6 => simd_shuffle4(a, b, [0, 5, 6, 3]),
            0x7 => simd_shuffle4(a, b, [4, 5, 6, 3]),
            0x8 => simd_shuffle4(a, b, [0, 1, 2, 7]),
            0x9 => simd_shuffle4(a, b, [4, 1, 2, 7]),
            0xA => simd_shuffle4(a, b, [0, 5, 2, 7]),
            0xB => simd_shuffle4(a, b, [4, 5, 2, 7]),
            0xC => simd_shuffle4(a, b, [0, 1, 6, 7]),
            0xD => simd_shuffle4(a, b, [4, 1, 6, 7]),
            0xE => simd_shuffle4(a, b, [0, 5, 6, 7]),
            0xF => simd_shuffle4(a, b, [4, 5, 6, 7]),
            _ => unreachable!()
        }
    }
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

// TODO(mayah): Hard to implement these?
// pmovsxwd
// __m128i _mm_cvtepi16_epi32 (__m128i a)
#[inline]
#[allow(unused_variables)]
pub fn mm_cvtepi16_epi32(a: m128i) -> m128i {
//    let x: i32x4 = unsafe { simd_shuffle4(a.as_i16x8(), a.as_i16x8(), [0, 1, 2, 3]) };
//    let y: i32x4 = unsafe { simd_cast(x) };
//    y.as_m128i()
    unimplemented!()
}

// pmovsxwq
// __m128i _mm_cvtepi16_epi64 (__m128i a)
#[inline]
#[allow(unused_variables)]
pub fn mm_cvtepi16_epi64(a: m128i) -> m128i {
    unimplemented!()
    // return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v8hi)__V, (__v8hi)__V, 0, 1), __v2di);
}

// pmovsxdq
// __m128i _mm_cvtepi32_epi64 (__m128i a)
#[inline]
#[allow(unused_variables)]
pub fn mm_cvtepi32_epi64(a: m128i) -> m128i {
    unimplemented!()
    // return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v4si)__V, (__v4si)__V, 0, 1), __v2di);
}

// pmovsxbw
// __m128i _mm_cvtepi8_epi16 (__m128i a)
#[inline]
#[allow(unused_variables)]
pub fn mm_cvtepi8_epi16(a: m128i) -> m128i {
    unimplemented!()
      /* This function always performs a signed extension, but __v16qi is a char
    285      which may be signed or unsigned, so use __v16qs. */
   // 286   return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v16qs)__V, (__v16qs)__V, 0, 1, 2, 3, 4, 5, 6, 7), __v8hi);
     // 287
}

// pmovsxbd
// __m128i _mm_cvtepi8_epi32 (__m128i a)
#[inline]
#[allow(unused_variables)]
pub fn mm_cvtepi8_epi32(a: m128i) -> m128i {
    unimplemented!()
      /* This function always performs a signed extension, but __v16qi is a char
    293      which may be signed or unsigned, so use __v16qs. */
//    294   return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v16qs)__V, (__v16qs)__V, 0, 1, 2, 3), __v4si);
//      295
}

// pmovsxbq
// __m128i _mm_cvtepi8_epi64 (__m128i a)
#[inline]
#[allow(unused_variables)]
pub fn mm_cvtepi8_epi64(a: m128i) -> m128i {
    unimplemented!()
      /* This function always performs a signed extension, but __v16qi is a char
    301      which may be signed or unsigned, so use __v16qs. */
//    302   return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v16qs)__V, (__v16qs)__V, 0, 1), __v2di);
//      303
}

// pmovzxwd
// __m128i _mm_cvtepu16_epi32 (__m128i a)
#[inline]
#[allow(unused_variables)]
pub fn mm_cvtepu16_epi32(a: m128i) -> m128i {
    unimplemented!()
//    return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v8hu)__V, (__v8hu)__V, 0, 1, 2, 3), __v4si);
//      346
}

// pmovzxwq
// __m128i _mm_cvtepu16_epi64 (__m128i a)
#[inline]
#[allow(unused_variables)]
pub fn mm_cvtepu16_epi64(a: m128i) -> m128i {
    unimplemented!()
//    return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v8hu)__V, (__v8hu)__V, 0, 1), __v2di);
//      352
}

// pmovzxdq
// __m128i _mm_cvtepu32_epi64 (__m128i a)
#[inline]
#[allow(unused_variables)]
pub fn mm_cvtepu32_epi64(a: m128i) -> m128i {
    unimplemented!()

        //return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v4su)__V, (__v4su)__V, 0, 1), __v2di);
//      358
}

// pmovzxbw
// __m128i _mm_cvtepu8_epi16 (__m128i a)
#[inline]
#[allow(unused_variables)]
pub fn mm_cvtepu8_epi16(a: m128i) -> m128i {
    unimplemented!()

        //return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v16qu)__V, (__v16qu)__V, 0, 1, 2, 3, 4, 5, 6, 7), __v8hi);
//      328
}

// pmovzxbd
// __m128i _mm_cvtepu8_epi32 (__m128i a)
#[inline]
#[allow(unused_variables)]
pub fn mm_cvtepu8_epi32(a: m128i) -> m128i {
    unimplemented!()

        //return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v16qu)__V, (__v16qu)__V, 0, 1, 2, 3), __v4si);
//    334 }
}

// pmovzxbq
// __m128i _mm_cvtepu8_epi64 (__m128i a)
#[inline]
#[allow(unused_variables)]
pub fn mm_cvtepu8_epi64(a: m128i) -> m128i {
    unimplemented!()
//    return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v16qu)__V, (__v16qu)__V, 0, 1), __v2di);
//  340
}

// dppd
// __m128d _mm_dp_pd (__m128d a, __m128d b, const int imm8)
// dpps
// __m128 _mm_dp_ps (__m128 a, __m128 b, const int imm8)
// pextrd
// int _mm_extract_epi32 (__m128i a, const int imm8)
// pextrq
// __int64 _mm_extract_epi64 (__m128i a, const int imm8)
// pextrb
// int _mm_extract_epi8 (__m128i a, const int imm8)
// extractps
// int _mm_extract_ps (__m128 a, const int imm8)

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
// pinsrq
// __m128i _mm_insert_epi64 (__m128i a, __int64 i, const int imm8)
// pinsrb
// __m128i _mm_insert_epi8 (__m128i a, int i, const int imm8)
// insertps
// __m128 _mm_insert_ps (__m128 a, __m128 b, const int imm8)
// pmaxsd
// __m128i _mm_max_epi32 (__m128i a, __m128i b)
// pmaxsb
// __m128i _mm_max_epi8 (__m128i a, __m128i b)
// pmaxuw
// __m128i _mm_max_epu16 (__m128i a, __m128i b)
// pmaxud
// __m128i _mm_max_epu32 (__m128i a, __m128i b)
// pminsd
// __m128i _mm_min_epi32 (__m128i a, __m128i b)
// pminsb
// __m128i _mm_min_epi8 (__m128i a, __m128i b)
// pminuw
// __m128i _mm_min_epu16 (__m128i a, __m128i b)
// pminud
// __m128i _mm_min_epu32 (__m128i a, __m128i b)
// phminposuw
// __m128i _mm_minpos_epu16 (__m128i a)
// mpsadbw
// __m128i _mm_mpsadbw_epu8 (__m128i a, __m128i b, const int imm8)
// pmuldq
// __m128i _mm_mul_epi32 (__m128i a, __m128i b)
// pmulld
// __m128i _mm_mullo_epi32 (__m128i a, __m128i b)
// packusdw
// __m128i _mm_packus_epi32 (__m128i a, __m128i b)

// roundpd
// __m128d _mm_round_pd (__m128d a, int rounding)
// TODO(mayah): Implement this in librustc_platform_intrinsics.
#[inline]
#[allow(unused_variables)]
pub fn mm_round_pd(a: m128d, rounding: i32) -> m128d {
    // unsafe { sse41_round_pd(a, rounding) }
    unimplemented!()
}

// roundps
// __m128 _mm_round_ps (__m128 a, int rounding)
// TODO(mayah): Implement this in librustc_platform_intrinsics.
#[inline]
#[allow(unused_variables)]
pub fn mm_round_ps(a: m128, rounding: i32) -> m128 {
    // unsafe { sse41_round_ps(a, rounding) }
    unimplemented!()
}

// roundsd
// __m128d _mm_round_sd (__m128d a, __m128d b, int rounding)
// TODO(mayah): Implement this in librustc_platform_intrinsics.
#[inline]
#[allow(unused_variables)]
pub fn mm_round_sd(a: m128d, b: m128d, rounding: i32) -> m128d {
    // unsafe { sse41_round_sd(a, b, rounding) }
    unimplemented!()
}

// roundss
// __m128 _mm_round_ss (__m128 a, __m128 b, int rounding)
// TODO(mayah): Implement this in librustc_platform_intrinsics.
#[inline]
#[allow(unused_variables)]
pub fn mm_round_ss(a: m128, b: m128, rounding: i32) -> m128 {
    // unsafe { sse41_round_ss(a, b, rounding) }
    unimplemented!()
}

// movntdqa
// __m128i _mm_stream_load_si128 (__m128i* mem_addr)

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
        // TODO(mayah): Since mm_round_* are not published, these won't work.

        // let ps = mm_setr_ps(1.5, 2.5, 3.5, 4.5);
        // let pd = mm_setr_pd(1.5, 2.5);
        // let ps1 = mm_set1_ps(6.0);
        // let pd1 = mm_set1_pd(6.0);

        // assert_eq!(mm_ceil_pd(pd).as_f64x2().as_array(), [2.0, 3.0]);
        // assert_eq!(mm_floor_pd(pd).as_f64x2().as_array(), [1.0, 2.0]);

        // assert_eq!(mm_ceil_ps(ps).as_f32x4().as_array(), [2.0, 3.0, 4.0, 5.0]);
        // assert_eq!(mm_floor_ps(ps).as_f32x4().as_array(), [1.0, 2.0, 3.0, 4.0]);

        // assert_eq!(mm_ceil_sd(pd1, pd).as_f64x2().as_array(), [2.0, 6.0]);
        // assert_eq!(mm_floor_sd(pd1, pd).as_f64x2().as_array(), [1.0, 6.0]);

        // assert_eq!(mm_ceil_ss(ps1, ps).as_f32x4().as_array(), [2.0, 6.0, 6.0, 6.0]);
        // assert_eq!(mm_floor_ss(ps1, ps).as_f32x4().as_array(), [1.0, 6.0, 6.0, 6.0]);
    }

    #[test]
    fn test_cmpeq_epi64() {
        let x = i64x2(1, 1).as_m128i();
        let y = i64x2(0, 1).as_m128i();
        assert_eq!(mm_cmpeq_epi64(x, y).as_i64x2().as_array(), [0, !0]);
    }

    #[test]
    fn test_convert() {
        // TODO(mayah): Currently hard to implement these intrinsics.

        // let x8 = mm_setr_epi8(1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16);
        // let x16 = mm_setr_epi16(1, -2, 3, -4, 5, -6, 7, -8);
        // let x32 = mm_setr_epi32(1, -2, 3, -4);

        // assert_eq!(mm_cvtepi16_epi32(x16).as_i32x4().as_array(), [1, -2, 3, -4]);
        // assert_eq!(mm_cvtepi16_epi64(x16).as_i64x2().as_array(), [1, -2]);
        // assert_eq!(mm_cvtepi32_epi64(x32).as_i64x2().as_array(), [1, -2]);
        // assert_eq!(mm_cvtepi8_epi16(x8).as_i16x8().as_array(), [1, -2, 3, -4, 5, -6, 7, -8]);
        // assert_eq!(mm_cvtepi8_epi32(x8).as_i32x4().as_array(), [1, -2, 3, -4]);
        // assert_eq!(mm_cvtepi8_epi64(x8).as_i64x2().as_array(), [1, -2]);

        // assert_eq!(mm_cvtepu16_epi32(x16).as_i32x4().as_array(), [1, -2 & 0xFFFF, 3, -4 & 0xFFFF]);
        // assert_eq!(mm_cvtepu16_epi64(x16).as_i64x2().as_array(), [1, -2 & 0xFFFF]);
        // assert_eq!(mm_cvtepu32_epi64(x32).as_i64x2().as_array(), [1, -2 & 0xFFFFFFFF]);
        // assert_eq!(mm_cvtepu8_epi16(x8).as_i16x8().as_array(), [1, -2 & 0xFF, 3, -4 & 0xFF, 5, -6 & 0xFF, 7, -8 & 0xFF]);
        // assert_eq!(mm_cvtepu8_epi32(x8).as_i32x4().as_array(), [1, -2 & 0xFF, 3, -4 & 0xFF]);
        // assert_eq!(mm_cvtepu8_epi64(x8).as_i64x2().as_array(), [1, -2 & 0xFF]);
    }
}
