use super::*;
use super::{simd_shuffle8};

extern {
    #[link_name = "llvm.x86.sse41.ptestc"]
    fn sse41_ptestc(a: i64x2, b: i64x2) -> i32;
    #[link_name = "llvm.x86.sse41.ptestnzc"]
    fn sse41_ptestnzc(a: i64x2, b: i64x2) -> i32;
    #[link_name = "llvm.x86.sse41.ptestz"]
    fn sse41_ptestz(a: i64x2, b: i64x2) -> i32;
}

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
// blendps
// __m128 _mm_blend_ps (__m128 a, __m128 b, const int imm8)
// pblendvb
// __m128i _mm_blendv_epi8 (__m128i a, __m128i b, __m128i mask)
// blendvpd
// __m128d _mm_blendv_pd (__m128d a, __m128d b, __m128d mask)
// blendvps
// __m128 _mm_blendv_ps (__m128 a, __m128 b, __m128 mask)
// roundpd
// __m128d _mm_ceil_pd (__m128d a)
// roundps
// __m128 _mm_ceil_ps (__m128 a)
// roundsd
// __m128d _mm_ceil_sd (__m128d a, __m128d b)
// roundss
// __m128 _mm_ceil_ss (__m128 a, __m128 b)
// pcmpeqq
// __m128i _mm_cmpeq_epi64 (__m128i a, __m128i b)
// pmovsxwd
// __m128i _mm_cvtepi16_epi32 (__m128i a)
// pmovsxwq
// __m128i _mm_cvtepi16_epi64 (__m128i a)
// pmovsxdq
// __m128i _mm_cvtepi32_epi64 (__m128i a)
// pmovsxbw
// __m128i _mm_cvtepi8_epi16 (__m128i a)
// pmovsxbd
// __m128i _mm_cvtepi8_epi32 (__m128i a)
// pmovsxbq
// __m128i _mm_cvtepi8_epi64 (__m128i a)
// pmovzxwd
// __m128i _mm_cvtepu16_epi32 (__m128i a)
// pmovzxwq
// __m128i _mm_cvtepu16_epi64 (__m128i a)
// pmovzxdq
// __m128i _mm_cvtepu32_epi64 (__m128i a)
// pmovzxbw
// __m128i _mm_cvtepu8_epi16 (__m128i a)
// pmovzxbd
// __m128i _mm_cvtepu8_epi32 (__m128i a)
// pmovzxbq
// __m128i _mm_cvtepu8_epi64 (__m128i a)
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
// roundps
// __m128 _mm_floor_ps (__m128 a)
// roundsd
// __m128d _mm_floor_sd (__m128d a, __m128d b)
// roundss
// __m128 _mm_floor_ss (__m128 a, __m128 b)
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
// roundps
// __m128 _mm_round_ps (__m128 a, int rounding)
// roundsd
// __m128d _mm_round_sd (__m128d a, __m128d b, int rounding)
// roundss
// __m128 _mm_round_ss (__m128 a, __m128 b, int rounding)
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
    }
}
