use super::*;
use super::simd_shuffle16;

extern "platform-intrinsic" {
    fn x86_mm_abs_epi8(x: i8x16) -> i8x16;
    fn x86_mm_abs_epi16(x: i16x8) -> i16x8;
    fn x86_mm_abs_epi32(x: i32x4) -> i32x4;
    fn x86_mm_hadd_epi16(x: i16x8, y: i16x8) -> i16x8;
    fn x86_mm_hadd_epi32(x: i32x4, y: i32x4) -> i32x4;
    fn x86_mm_hadds_epi16(x: i16x8, y: i16x8) -> i16x8;
    fn x86_mm_hsub_epi16(x: i16x8, y: i16x8) -> i16x8;
    fn x86_mm_hsub_epi32(x: i32x4, y: i32x4) -> i32x4;
    fn x86_mm_hsubs_epi16(x: i16x8, y: i16x8) -> i16x8;
    fn x86_mm_maddubs_epi16(x: u8x16, y: i8x16) -> i16x8;
    fn x86_mm_mulhrs_epi16(x: i16x8, y: i16x8) -> i16x8;
    fn x86_mm_shuffle_epi8(x: i8x16, y: i8x16) -> i8x16;
    fn x86_mm_sign_epi8(x: i8x16, y: i8x16) -> i8x16;
    fn x86_mm_sign_epi16(x: i16x8, y: i16x8) -> i16x8;
    fn x86_mm_sign_epi32(x: i32x4, y: i32x4) -> i32x4;
}

// pabsw
// __m128i _mm_abs_epi16 (__m128i a)
#[inline]
pub fn mm_abs_epi16(a: m128i) -> m128i {
    unsafe { x86_mm_abs_epi16(a.as_i16x8()).as_m128i() }
}

// pabsd
// __m128i _mm_abs_epi32 (__m128i a)
#[inline]
pub fn mm_abs_epi32(a: m128i) -> m128i {
    unsafe { x86_mm_abs_epi32(a.as_i32x4()).as_m128i() }
}

// pabsb
// __m128i _mm_abs_epi8 (__m128i a)
#[inline]
pub fn mm_abs_epi8(a: m128i) -> m128i {
    unsafe { x86_mm_abs_epi8(a.as_i8x16()).as_m128i() }
}

// palignr
// __m128i _mm_alignr_epi8 (__m128i a, __m128i b, int count)
#[inline]
pub fn mm_alignr_epi8(a: m128i, b: m128i, count: i32) -> m128i {
    let ai = a.as_i8x16();
    let bi = b.as_i8x16();
    let zi = mm_setzero_si128().as_i8x16();
    unsafe {
        let c: i8x16 = match count {
            0 => simd_shuffle16(ai, bi, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
            1 => simd_shuffle16(ai, bi, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            2 => simd_shuffle16(ai, bi, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
            3 => simd_shuffle16(ai, bi, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
            4 => simd_shuffle16(ai, bi, [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
            5 => simd_shuffle16(ai, bi, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
            6 => simd_shuffle16(ai, bi, [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]),
            7 => simd_shuffle16(ai, bi, [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]),
            8 => simd_shuffle16(ai, bi, [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]),
            9 => simd_shuffle16(ai, bi, [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]),
            10 => simd_shuffle16(ai, bi, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]),
            11 => simd_shuffle16(ai, bi, [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]),
            12 => simd_shuffle16(ai, bi, [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]),
            13 => simd_shuffle16(ai, bi, [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]),
            14 => simd_shuffle16(ai, bi, [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]),
            15 => simd_shuffle16(ai, bi, [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]),
            16 => simd_shuffle16(bi, zi, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
            17 => simd_shuffle16(bi, zi, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            18 => simd_shuffle16(bi, zi, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
            19 => simd_shuffle16(bi, zi, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
            20 => simd_shuffle16(bi, zi, [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
            21 => simd_shuffle16(bi, zi, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
            22 => simd_shuffle16(bi, zi, [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]),
            23 => simd_shuffle16(bi, zi, [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]),
            24 => simd_shuffle16(bi, zi, [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]),
            25 => simd_shuffle16(bi, zi, [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]),
            26 => simd_shuffle16(bi, zi, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]),
            27 => simd_shuffle16(bi, zi, [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]),
            28 => simd_shuffle16(bi, zi, [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]),
            29 => simd_shuffle16(bi, zi, [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]),
            30 => simd_shuffle16(bi, zi, [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]),
            31 => simd_shuffle16(bi, zi, [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]),
            _ => zi,
        };
        c.as_m128i()
    }
}

// phaddw
// __m128i _mm_hadd_epi16 (__m128i a, __m128i b)
#[inline]
pub fn mm_hadd_epi16(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_hadd_epi16(a.as_i16x8(), b.as_i16x8()).as_m128i() }
}

// phaddd
// __m128i _mm_hadd_epi32 (__m128i a, __m128i b)
#[inline]
pub fn mm_hadd_epi32(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_hadd_epi32(a.as_i32x4(), b.as_i32x4()).as_m128i() }
}

// phaddsw
// __m128i _mm_hadds_epi16 (__m128i a, __m128i b)
#[inline]
pub fn mm_hadds_epi16(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_hadds_epi16(a.as_i16x8(), b.as_i16x8()).as_m128i() }
}

// phsubw
// __m128i _mm_hsub_epi16 (__m128i a, __m128i b)
#[inline]
pub fn mm_hsub_epi16(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_hsub_epi16(a.as_i16x8(), b.as_i16x8()).as_m128i() }
}

// phsubd
// __m128i _mm_hsub_epi32 (__m128i a, __m128i b)
#[inline]
pub fn mm_hsub_epi32(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_hsub_epi32(a.as_i32x4(), b.as_i32x4()).as_m128i() }
}

// phsubsw
// __m128i _mm_hsubs_epi16 (__m128i a, __m128i b)
#[inline]
pub fn mm_hsubs_epi16(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_hsubs_epi16(a.as_i16x8(), b.as_i16x8()).as_m128i() }
}

// pmaddubsw
// __m128i _mm_maddubs_epi16 (__m128i a, __m128i b)
#[inline]
pub fn mm_maddubs_epi16(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_maddubs_epi16(a.as_u8x16(), b.as_i8x16()).as_m128i() }
}

// pmulhrsw
// __m128i _mm_mulhrs_epi16 (__m128i a, __m128i b)
#[inline]
pub fn mm_mulhrs_epi16(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_mulhrs_epi16(a.as_i16x8(), b.as_i16x8()).as_m128i() }
}

// pshufb
// __m128i _mm_shuffle_epi8 (__m128i a, __m128i b)
#[inline]
pub fn mm_shuffle_epi8(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_shuffle_epi8(a.as_i8x16(), b.as_i8x16()).as_m128i() }
}

// psignw
// __m128i _mm_sign_epi16 (__m128i a, __m128i b)
#[inline]
pub fn mm_sign_epi16(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_sign_epi16(a.as_i16x8(), b.as_i16x8()).as_m128i() }
}

// psignd
// __m128i _mm_sign_epi32 (__m128i a, __m128i b)
#[inline]
pub fn mm_sign_epi32(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_sign_epi32(a.as_i32x4(), b.as_i32x4()).as_m128i() }
}

// psignb
// __m128i _mm_sign_epi8 (__m128i a, __m128i b)
#[inline]
pub fn mm_sign_epi8(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_sign_epi8(a.as_i8x16(), b.as_i8x16()).as_m128i() }
}

// pabsw
// __m64 _mm_abs_pi16 (__m64 a)
// pabsd
// __m64 _mm_abs_pi32 (__m64 a)
// pabsb
// __m64 _mm_abs_pi8 (__m64 a)
// palignr
// __m64 _mm_alignr_pi8 (__m64 a, __m64 b, int count)
// phaddw
// __m64 _mm_hadd_pi16 (__m64 a, __m64 b)
// phaddw
// __m64 _mm_hadd_pi32 (__m64 a, __m64 b)
// phaddsw
// __m64 _mm_hadds_pi16 (__m64 a, __m64 b)
// phsubw
// __m64 _mm_hsub_pi16 (__m64 a, __m64 b)
// phsubd
// __m64 _mm_hsub_pi32 (__m64 a, __m64 b)
// phsubsw
// __m64 _mm_hsubs_pi16 (__m64 a, __m64 b)
// pmaddubsw
// __m64 _mm_maddubs_pi16 (__m64 a, __m64 b)
// pmulhrsw
// __m64 _mm_mulhrs_pi16 (__m64 a, __m64 b)
// pshufb
// __m64 _mm_shuffle_pi8 (__m64 a, __m64 b)
// psignw
// __m64 _mm_sign_pi16 (__m64 a, __m64 b)
// psignd
// __m64 _mm_sign_pi32 (__m64 a, __m64 b)
// psignb
// __m64 _mm_sign_pi8 (__m64 a, __m64 b)

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test_abs() {
        let x8 = mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -0x80);
        let x16 = mm_setr_epi16(1, 2, 3, 4, -1, -2, -3, -0x8000);
        let x32 = mm_setr_epi32(1, 2, -1, -0x80000000);

        assert_eq!(mm_abs_epi8(x8).as_i8x16().as_array(), [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, -0x80]);
        assert_eq!(mm_abs_epi16(x16).as_i16x8().as_array(), [1, 2, 3, 4, 1, 2, 3, -0x8000]);
        assert_eq!(mm_abs_epi32(x32).as_i32x4().as_array(), [1, 2, 1, -0x80000000]);
    }

    #[test]
    fn test_hadd() {
        let x16 = mm_setr_epi16(1, 2, 0x7000, 0x7000, -1, -2, -0x7000, -0x7000);
        let x32 = mm_setr_epi32(1, 2, -1, -2);

        // 0x7000 + 0x7000 = 0xE000
        let e = 0xE000u16 as i16;
        assert_eq!(mm_hadd_epi16(x16, x16).as_i16x8().as_array(), [3, e, -3, -e, 3, e, -3, -e]);
        assert_eq!(mm_hadd_epi32(x32, x32).as_i32x4().as_array(), [3, -3, 3, -3]);
        assert_eq!(mm_hadds_epi16(x16, x16).as_i16x8().as_array(), [3, 0x7FFF, -3, -0x8000, 3, 0x7FFF, -3, -0x8000]);
    }

    #[test]
    fn test_hsub() {
        let x16 = mm_setr_epi16(1, 2, 0x7000, -0x7000, -1, -2, -0x7000, 0x7000);
        let x32 = mm_setr_epi32(1, 2, -1, -2);

        // 0x7000 + 0x7000 = 0xE000
        let e = 0xE000u16 as i16;
        assert_eq!(mm_hsub_epi16(x16, x16).as_i16x8().as_array(), [-1, e, 1, -e, -1, e, 1, -e]);
        assert_eq!(mm_hsub_epi32(x32, x32).as_i32x4().as_array(), [-1, 1, -1, 1]);
        assert_eq!(mm_hsubs_epi16(x16, x16).as_i16x8().as_array(), [-1, 0x7FFF, 1, -0x8000, -1, 0x7FFF, 1, -0x8000]);
    }

    #[test]
    fn test_maddubs() {
        let xi8 = mm_setr_epi8(1, -2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 127, 127);
        let xu8 = mm_setr_epi8(1, -2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, -1, -1);

        assert_eq!(mm_maddubs_epi16(xi8, xu8).as_i16x8().as_array(),
                   [1 * 1 + -2 * 254,
                    3 * 3 + 4 * 4,
                    5 * 5 + 6 * 6,
                    7 * 7 + 8 * 8,
                    9 * 9 + 10 * 10,
                    11 * 11 + 12 * 12,
                    13 * 13 + 14 * 14,
                    -254]);
    }

    #[test]
    fn test_mulhrs() {
        let x16 = mm_setr_epi16(-0x5CEE, 0x0105, 0x3DA9, -0x7FFF, 0x7FFF, 0x1111, -0x219D, -0x1DBC);
        let y16 = mm_setr_epi16(0x4000, -0x510A, 0x209D, -0x7FFF, 0x0000, 0x2222, 0x1027, 0x7AEF);

        assert_eq!(mm_mulhrs_epi16(x16, y16).as_i16x8().as_array(), [-11895, -165, 4022, 32766, 0, 1165, -1086, -7311]);
    }

    #[test]
    fn test_shuffle_epi8() {
        let x8 = mm_setr_epi8(51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66);
        let idx = mm_setr_epi8(4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11);

        assert_eq!(mm_shuffle_epi8(x8, idx).as_i8x16().as_array(),
                   [55, 56, 57, 58, 51, 52, 53, 54, 63, 64, 65, 66, 59, 60, 61, 62]);
    }

    #[test]
    fn test_sign() {
        let x8 = mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let x16 = mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let x32 = mm_setr_epi32(1, 2, 3, 4);

        let idx8 = mm_setr_epi8(0, 1, 2, 3, -1, -2, -3, -4, 0, 1, 2, 3, -1, -2, -3, -4);
        let idx16 = mm_setr_epi16(0, 1, -1, 0, 1, -1, 0, 1);
        let idx32 = mm_setr_epi32(0, 1, -1, 0);

        assert_eq!(mm_sign_epi8(x8, idx8).as_i8x16().as_array(),
                   [0, 2, 3, 4, -5, -6, -7, -8, 0, 10, 11, 12, -13, -14, -15, -16]);
        assert_eq!(mm_sign_epi16(x16, idx16).as_i16x8().as_array(),
                   [0, 2, -3, 0, 5, -6, 0, 8]);
        assert_eq!(mm_sign_epi32(x32, idx32).as_i32x4().as_array(),
                   [0, 2, -3, 0]);
    }

    #[test]
    fn test_palignr() {
        let a = mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = mm_setr_epi8(17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);

        assert_eq!(mm_alignr_epi8(a, b, 1).as_i8x16().as_array(), [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]);
        assert_eq!(mm_alignr_epi8(a, b, 31).as_i8x16().as_array(), [32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

        for i in 0..33 {
            let c = mm_alignr_epi8(a, b, i).as_i8x16().as_array();
            for j in 0..16 {
                if i + j + 1 > 32 {
                    assert_eq!(c[j as usize], 0);
                } else {
                    assert_eq!(c[j as usize], (i + j + 1) as i8);
                }
            }
        }
    }
}
