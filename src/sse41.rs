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

    // TODO(mayah): These functions are not published yet?
    // #[link_name = "llvm.x86.sse41.round.ss"]
    // fn sse41_round_ss(a: m128, b: m128, c: i32) -> m128;
    // #[link_name = "llvm.x86.sse41.round.ps"]
    // fn sse41_round_ps(a: m128, b: i32) -> m128;
    // #[link_name = "llvm.x86.sse41.round.sd"]
    // fn sse41_round_sd(a: m128d, b: m128d, c: i32) -> m128d;
    // #[link_name = "llvm.x86.sse41.round.pd"]
    // fn sse41_round_pd(a: m128d, b: i32) -> m128d;

    #[link_name = "llvm.x86.sse41.insertps"]
    fn sse41_insertps(a: m128, b: m128, c: u8) -> m128;

    #[link_name = "llvm.x86.sse41.ptestc"]
    fn sse41_ptestc(a: i64x2, b: i64x2) -> i32;
    #[link_name = "llvm.x86.sse41.ptestnzc"]
    fn sse41_ptestnzc(a: i64x2, b: i64x2) -> i32;
    #[link_name = "llvm.x86.sse41.ptestz"]
    fn sse41_ptestz(a: i64x2, b: i64x2) -> i32;
}

extern "platform-intrinsic" {
    // TODO(mayah): This is not implemented in rust yet?
    // fn x86_mm_dp_pd(x: m128d, y: m128d, z: i32) -> m128d;
    // fn x86_mm_dp_ps(x: m128, y: m128, z: i32) -> m128;

    fn x86_mm_max_epi32(x: i32x4, y: i32x4) -> i32x4;
    fn x86_mm_max_epi8(x: i8x16, y: i8x16) -> i8x16;
    fn x86_mm_max_epu16(x: u16x8, y: u16x8) -> u16x8;
    fn x86_mm_max_epu32(x: u32x4, y: u32x4) -> u32x4;
    fn x86_mm_min_epi32(x: i32x4, y: i32x4) -> i32x4;
    fn x86_mm_min_epi8(x: i8x16, y: i8x16) -> i8x16;
    fn x86_mm_min_epu16(x: u16x8, y: u16x8) -> u16x8;
    fn x86_mm_min_epu32(x: u32x4, y: u32x4) -> u32x4;

    fn x86_mm_minpos_epu16(x: u16x8) -> u16x8;
    // fn x86_mm_mpsadbw_epu8(x: u8x16, y: u8x16, z: i32) -> u16x8;
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
   // return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v16qs)__V, (__v16qs)__V, 0, 1, 2, 3, 4, 5, 6, 7), __v8hi);
}

// pmovsxbd
// __m128i _mm_cvtepi8_epi32 (__m128i a)
#[inline]
#[allow(unused_variables)]
pub fn mm_cvtepi8_epi32(a: m128i) -> m128i {
    unimplemented!()
    // return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v16qs)__V, (__v16qs)__V, 0, 1, 2, 3), __v4si);
}

// pmovsxbq
// __m128i _mm_cvtepi8_epi64 (__m128i a)
#[inline]
#[allow(unused_variables)]
pub fn mm_cvtepi8_epi64(a: m128i) -> m128i {
    unimplemented!()
    // return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v16qs)__V, (__v16qs)__V, 0, 1), __v2di);
}

// pmovzxwd
// __m128i _mm_cvtepu16_epi32 (__m128i a)
#[inline]
#[allow(unused_variables)]
pub fn mm_cvtepu16_epi32(a: m128i) -> m128i {
    unimplemented!()
    // return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v8hu)__V, (__v8hu)__V, 0, 1, 2, 3), __v4si);
}

// pmovzxwq
// __m128i _mm_cvtepu16_epi64 (__m128i a)
#[inline]
#[allow(unused_variables)]
pub fn mm_cvtepu16_epi64(a: m128i) -> m128i {
    unimplemented!()
    // return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v8hu)__V, (__v8hu)__V, 0, 1), __v2di);
}

// pmovzxdq
// __m128i _mm_cvtepu32_epi64 (__m128i a)
#[inline]
#[allow(unused_variables)]
pub fn mm_cvtepu32_epi64(a: m128i) -> m128i {
    unimplemented!()
    // return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v4su)__V, (__v4su)__V, 0, 1), __v2di);
}

// pmovzxbw
// __m128i _mm_cvtepu8_epi16 (__m128i a)
#[inline]
#[allow(unused_variables)]
pub fn mm_cvtepu8_epi16(a: m128i) -> m128i {
    unimplemented!()
    // return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v16qu)__V, (__v16qu)__V, 0, 1, 2, 3, 4, 5, 6, 7), __v8hi);
}

// pmovzxbd
// __m128i _mm_cvtepu8_epi32 (__m128i a)
#[inline]
#[allow(unused_variables)]
pub fn mm_cvtepu8_epi32(a: m128i) -> m128i {
    unimplemented!()
    // return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v16qu)__V, (__v16qu)__V, 0, 1, 2, 3), __v4si);
}

// pmovzxbq
// __m128i _mm_cvtepu8_epi64 (__m128i a)
#[inline]
#[allow(unused_variables)]
pub fn mm_cvtepu8_epi64(a: m128i) -> m128i {
    unimplemented!()
    // return (__m128i)__builtin_convertvector(__builtin_shufflevector((__v16qu)__V, (__v16qu)__V, 0, 1), __v2di);
}

// dppd
// __m128d _mm_dp_pd (__m128d a, __m128d b, const int imm8)
#[inline]
#[allow(unused_variables)]
pub fn mm_dp_pd(a: m128d, b: m128d, imm8: i32) -> m128d {
    unimplemented!()
    // unsafe { x86_mm_dp_pd(a, b, imm8) }
}

// dpps
// __m128 _mm_dp_ps (__m128 a, __m128 b, const int imm8)
#[inline]
#[allow(unused_variables)]
pub fn mm_dp_ps(a: m128, b: m128, imm8: i32) -> m128 {
    unimplemented!()
    // unsafe { x86_mm_dp_ps(a, b, imm8) }
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
    // TODO(mayah): the third argument of sse41_insertps should be immediate.
    unsafe {
        match imm8 & 0xFF {
            0x00 => sse41_insertps(a, b, 0x00),
            0x01 => sse41_insertps(a, b, 0x01),
            0x02 => sse41_insertps(a, b, 0x02),
            0x03 => sse41_insertps(a, b, 0x03),
            0x04 => sse41_insertps(a, b, 0x04),
            0x05 => sse41_insertps(a, b, 0x05),
            0x06 => sse41_insertps(a, b, 0x06),
            0x07 => sse41_insertps(a, b, 0x07),
            0x08 => sse41_insertps(a, b, 0x08),
            0x09 => sse41_insertps(a, b, 0x09),
            0x0A => sse41_insertps(a, b, 0x0A),
            0x0B => sse41_insertps(a, b, 0x0B),
            0x0C => sse41_insertps(a, b, 0x0C),
            0x0D => sse41_insertps(a, b, 0x0D),
            0x0E => sse41_insertps(a, b, 0x0E),
            0x0F => sse41_insertps(a, b, 0x0F),
            0x10 => sse41_insertps(a, b, 0x10),
            0x11 => sse41_insertps(a, b, 0x11),
            0x12 => sse41_insertps(a, b, 0x12),
            0x13 => sse41_insertps(a, b, 0x13),
            0x14 => sse41_insertps(a, b, 0x14),
            0x15 => sse41_insertps(a, b, 0x15),
            0x16 => sse41_insertps(a, b, 0x16),
            0x17 => sse41_insertps(a, b, 0x17),
            0x18 => sse41_insertps(a, b, 0x18),
            0x19 => sse41_insertps(a, b, 0x19),
            0x1A => sse41_insertps(a, b, 0x1A),
            0x1B => sse41_insertps(a, b, 0x1B),
            0x1C => sse41_insertps(a, b, 0x1C),
            0x1D => sse41_insertps(a, b, 0x1D),
            0x1E => sse41_insertps(a, b, 0x1E),
            0x1F => sse41_insertps(a, b, 0x1F),
            0x20 => sse41_insertps(a, b, 0x20),
            0x21 => sse41_insertps(a, b, 0x21),
            0x22 => sse41_insertps(a, b, 0x22),
            0x23 => sse41_insertps(a, b, 0x23),
            0x24 => sse41_insertps(a, b, 0x24),
            0x25 => sse41_insertps(a, b, 0x25),
            0x26 => sse41_insertps(a, b, 0x26),
            0x27 => sse41_insertps(a, b, 0x27),
            0x28 => sse41_insertps(a, b, 0x28),
            0x29 => sse41_insertps(a, b, 0x29),
            0x2A => sse41_insertps(a, b, 0x2A),
            0x2B => sse41_insertps(a, b, 0x2B),
            0x2C => sse41_insertps(a, b, 0x2C),
            0x2D => sse41_insertps(a, b, 0x2D),
            0x2E => sse41_insertps(a, b, 0x2E),
            0x2F => sse41_insertps(a, b, 0x2F),
            0x30 => sse41_insertps(a, b, 0x30),
            0x31 => sse41_insertps(a, b, 0x31),
            0x32 => sse41_insertps(a, b, 0x32),
            0x33 => sse41_insertps(a, b, 0x33),
            0x34 => sse41_insertps(a, b, 0x34),
            0x35 => sse41_insertps(a, b, 0x35),
            0x36 => sse41_insertps(a, b, 0x36),
            0x37 => sse41_insertps(a, b, 0x37),
            0x38 => sse41_insertps(a, b, 0x38),
            0x39 => sse41_insertps(a, b, 0x39),
            0x3A => sse41_insertps(a, b, 0x3A),
            0x3B => sse41_insertps(a, b, 0x3B),
            0x3C => sse41_insertps(a, b, 0x3C),
            0x3D => sse41_insertps(a, b, 0x3D),
            0x3E => sse41_insertps(a, b, 0x3E),
            0x3F => sse41_insertps(a, b, 0x3F),
            0x40 => sse41_insertps(a, b, 0x40),
            0x41 => sse41_insertps(a, b, 0x41),
            0x42 => sse41_insertps(a, b, 0x42),
            0x43 => sse41_insertps(a, b, 0x43),
            0x44 => sse41_insertps(a, b, 0x44),
            0x45 => sse41_insertps(a, b, 0x45),
            0x46 => sse41_insertps(a, b, 0x46),
            0x47 => sse41_insertps(a, b, 0x47),
            0x48 => sse41_insertps(a, b, 0x48),
            0x49 => sse41_insertps(a, b, 0x49),
            0x4A => sse41_insertps(a, b, 0x4A),
            0x4B => sse41_insertps(a, b, 0x4B),
            0x4C => sse41_insertps(a, b, 0x4C),
            0x4D => sse41_insertps(a, b, 0x4D),
            0x4E => sse41_insertps(a, b, 0x4E),
            0x4F => sse41_insertps(a, b, 0x4F),
            0x50 => sse41_insertps(a, b, 0x50),
            0x51 => sse41_insertps(a, b, 0x51),
            0x52 => sse41_insertps(a, b, 0x52),
            0x53 => sse41_insertps(a, b, 0x53),
            0x54 => sse41_insertps(a, b, 0x54),
            0x55 => sse41_insertps(a, b, 0x55),
            0x56 => sse41_insertps(a, b, 0x56),
            0x57 => sse41_insertps(a, b, 0x57),
            0x58 => sse41_insertps(a, b, 0x58),
            0x59 => sse41_insertps(a, b, 0x59),
            0x5A => sse41_insertps(a, b, 0x5A),
            0x5B => sse41_insertps(a, b, 0x5B),
            0x5C => sse41_insertps(a, b, 0x5C),
            0x5D => sse41_insertps(a, b, 0x5D),
            0x5E => sse41_insertps(a, b, 0x5E),
            0x5F => sse41_insertps(a, b, 0x5F),
            0x60 => sse41_insertps(a, b, 0x60),
            0x61 => sse41_insertps(a, b, 0x61),
            0x62 => sse41_insertps(a, b, 0x62),
            0x63 => sse41_insertps(a, b, 0x63),
            0x64 => sse41_insertps(a, b, 0x64),
            0x65 => sse41_insertps(a, b, 0x65),
            0x66 => sse41_insertps(a, b, 0x66),
            0x67 => sse41_insertps(a, b, 0x67),
            0x68 => sse41_insertps(a, b, 0x68),
            0x69 => sse41_insertps(a, b, 0x69),
            0x6A => sse41_insertps(a, b, 0x6A),
            0x6B => sse41_insertps(a, b, 0x6B),
            0x6C => sse41_insertps(a, b, 0x6C),
            0x6D => sse41_insertps(a, b, 0x6D),
            0x6E => sse41_insertps(a, b, 0x6E),
            0x6F => sse41_insertps(a, b, 0x6F),
            0x70 => sse41_insertps(a, b, 0x70),
            0x71 => sse41_insertps(a, b, 0x71),
            0x72 => sse41_insertps(a, b, 0x72),
            0x73 => sse41_insertps(a, b, 0x73),
            0x74 => sse41_insertps(a, b, 0x74),
            0x75 => sse41_insertps(a, b, 0x75),
            0x76 => sse41_insertps(a, b, 0x76),
            0x77 => sse41_insertps(a, b, 0x77),
            0x78 => sse41_insertps(a, b, 0x78),
            0x79 => sse41_insertps(a, b, 0x79),
            0x7A => sse41_insertps(a, b, 0x7A),
            0x7B => sse41_insertps(a, b, 0x7B),
            0x7C => sse41_insertps(a, b, 0x7C),
            0x7D => sse41_insertps(a, b, 0x7D),
            0x7E => sse41_insertps(a, b, 0x7E),
            0x7F => sse41_insertps(a, b, 0x7F),
            0x80 => sse41_insertps(a, b, 0x80),
            0x81 => sse41_insertps(a, b, 0x81),
            0x82 => sse41_insertps(a, b, 0x82),
            0x83 => sse41_insertps(a, b, 0x83),
            0x84 => sse41_insertps(a, b, 0x84),
            0x85 => sse41_insertps(a, b, 0x85),
            0x86 => sse41_insertps(a, b, 0x86),
            0x87 => sse41_insertps(a, b, 0x87),
            0x88 => sse41_insertps(a, b, 0x88),
            0x89 => sse41_insertps(a, b, 0x89),
            0x8A => sse41_insertps(a, b, 0x8A),
            0x8B => sse41_insertps(a, b, 0x8B),
            0x8C => sse41_insertps(a, b, 0x8C),
            0x8D => sse41_insertps(a, b, 0x8D),
            0x8E => sse41_insertps(a, b, 0x8E),
            0x8F => sse41_insertps(a, b, 0x8F),
            0x90 => sse41_insertps(a, b, 0x90),
            0x91 => sse41_insertps(a, b, 0x91),
            0x92 => sse41_insertps(a, b, 0x92),
            0x93 => sse41_insertps(a, b, 0x93),
            0x94 => sse41_insertps(a, b, 0x94),
            0x95 => sse41_insertps(a, b, 0x95),
            0x96 => sse41_insertps(a, b, 0x96),
            0x97 => sse41_insertps(a, b, 0x97),
            0x98 => sse41_insertps(a, b, 0x98),
            0x99 => sse41_insertps(a, b, 0x99),
            0x9A => sse41_insertps(a, b, 0x9A),
            0x9B => sse41_insertps(a, b, 0x9B),
            0x9C => sse41_insertps(a, b, 0x9C),
            0x9D => sse41_insertps(a, b, 0x9D),
            0x9E => sse41_insertps(a, b, 0x9E),
            0x9F => sse41_insertps(a, b, 0x9F),
            0xA0 => sse41_insertps(a, b, 0xA0),
            0xA1 => sse41_insertps(a, b, 0xA1),
            0xA2 => sse41_insertps(a, b, 0xA2),
            0xA3 => sse41_insertps(a, b, 0xA3),
            0xA4 => sse41_insertps(a, b, 0xA4),
            0xA5 => sse41_insertps(a, b, 0xA5),
            0xA6 => sse41_insertps(a, b, 0xA6),
            0xA7 => sse41_insertps(a, b, 0xA7),
            0xA8 => sse41_insertps(a, b, 0xA8),
            0xA9 => sse41_insertps(a, b, 0xA9),
            0xAA => sse41_insertps(a, b, 0xAA),
            0xAB => sse41_insertps(a, b, 0xAB),
            0xAC => sse41_insertps(a, b, 0xAC),
            0xAD => sse41_insertps(a, b, 0xAD),
            0xAE => sse41_insertps(a, b, 0xAE),
            0xAF => sse41_insertps(a, b, 0xAF),
            0xB0 => sse41_insertps(a, b, 0xB0),
            0xB1 => sse41_insertps(a, b, 0xB1),
            0xB2 => sse41_insertps(a, b, 0xB2),
            0xB3 => sse41_insertps(a, b, 0xB3),
            0xB4 => sse41_insertps(a, b, 0xB4),
            0xB5 => sse41_insertps(a, b, 0xB5),
            0xB6 => sse41_insertps(a, b, 0xB6),
            0xB7 => sse41_insertps(a, b, 0xB7),
            0xB8 => sse41_insertps(a, b, 0xB8),
            0xB9 => sse41_insertps(a, b, 0xB9),
            0xBA => sse41_insertps(a, b, 0xBA),
            0xBB => sse41_insertps(a, b, 0xBB),
            0xBC => sse41_insertps(a, b, 0xBC),
            0xBD => sse41_insertps(a, b, 0xBD),
            0xBE => sse41_insertps(a, b, 0xBE),
            0xBF => sse41_insertps(a, b, 0xBF),
            0xC0 => sse41_insertps(a, b, 0xC0),
            0xC1 => sse41_insertps(a, b, 0xC1),
            0xC2 => sse41_insertps(a, b, 0xC2),
            0xC3 => sse41_insertps(a, b, 0xC3),
            0xC4 => sse41_insertps(a, b, 0xC4),
            0xC5 => sse41_insertps(a, b, 0xC5),
            0xC6 => sse41_insertps(a, b, 0xC6),
            0xC7 => sse41_insertps(a, b, 0xC7),
            0xC8 => sse41_insertps(a, b, 0xC8),
            0xC9 => sse41_insertps(a, b, 0xC9),
            0xCA => sse41_insertps(a, b, 0xCA),
            0xCB => sse41_insertps(a, b, 0xCB),
            0xCC => sse41_insertps(a, b, 0xCC),
            0xCD => sse41_insertps(a, b, 0xCD),
            0xCE => sse41_insertps(a, b, 0xCE),
            0xCF => sse41_insertps(a, b, 0xCF),
            0xD0 => sse41_insertps(a, b, 0xD0),
            0xD1 => sse41_insertps(a, b, 0xD1),
            0xD2 => sse41_insertps(a, b, 0xD2),
            0xD3 => sse41_insertps(a, b, 0xD3),
            0xD4 => sse41_insertps(a, b, 0xD4),
            0xD5 => sse41_insertps(a, b, 0xD5),
            0xD6 => sse41_insertps(a, b, 0xD6),
            0xD7 => sse41_insertps(a, b, 0xD7),
            0xD8 => sse41_insertps(a, b, 0xD8),
            0xD9 => sse41_insertps(a, b, 0xD9),
            0xDA => sse41_insertps(a, b, 0xDA),
            0xDB => sse41_insertps(a, b, 0xDB),
            0xDC => sse41_insertps(a, b, 0xDC),
            0xDD => sse41_insertps(a, b, 0xDD),
            0xDE => sse41_insertps(a, b, 0xDE),
            0xDF => sse41_insertps(a, b, 0xDF),
            0xE0 => sse41_insertps(a, b, 0xE0),
            0xE1 => sse41_insertps(a, b, 0xE1),
            0xE2 => sse41_insertps(a, b, 0xE2),
            0xE3 => sse41_insertps(a, b, 0xE3),
            0xE4 => sse41_insertps(a, b, 0xE4),
            0xE5 => sse41_insertps(a, b, 0xE5),
            0xE6 => sse41_insertps(a, b, 0xE6),
            0xE7 => sse41_insertps(a, b, 0xE7),
            0xE8 => sse41_insertps(a, b, 0xE8),
            0xE9 => sse41_insertps(a, b, 0xE9),
            0xEA => sse41_insertps(a, b, 0xEA),
            0xEB => sse41_insertps(a, b, 0xEB),
            0xEC => sse41_insertps(a, b, 0xEC),
            0xED => sse41_insertps(a, b, 0xED),
            0xEE => sse41_insertps(a, b, 0xEE),
            0xEF => sse41_insertps(a, b, 0xEF),
            0xF0 => sse41_insertps(a, b, 0xF0),
            0xF1 => sse41_insertps(a, b, 0xF1),
            0xF2 => sse41_insertps(a, b, 0xF2),
            0xF3 => sse41_insertps(a, b, 0xF3),
            0xF4 => sse41_insertps(a, b, 0xF4),
            0xF5 => sse41_insertps(a, b, 0xF5),
            0xF6 => sse41_insertps(a, b, 0xF6),
            0xF7 => sse41_insertps(a, b, 0xF7),
            0xF8 => sse41_insertps(a, b, 0xF8),
            0xF9 => sse41_insertps(a, b, 0xF9),
            0xFA => sse41_insertps(a, b, 0xFA),
            0xFB => sse41_insertps(a, b, 0xFB),
            0xFC => sse41_insertps(a, b, 0xFC),
            0xFD => sse41_insertps(a, b, 0xFD),
            0xFE => sse41_insertps(a, b, 0xFE),
            0xFF => sse41_insertps(a, b, 0xFF),
            _ => unimplemented!()
        }
    }
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
#[allow(unused_variables)]
pub fn mm_mpsadbw_epu8(a: m128i, b: m128i, imm8: i32) -> m128i {
    unimplemented!()
    // unsafe { x86_mm_mpsadbw_epu8(a.as_u8x16(), b.as_u8x16(), imm8).as_m128i() }
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

    #[test]
    fn test_mm_dp_pd() {
        // let a = mm_setr_pd(1.5, 10.25);
        // let b = mm_setr_pd(-1.5, 3.125);
        // assert_eq!(mm_dp_pd(a, b, 0x31).as_f64x2().as_array(), [-1.5 * 1.5 + 10.25 * 3.125, 0.0]);
    }

    #[test]
    fn test_mm_dp_ps() {
        // let a = mm_setr_ps(1.5, 10.25, -11.0625, 81.0);
        // let b = mm_setr_ps(-1.5, 3.125, -50.5, 100.0);
        // assert_eq!(mm_dp_ps(a, b, 0x55).as_f32x4().as_array(), [556.406250, 0.000000, 556.406250, 0.000000]);
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
        // let a = u8x16(15, 60, 55, 31, 0, 1, 2, 4, 8, 16, 32, 64, 128, 255, 1, 17).as_m128i();
        // let b = u8x16(2, 4, 8, 64, 255, 0, 1, 16, 32, 64, 128, 255, 75, 31, 42, 11).as_m128i();
        // assert_eq!(mm_mpsadbw_epu8(a, b, 5).as_u16x8().as_array(), [269, 267, 264, 290, 342, 446, 653, 588]);
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
