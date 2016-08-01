use std;
use super::*;
use super::{simd_add, simd_sub, simd_mul, simd_div,
            simd_and, simd_or, simd_xor,
            simd_eq, simd_ge, simd_gt, simd_lt, simd_le, simd_ne,
            simd_shuffle16};

extern {
    #[link_name = "llvm.x86.sse2.cmp.sd"]
    pub fn sse2_cmp_sd(a: m128d, b: m128d, c: i8) -> m128d;
    #[link_name = "llvm.x86.sse2.cmp.pd"]
    pub fn sse2_cmp_pd(a: m128d, b: m128d, c: i8) -> m128d;

    #[link_name = "llvm.x86.sse2.comieq.sd"]
    pub fn sse2_comieq_sd(a: m128d, b: m128d) -> i32;
    #[link_name = "llvm.x86.sse2.comigt.sd"]
    pub fn sse2_comigt_sd(a: m128d, b: m128d) -> i32;
    #[link_name = "llvm.x86.sse2.comige.sd"]
    pub fn sse2_comige_sd(a: m128d, b: m128d) -> i32;
    #[link_name = "llvm.x86.sse2.comilt.sd"]
    pub fn sse2_comilt_sd(a: m128d, b: m128d) -> i32;
    #[link_name = "llvm.x86.sse2.comile.sd"]
    pub fn sse2_comile_sd(a: m128d, b: m128d) -> i32;
    #[link_name = "llvm.x86.sse2.comineq.sd"]
    pub fn sse2_comineq_sd(a: m128d, b: m128d) -> i32;
    #[link_name = "llvm.x86.sse2.ucomieq.sd"]
    pub fn sse2_ucomieq_sd(a: m128d, b: m128d) -> i32;
    #[link_name = "llvm.x86.sse2.ucomigt.sd"]
    pub fn sse2_ucomigt_sd(a: m128d, b: m128d) -> i32;
    #[link_name = "llvm.x86.sse2.ucomige.sd"]
    pub fn sse2_ucomige_sd(a: m128d, b: m128d) -> i32;
    #[link_name = "llvm.x86.sse2.ucomilt.sd"]
    pub fn sse2_ucomilt_sd(a: m128d, b: m128d) -> i32;
    #[link_name = "llvm.x86.sse2.ucomile.sd"]
    pub fn sse2_ucomile_sd(a: m128d, b: m128d) -> i32;
    #[link_name = "llvm.x86.sse2.ucomineq.sd"]
    pub fn sse2_ucomineq_sd(a: m128d, b: m128d) -> i32;

    #[link_name = "llvm.x86.sse2.pslli.w"]
    pub fn sse2_pslli_w(a: i16x8, b: i32) -> i16x8;
    #[link_name = "llvm.x86.sse2.psrli.w"]
    pub fn sse2_psrli_w(a: i16x8, b: i32) -> i16x8;
}

extern "platform-intrinsic" {
    fn x86_mm_adds_epi8(x: i8x16, y: i8x16) -> i8x16;
    fn x86_mm_adds_epu8(x: u8x16, y: u8x16) -> u8x16;
    fn x86_mm_adds_epi16(x: i16x8, y: i16x8) -> i16x8;
    fn x86_mm_adds_epu16(x: u16x8, y: u16x8) -> u16x8;
    fn x86_mm_subs_epi8(x: i8x16, y: i8x16) -> i8x16;
    fn x86_mm_subs_epu8(x: u8x16, y: u8x16) -> u8x16;
    fn x86_mm_subs_epi16(x: i16x8, y: i16x8) -> i16x8;
    fn x86_mm_subs_epu16(x: u16x8, y: u16x8) -> u16x8;

    fn x86_mm_avg_epu8(x: u8x16, y: u8x16) -> u8x16;
    fn x86_mm_avg_epu16(x: u16x8, y: u16x8) -> u16x8;
}

macro_rules! m128i_operators {
    ($name: ident, $method: ident, $func: ident) => {
        impl std::ops::$name for m128i {
            type Output = Self;

            #[inline]
            fn $method(self, x: Self) -> Self {
                unsafe { $func(self, x) }
            }
        }
    }
}

// Add &, |, ^ operators for m128i.
m128i_operators! { BitAnd, bitand, simd_and }
m128i_operators! { BitOr,  bitor,  simd_or }
m128i_operators! { BitXor, bitxor, simd_xor }

// paddw
// __m128i _mm_add_epi16 (__m128i a, __m128i b)
#[inline]
pub fn mm_add_epi16(a: m128i, b: m128i) -> m128i {
    unsafe { simd_add(a.as_i16x8(), b.as_i16x8()).as_m128i() }
}

// paddd
// __m128i _mm_add_epi32 (__m128i a, __m128i b)
#[inline]
pub fn mm_add_epi32(a: m128i, b: m128i) -> m128i {
    unsafe { simd_add(a.as_i32x4(), b.as_i32x4()).as_m128i() }
}

// paddq
// __m128i _mm_add_epi64 (__m128i a, __m128i b)
#[inline]
pub fn mm_add_epi64(a: m128i, b: m128i) -> m128i {
    unsafe { simd_add(a.as_i64x2(), b.as_i64x2()).as_m128i() }
}

// paddb
// __m128i _mm_add_epi8 (__m128i a, __m128i b)
#[inline]
pub fn mm_add_epi8(a: m128i, b: m128i) -> m128i {
    unsafe { simd_add(a.as_i8x16(), b.as_i8x16()).as_m128i() }
}

// addpd
// __m128d _mm_add_pd (__m128d a, __m128d b)
#[inline]
pub fn mm_add_pd(a: m128d, b: m128d) -> m128d {
    unsafe { simd_add(a.as_f64x2(), b.as_f64x2()).as_m128d() }
}

// addsd
// __m128d _mm_add_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_add_sd(a: m128d, b: m128d) -> m128d {
    let v = a.as_f64x2().extract(0) + b.as_f64x2().extract(0);
    a.as_f64x2().insert(0, v).as_m128d()
}

// paddq
// __m64 _mm_add_si64 (__m64 a, __m64 b)

// paddsw
// __m128i _mm_adds_epi16 (__m128i a, __m128i b)
#[inline]
pub fn mm_adds_epi16(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_adds_epi16(a.as_i16x8(), b.as_i16x8()).as_m128i() }
}

// paddsb
// __m128i _mm_adds_epi8 (__m128i a, __m128i b)
#[inline]
pub fn mm_adds_epi8(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_adds_epi8(a.as_i8x16(), b.as_i8x16()).as_m128i() }
}

// paddusw
// __m128i _mm_adds_epu16 (__m128i a, __m128i b)
#[inline]
pub fn mm_adds_epu16(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_adds_epu16(a.as_u16x8(), b.as_u16x8()).as_m128i() }
}

// paddusb
// __m128i _mm_adds_epu8 (__m128i a, __m128i b)
#[inline]
pub fn mm_adds_epu8(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_adds_epu8(a.as_u8x16(), b.as_u8x16()).as_m128i() }
}

// andpd
// __m128d _mm_and_pd (__m128d a, __m128d b)
#[inline]
pub fn mm_and_pd(a: m128d, b: m128d) -> m128d {
    mm_and_si128(a.as_m128i(), b.as_m128i()).as_m128d()
}

// pand
// __m128i _mm_and_si128 (__m128i a, __m128i b)
#[inline]
pub fn mm_and_si128(a: m128i, b: m128i) -> m128i {
    unsafe { simd_and(a, b) }
}

// andnpd
// __m128d _mm_andnot_pd (__m128d a, __m128d b)
#[inline]
pub fn mm_andnot_pd(a: m128d, b: m128d) -> m128d {
    mm_andnot_si128(a.as_m128i(), b.as_m128i()).as_m128d()
}

// pandn
// __m128i _mm_andnot_si128 (__m128i a, __m128i b)
#[inline]
pub fn mm_andnot_si128(a: m128i, b: m128i) -> m128i {
    let ones = i64x2(!0, !0).as_m128i();
    mm_and_si128(mm_xor_si128(a, ones), b)
}

// pavgw
// __m128i _mm_avg_epu16 (__m128i a, __m128i b)
#[inline]
pub fn mm_avg_epu16(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_avg_epu16(a.as_u16x8(), b.as_u16x8()).as_m128i() }
}

// pavgb
// __m128i _mm_avg_epu8 (__m128i a, __m128i b)
#[inline]
pub fn mm_avg_epu8(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_avg_epu8(a.as_u8x16(), b.as_u8x16()).as_m128i() }
}

// pslldq
// __m128i _mm_bslli_si128 (__m128i a, int imm8)
#[inline]
pub fn mm_bslli_si128(a: m128i, imm8: i32) -> m128i {
    mm_slli_si128(a, imm8)
}

// psrldq
// __m128i _mm_bsrli_si128 (__m128i a, int imm8)
#[inline]
pub fn mm_bsrli_si128(a: m128i, imm8: i32) -> m128i {
    mm_srli_si128(a, imm8)
}

// __m128 _mm_castpd_ps (__m128d a)
#[inline]
pub fn mm_castpd_ps(a: m128d) -> m128 {
    a.as_m128()
}

// __m128i _mm_castpd_si128 (__m128d a)
#[inline]
pub fn mm_castpd_si128(a: m128d) -> m128i {
    a.as_m128i()
}

// __m128d _mm_castps_pd (__m128 a)
#[inline]
pub fn mm_castps_pd(a: m128) -> m128d {
    a.as_m128d()
}

// __m128i _mm_castps_si128 (__m128 a)
#[inline]
pub fn mm_castps_si128(a: m128) -> m128i {
    a.as_m128i()
}

// __m128d _mm_castsi128_pd (__m128i a)
#[inline]
pub fn mm_castsi128_pd(a: m128i) -> m128d {
    a.as_m128d()
}

// __m128 _mm_castsi128_ps (__m128i a)
#[inline]
pub fn mm_castsi128_ps(a: m128i) -> m128 {
    a.as_m128()
}

// clflush
// void _mm_clflush (void const* p)

// pcmpeqw
// __m128i _mm_cmpeq_epi16 (__m128i a, __m128i b)
#[inline]
pub fn mm_cmpeq_epi16(a: m128i, b: m128i) -> m128i {
    let x: i16x8 = unsafe { simd_eq(a.as_i16x8(), b.as_i16x8()) };
    x.as_m128i()
}

// pcmpeqd
// __m128i _mm_cmpeq_epi32 (__m128i a, __m128i b)
#[inline]
pub fn mm_cmpeq_epi32(a: m128i, b: m128i) -> m128i {
    let x: i32x4 = unsafe { simd_eq(a.as_i32x4(), b.as_i32x4()) };
    x.as_m128i()
}

// pcmpeqb
// __m128i _mm_cmpeq_epi8 (__m128i a, __m128i b)
#[inline]
pub fn mm_cmpeq_epi8(a: m128i, b: m128i) -> m128i {
    let x: i8x16 = unsafe { simd_eq(a.as_i8x16(), b.as_i8x16()) };
    x.as_m128i()
}

// cmppd
// __m128d _mm_cmpeq_pd (__m128d a, __m128d b)
#[inline]
pub fn mm_cmpeq_pd(a: m128d, b: m128d) -> m128d {
    let x: i64x2 = unsafe { simd_eq(a.as_f64x2(), b.as_f64x2()) };
    x.as_m128d()
}

// cmpsd
// __m128d _mm_cmpeq_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_cmpeq_sd(a: m128d, b: m128d) -> m128d {
    unsafe { sse2_cmp_sd(a, b, 0) }
}

// cmppd
// __m128d _mm_cmpge_pd (__m128d a, __m128d b)
#[inline]
pub fn mm_cmpge_pd(a: m128d, b: m128d) -> m128d {
    let x: i64x2 = unsafe { simd_ge(a.as_f64x2(), b.as_f64x2()) };
    x.as_m128d()
}

// cmpsd
// __m128d _mm_cmpge_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_cmpge_sd(a: m128d, b: m128d) -> m128d {
    let c = mm_cmple_sd(b, a);
    f64x2(c.as_f64x2().extract(0), a.as_f64x2().extract(1)).as_m128d()
}

// pcmpgtw
// __m128i _mm_cmpgt_epi16 (__m128i a, __m128i b)
#[inline]
pub fn mm_cmpgt_epi16(a: m128i, b: m128i) -> m128i {
    let x: i16x8 = unsafe { simd_gt(a.as_i16x8(), b.as_i16x8()) };
    x.as_m128i()
}

// pcmpgtd
// __m128i _mm_cmpgt_epi32 (__m128i a, __m128i b)
#[inline]
pub fn mm_cmpgt_epi32(a: m128i, b: m128i) -> m128i {
    let x: i32x4 = unsafe { simd_gt(a.as_i32x4(), b.as_i32x4()) };
    x.as_m128i()
}

// pcmpgtb
// __m128i _mm_cmpgt_epi8 (__m128i a, __m128i b)
#[inline]
pub fn mm_cmpgt_epi8(a: m128i, b: m128i) -> m128i {
    let x: i8x16 = unsafe { simd_gt(a.as_i8x16(), b.as_i8x16()) };
    x.as_m128i()
}

// cmppd
// __m128d _mm_cmpgt_pd (__m128d a, __m128d b)
#[inline]
pub fn mm_cmpgt_pd(a: m128d, b: m128d) -> m128d {
    let x: i64x2 = unsafe { simd_gt(a.as_f64x2(), b.as_f64x2()) };
    x.as_m128d()
}

// cmpsd
// __m128d _mm_cmpgt_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_cmpgt_sd(a: m128d, b: m128d) -> m128d {
    let c = mm_cmplt_sd(b, a);
    f64x2(c.as_f64x2().extract(0), a.as_f64x2().extract(1)).as_m128d()
}

// cmppd
// __m128d _mm_cmple_pd (__m128d a, __m128d b)
#[inline]
pub fn mm_cmple_pd(a: m128d, b: m128d) -> m128d {
    let x: i64x2 = unsafe { simd_le(a.as_f64x2(), b.as_f64x2()) };
    x.as_m128d()
}

// cmpsd
// __m128d _mm_cmple_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_cmple_sd(a: m128d, b: m128d) -> m128d {
    unsafe { sse2_cmp_sd(a, b, 2) }
}

// pcmpgtw
// __m128i _mm_cmplt_epi16 (__m128i a, __m128i b)
#[inline]
pub fn mm_cmplt_epi16(a: m128i, b: m128i) -> m128i {
    let x: i16x8 = unsafe { simd_lt(a.as_i16x8(), b.as_i16x8()) };
    x.as_m128i()
}

// pcmpgtd
// __m128i _mm_cmplt_epi32 (__m128i a, __m128i b)
#[inline]
pub fn mm_cmplt_epi32(a: m128i, b: m128i) -> m128i {
    let x: i32x4 = unsafe { simd_lt(a.as_i32x4(), b.as_i32x4()) };
    x.as_m128i()
}

// pcmpgtb
// __m128i _mm_cmplt_epi8 (__m128i a, __m128i b)
#[inline]
pub fn mm_cmplt_epi8(a: m128i, b: m128i) -> m128i {
    let x: i8x16 = unsafe { simd_lt(a.as_i8x16(), b.as_i8x16()) };
    x.as_m128i()
}

// cmppd
// __m128d _mm_cmplt_pd (__m128d a, __m128d b)
#[inline]
pub fn mm_cmplt_pd(a: m128d, b: m128d) -> m128d {
    let x: i64x2 = unsafe { simd_lt(a.as_f64x2(), b.as_f64x2()) };
    x.as_m128d()
}

// cmpsd
// __m128d _mm_cmplt_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_cmplt_sd(a: m128d, b: m128d) -> m128d {
    unsafe { sse2_cmp_sd(a, b, 1) }
}

// cmppd
// __m128d _mm_cmpneq_pd (__m128d a, __m128d b)
#[inline]
pub fn mm_cmpneq_pd(a: m128d, b: m128d) -> m128d {
    let x: i64x2 = unsafe { simd_ne(a.as_f64x2(), b.as_f64x2()) };
    x.as_m128d()
}

// cmpsd
// __m128d _mm_cmpneq_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_cmpneq_sd(a: m128d, b: m128d) -> m128d {
    unsafe { sse2_cmp_sd(a, b, 4) }
}

// cmppd
// __m128d _mm_cmpnge_pd (__m128d a, __m128d b)
#[inline]
pub fn mm_cmpnge_pd(a: m128d, b: m128d) -> m128d {
    unsafe { sse2_cmp_pd(b, a, 6) }
}

// cmpsd
// __m128d _mm_cmpnge_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_cmpnge_sd(a: m128d, b: m128d) -> m128d {
    let c = mm_cmpnle_sd(b, a);
    f64x2(c.as_f64x2().extract(0), a.as_f64x2().extract(1)).as_m128d()
}

// cmppd
// __m128d _mm_cmpngt_pd (__m128d a, __m128d b)
#[inline]
pub fn mm_cmpngt_pd(a: m128d, b: m128d) -> m128d {
    unsafe { sse2_cmp_pd(b, a, 5) }
}

// cmpsd
// __m128d _mm_cmpngt_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_cmpngt_sd(a: m128d, b: m128d) -> m128d {
    let c = unsafe { sse2_cmp_sd(b, a, 5) };
    f64x2(c.as_f64x2().extract(0), a.as_f64x2().extract(1)).as_m128d()
}

// cmppd
// __m128d _mm_cmpnle_pd (__m128d a, __m128d b)
#[inline]
pub fn mm_cmpnle_pd(a: m128d, b: m128d) -> m128d {
    unsafe { sse2_cmp_pd(a, b, 6) }
}

// cmpsd
// __m128d _mm_cmpnle_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_cmpnle_sd(a: m128d, b: m128d) -> m128d {
    unsafe { sse2_cmp_sd(a, b, 6) }
}

// cmppd
// __m128d _mm_cmpnlt_pd (__m128d a, __m128d b)
#[inline]
pub fn mm_cmpnlt_pd(a: m128d, b: m128d) -> m128d {
    unsafe { sse2_cmp_pd(a, b, 5) }
}

// cmpsd
// __m128d _mm_cmpnlt_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_cmpnlt_sd(a: m128d, b: m128d) -> m128d {
    unsafe { sse2_cmp_sd(a, b, 5) }
}

// cmppd
// __m128d _mm_cmpord_pd (__m128d a, __m128d b)
#[inline]
pub fn mm_cmpord_pd(a: m128d, b: m128d) -> m128d {
    unsafe { sse2_cmp_pd(a, b, 7) }
}

// cmpsd
// __m128d _mm_cmpord_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_cmpord_sd(a: m128d, b: m128d) -> m128d {
    unsafe { sse2_cmp_sd(a, b, 7) }
}

// cmppd
// __m128d _mm_cmpunord_pd (__m128d a, __m128d b)
#[inline]
pub fn mm_cmpunord_pd(a: m128d, b: m128d) -> m128d {
    unsafe { sse2_cmp_pd(a, b, 3) }
}

// cmpsd
// __m128d _mm_cmpunord_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_cmpunord_sd(a: m128d, b: m128d) -> m128d {
    unsafe { sse2_cmp_sd(a, b, 3) }
}

// comisd
// int _mm_comieq_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_comieq_sd(a: m128d, b: m128d) -> i32 {
    unsafe { sse2_comieq_sd(a, b) }
}

// comisd
// int _mm_comige_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_comige_sd(a: m128d, b: m128d) -> i32 {
    unsafe { sse2_comige_sd(a, b) }
}

// comisd
// int _mm_comigt_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_comigt_sd(a: m128d, b: m128d) -> i32 {
    unsafe { sse2_comigt_sd(a, b) }
}

// comisd
// int _mm_comile_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_comile_sd(a: m128d, b: m128d) -> i32 {
    unsafe { sse2_comile_sd(a, b) }
}

// comisd
// int _mm_comilt_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_comilt_sd(a: m128d, b: m128d) -> i32 {
    unsafe { sse2_comilt_sd(a, b) }
}

// comisd
// int _mm_comineq_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_comineq_sd(a: m128d, b: m128d) -> i32 {
    unsafe { sse2_comineq_sd(a, b) }
}

// cvtdq2pd
// __m128d _mm_cvtepi32_pd (__m128i a)
// cvtdq2ps
// __m128 _mm_cvtepi32_ps (__m128i a)
// cvtpd2dq
// __m128i _mm_cvtpd_epi32 (__m128d a)
// cvtpd2pi
// __m64 _mm_cvtpd_pi32 (__m128d a)
// cvtpd2ps
// __m128 _mm_cvtpd_ps (__m128d a)
// cvtpi2pd
// __m128d _mm_cvtpi32_pd (__m64 a)
// cvtps2dq
// __m128i _mm_cvtps_epi32 (__m128 a)
// cvtps2pd
// __m128d _mm_cvtps_pd (__m128 a)
// movsd
// double _mm_cvtsd_f64 (__m128d a)
// cvtsd2si
// int _mm_cvtsd_si32 (__m128d a)
// cvtsd2si
// __int64 _mm_cvtsd_si64 (__m128d a)
// cvtsd2si
// __int64 _mm_cvtsd_si64x (__m128d a)
// cvtsd2ss
// __m128 _mm_cvtsd_ss (__m128 a, __m128d b)
// movd
// int _mm_cvtsi128_si32 (__m128i a)
// movq
// __int64 _mm_cvtsi128_si64 (__m128i a)
// movq
// __int64 _mm_cvtsi128_si64x (__m128i a)
// cvtsi2sd
// __m128d _mm_cvtsi32_sd (__m128d a, int b)
// movd
// __m128i _mm_cvtsi32_si128 (int a)
// cvtsi2sd
// __m128d _mm_cvtsi64_sd (__m128d a, __int64 b)
// movq
// __m128i _mm_cvtsi64_si128 (__int64 a)
// cvtsi2sd
// __m128d _mm_cvtsi64x_sd (__m128d a, __int64 b)
// movq
// __m128i _mm_cvtsi64x_si128 (__int64 a)
// cvtss2sd
// __m128d _mm_cvtss_sd (__m128d a, __m128 b)
// cvttpd2dq
// __m128i _mm_cvttpd_epi32 (__m128d a)
// cvttpd2pi
// __m64 _mm_cvttpd_pi32 (__m128d a)
// cvttps2dq
// __m128i _mm_cvttps_epi32 (__m128 a)
// cvttsd2si
// int _mm_cvttsd_si32 (__m128d a)
// cvttsd2si
// __int64 _mm_cvttsd_si64 (__m128d a)
// cvttsd2si
// __int64 _mm_cvttsd_si64x (__m128d a)

// divpd
// __m128d _mm_div_pd (__m128d a, __m128d b)
#[inline]
pub fn mm_div_pd(a: m128d, b: m128d) -> m128d {
    unsafe { simd_div(a.as_f64x2(), b.as_f64x2()).as_m128d() }
}

// divsd
// __m128d _mm_div_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_div_sd(a: m128d, b: m128d) -> m128d {
    let v = a.as_f64x2().extract(0) / b.as_f64x2().extract(0);
    a.as_f64x2().insert(0, v).as_m128d()
}

// pextrw
// int _mm_extract_epi16 (__m128i a, int imm8)
// pinsrw
// __m128i _mm_insert_epi16 (__m128i a, int i, int imm8)
// lfence
// void _mm_lfence (void)
// movapd
// __m128d _mm_load_pd (double const* mem_addr)
// ...
// __m128d _mm_load_pd1 (double const* mem_addr)
// movsd
// __m128d _mm_load_sd (double const* mem_addr)
// movdqa
// __m128i _mm_load_si128 (__m128i const* mem_addr)
// ...
// __m128d _mm_load1_pd (double const* mem_addr)
// movhpd
// __m128d _mm_loadh_pd (__m128d a, double const* mem_addr)
// movq
// __m128i _mm_loadl_epi64 (__m128i const* mem_addr)
// movlpd
// __m128d _mm_loadl_pd (__m128d a, double const* mem_addr)
// ...
// __m128d _mm_loadr_pd (double const* mem_addr)
// movupd
// __m128d _mm_loadu_pd (double const* mem_addr)
// movdqu
// __m128i _mm_loadu_si128 (__m128i const* mem_addr)
// pmaddwd
// __m128i _mm_madd_epi16 (__m128i a, __m128i b)
// maskmovdqu
// void _mm_maskmoveu_si128 (__m128i a, __m128i mask, char* mem_addr)
// pmaxsw
// __m128i _mm_max_epi16 (__m128i a, __m128i b)
// pmaxub
// __m128i _mm_max_epu8 (__m128i a, __m128i b)
// maxpd
// __m128d _mm_max_pd (__m128d a, __m128d b)
// maxsd
// __m128d _mm_max_sd (__m128d a, __m128d b)
// mfence
// void _mm_mfence (void)
// pminsw
// __m128i _mm_min_epi16 (__m128i a, __m128i b)
// pminub
// __m128i _mm_min_epu8 (__m128i a, __m128i b)
// minpd
// __m128d _mm_min_pd (__m128d a, __m128d b)
// minsd
// __m128d _mm_min_sd (__m128d a, __m128d b)
// movq
// __m128i _mm_move_epi64 (__m128i a)
// movsd
// __m128d _mm_move_sd (__m128d a, __m128d b)
// pmovmskb
// int _mm_movemask_epi8 (__m128i a)
// movmskpd
// int _mm_movemask_pd (__m128d a)
// movdq2q
// __m64 _mm_movepi64_pi64 (__m128i a)
// movq2dq
// __m128i _mm_movpi64_epi64 (__m64 a)
// pmuludq
// __m128i _mm_mul_epu32 (__m128i a, __m128i b)

// mulpd
// __m128d _mm_mul_pd (__m128d a, __m128d b)
#[inline]
pub fn mm_mul_pd(a: m128d, b: m128d) -> m128d {
    unsafe { simd_mul(a.as_f64x2(), b.as_f64x2()).as_m128d() }
}

// mulsd
// __m128d _mm_mul_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_mul_sd(a: m128d, b: m128d) -> m128d {
    let v = a.as_f64x2().extract(0) * b.as_f64x2().extract(0);
    a.as_f64x2().insert(0, v).as_m128d()
}

// pmuludq
// __m64 _mm_mul_su32 (__m64 a, __m64 b)
// pmulhw
// __m128i _mm_mulhi_epi16 (__m128i a, __m128i b)
// pmulhuw
// __m128i _mm_mulhi_epu16 (__m128i a, __m128i b)
// pmullw
// __m128i _mm_mullo_epi16 (__m128i a, __m128i b)

// orpd
// __m128d _mm_or_pd (__m128d a, __m128d b)
#[inline]
pub fn mm_or_pd(a: m128d, b: m128d) -> m128d {
    mm_or_si128(a.as_m128i(), b.as_m128i()).as_m128d()
}

// por
// __m128i _mm_or_si128 (__m128i a, __m128i b)
#[inline]
pub fn mm_or_si128(a: m128i, b: m128i) -> m128i {
    unsafe { simd_or(a, b) }
}

// packsswb
// __m128i _mm_packs_epi16 (__m128i a, __m128i b)
// packssdw
// __m128i _mm_packs_epi32 (__m128i a, __m128i b)
// packuswb
// __m128i _mm_packus_epi16 (__m128i a, __m128i b)
// pause
// void _mm_pause (void)
// psadbw
// __m128i _mm_sad_epu8 (__m128i a, __m128i b)

// ...
// __m128i _mm_set_epi16 (short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0)
#[inline]
pub fn mm_set_epi16(e7: i16, e6: i16, e5: i16, e4: i16,
                    e3: i16, e2: i16, e1: i16, e0: i16) -> m128i {
    i16x8(e0, e1, e2, e3, e4, e5, e6, e7).as_m128i()
}

// ...
// __m128i _mm_set_epi32 (int e3, int e2, int e1, int e0)
#[inline]
pub fn mm_set_epi32(e3: i32, e2: i32, e1: i32, e0: i32) -> m128i {
    i32x4(e0, e1, e2, e3).as_m128i()
}

// ...
// __m128i _mm_set_epi64 (__m64 e1, __m64 e0)

// ...
// __m128i _mm_set_epi64x (__int64 e1, __int64 e0)
#[inline]
pub fn mm_set_epi64x(e1: i64, e0: i64) -> m128i {
    i64x2(e0, e1).as_m128i()
}

// ...
// __m128i _mm_set_epi8 (char e15, char e14, char e13, char e12, char e11, char e10, char e9, char e8, char e7, char e6, char e5, char e4, char e3, char e2, char e1, char e0)
#[inline]
pub fn mm_set_epi8(e15: i8, e14: i8, e13: i8, e12: i8, e11: i8, e10: i8, e9: i8, e8: i8,
                   e7: i8, e6: i8, e5: i8, e4: i8, e3: i8, e2: i8, e1: i8, e0: i8) -> m128i {
    i8x16(e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15).as_m128i()
}

// ...
// __m128d _mm_set_pd (double e1, double e0)
#[inline]
pub fn mm_set_pd(e1: f64, e0: f64) -> m128d {
    f64x2(e0, e1).as_m128d()
}

// ...
// __m128d _mm_set_pd1 (double a)
#[inline]
pub fn mm_set_pd1(a: f64) -> m128d {
    f64x2(a, a).as_m128d()
}

// ...
// __m128d _mm_set_sd (double a)
#[inline]
pub fn mm_set_sd(a: f64) -> m128d {
    f64x2(a, 0.0).as_m128d()
}

// ...
// __m128i _mm_set1_epi16 (short a)
#[inline]
pub fn mm_set1_epi16(a: i16) -> m128i {
    i16x8(a, a, a, a, a, a, a, a).as_m128i()
}

// ...
// __m128i _mm_set1_epi32 (int a)
#[inline]
pub fn mm_set1_epi32(a: i32) -> m128i {
    i32x4(a, a, a, a).as_m128i()
}

// ...
// __m128i _mm_set1_epi64 (__m64 a)

// ...
// __m128i _mm_set1_epi64x (__int64 a)
#[inline]
pub fn mm_set1_epi64(a: i64) -> m128i {
    i64x2(a, a).as_m128i()
}

// ...
// __m128i _mm_set1_epi8 (char a)
#[inline]
pub fn mm_set1_epi8(a: i8) -> m128i {
    i8x16(a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a).as_m128i()
}

// ...
// __m128d _mm_set1_pd (double a)
#[inline]
pub fn mm_set1_pd(a: f64) -> m128d {
    f64x2(a, a).as_m128d()
}

// ...
// __m128i _mm_setr_epi16 (short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0)
#[inline]
pub fn mm_setr_epi16(e0: i16, e1: i16, e2: i16, e3: i16,
                     e4: i16, e5: i16, e6: i16, e7: i16) -> m128i {
    i16x8(e0, e1, e2, e3, e4, e5, e6, e7).as_m128i()
}

// ...
// __m128i _mm_setr_epi32 (int e3, int e2, int e1, int e0)
#[inline]
pub fn mm_setr_epi32(e0: i32, e1: i32, e2: i32, e3: i32) -> m128i {
    i32x4(e0, e1, e2, e3).as_m128i()
}

// ...
// __m128i _mm_setr_epi64 (__m64 e1, __m64 e0)

// ...
// __m128i _mm_setr_epi8 (char e15, char e14, char e13, char e12, char e11, char e10, char e9, char e8, char e7, char e6, char e5, char e4, char e3, char e2, char e1, char e0)
#[inline]
pub fn mm_setr_epi8(e0: i8, e1: i8, e2: i8, e3: i8, e4: i8, e5: i8, e6: i8, e7: i8,
                    e8: i8, e9: i8, e10: i8, e11: i8, e12: i8, e13: i8, e14: i8, e15: i8) -> m128i {
    i8x16(e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15).as_m128i()
}

// ...
// __m128d _mm_setr_pd (double e1, double e0)
#[inline]
pub fn mm_setr_pd(e0: f64, e1: f64) -> m128d {
    f64x2(e0, e1).as_m128d()
}

// xorpd
// __m128d _mm_setzero_pd (void)
#[inline]
pub fn mm_setzero_pd() -> m128d {
    f64x2(0.0, 0.0).as_m128d()
}

// pxor
// __m128i _mm_setzero_si128 ()
#[inline]
pub fn mm_setzero_si128() -> m128i {
    i32x4(0, 0, 0, 0).as_m128i()
}

// pshufd
// __m128i _mm_shuffle_epi32 (__m128i a, int imm8)
// shufpd
// __m128d _mm_shuffle_pd (__m128d a, __m128d b, int imm8)
// pshufhw
// __m128i _mm_shufflehi_epi16 (__m128i a, int imm8)
// pshuflw
// __m128i _mm_shufflelo_epi16 (__m128i a, int imm8)
// psllw
// __m128i _mm_sll_epi16 (__m128i a, __m128i count)
// pslld
// __m128i _mm_sll_epi32 (__m128i a, __m128i count)
// psllq
// __m128i _mm_sll_epi64 (__m128i a, __m128i count)

// psllw
// __m128i _mm_slli_epi16 (__m128i a, int imm8)
#[inline]
pub fn mm_slli_epi16(a: m128i, imm8: i32) -> m128i {
    unsafe { sse2_pslli_w(a.as_i16x8(), imm8).as_m128i() }
}

// pslld
// __m128i _mm_slli_epi32 (__m128i a, int imm8)
// psllq
// __m128i _mm_slli_epi64 (__m128i a, int imm8)

// pslldq
// __m128i _mm_slli_si128 (__m128i a, int imm8)
#[inline]
pub fn mm_slli_si128(a: m128i, imm8: i32) -> m128i {
    let zero = mm_setzero_si128().as_i8x16();
    let aa = a.as_i8x16();
    let x: i8x16 = unsafe {
        match imm8 & 0xFF {
            0x0 => simd_shuffle16(zero, aa, [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]),
            0x1 => simd_shuffle16(zero, aa, [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]),
            0x2 => simd_shuffle16(zero, aa, [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]),
            0x3 => simd_shuffle16(zero, aa, [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]),
            0x4 => simd_shuffle16(zero, aa, [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]),
            0x5 => simd_shuffle16(zero, aa, [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]),
            0x6 => simd_shuffle16(zero, aa, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]),
            0x7 => simd_shuffle16(zero, aa, [ 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]),
            0x8 => simd_shuffle16(zero, aa, [ 8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]),
            0x9 => simd_shuffle16(zero, aa, [ 7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]),
            0xA => simd_shuffle16(zero, aa, [ 6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]),
            0xB => simd_shuffle16(zero, aa, [ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
            0xC => simd_shuffle16(zero, aa, [ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
            0xD => simd_shuffle16(zero, aa, [ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
            0xE => simd_shuffle16(zero, aa, [ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17]),
            0xF => simd_shuffle16(zero, aa, [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]),
              _ => simd_shuffle16(zero, aa, [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]),
        }
    };
    x.as_m128i()
}

// sqrtpd
// __m128d _mm_sqrt_pd (__m128d a)
// sqrtsd
// __m128d _mm_sqrt_sd (__m128d a, __m128d b)
// psraw
// __m128i _mm_sra_epi16 (__m128i a, __m128i count)
// psrad
// __m128i _mm_sra_epi32 (__m128i a, __m128i count)
// psraw
// __m128i _mm_srai_epi16 (__m128i a, int imm8)
// psrad
// __m128i _mm_srai_epi32 (__m128i a, int imm8)
// psrlw
// __m128i _mm_srl_epi16 (__m128i a, __m128i count)
// psrld
// __m128i _mm_srl_epi32 (__m128i a, __m128i count)
// psrlq
// __m128i _mm_srl_epi64 (__m128i a, __m128i count)

// psrlw
// __m128i _mm_srli_epi16 (__m128i a, int imm8)
#[inline]
pub fn mm_srli_epi16(a: m128i, imm8: i32) -> m128i {
    unsafe { sse2_psrli_w(a.as_i16x8(), imm8).as_m128i() }
}

// psrld
// __m128i _mm_srli_epi32 (__m128i a, int imm8)
// psrlq
// __m128i _mm_srli_epi64 (__m128i a, int imm8)

// psrldq
// __m128i _mm_srli_si128 (__m128i a, int imm8)
#[inline]
pub fn mm_srli_si128(a: m128i, imm8: i32) -> m128i {
    let zero = mm_setzero_si128().as_i8x16();
    let aa = a.as_i8x16();
    let x: i8x16 = unsafe {
        match imm8 & 0xFF {
            0x0 => simd_shuffle16(aa, zero, [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]),
            0x1 => simd_shuffle16(aa, zero, [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]),
            0x2 => simd_shuffle16(aa, zero, [ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17]),
            0x3 => simd_shuffle16(aa, zero, [ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
            0x4 => simd_shuffle16(aa, zero, [ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
            0x5 => simd_shuffle16(aa, zero, [ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
            0x6 => simd_shuffle16(aa, zero, [ 6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]),
            0x7 => simd_shuffle16(aa, zero, [ 7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]),
            0x8 => simd_shuffle16(aa, zero, [ 8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]),
            0x9 => simd_shuffle16(aa, zero, [ 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]),
            0xA => simd_shuffle16(aa, zero, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]),
            0xB => simd_shuffle16(aa, zero, [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]),
            0xC => simd_shuffle16(aa, zero, [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]),
            0xD => simd_shuffle16(aa, zero, [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]),
            0xE => simd_shuffle16(aa, zero, [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]),
            0xF => simd_shuffle16(aa, zero, [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]),
              _ => simd_shuffle16(aa, zero, [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]),
        }
    };
    x.as_m128i()
}

// movapd
// void _mm_store_pd (double* mem_addr, __m128d a)
// ...
// void _mm_store_pd1 (double* mem_addr, __m128d a)
// movsd
// void _mm_store_sd (double* mem_addr, __m128d a)
// movdqa
// void _mm_store_si128 (__m128i* mem_addr, __m128i a)
// ...
// void _mm_store1_pd (double* mem_addr, __m128d a)
// movhpd
// void _mm_storeh_pd (double* mem_addr, __m128d a)
// movq
// void _mm_storel_epi64 (__m128i* mem_addr, __m128i a)
// movlpd
// void _mm_storel_pd (double* mem_addr, __m128d a)
// ...
// void _mm_storer_pd (double* mem_addr, __m128d a)
// movupd
// void _mm_storeu_pd (double* mem_addr, __m128d a)
// movdqu
// void _mm_storeu_si128 (__m128i* mem_addr, __m128i a)
// movntpd
// void _mm_stream_pd (double* mem_addr, __m128d a)
// movntdq
// void _mm_stream_si128 (__m128i* mem_addr, __m128i a)
// movnti
// void _mm_stream_si32 (int* mem_addr, int a)
// movnti
// void _mm_stream_si64 (__int64* mem_addr, __int64 a)

// psubw
// __m128i _mm_sub_epi16 (__m128i a, __m128i b)
#[inline]
pub fn mm_sub_epi16(a: m128i, b: m128i) -> m128i {
    unsafe { simd_sub(a.as_i16x8(), b.as_i16x8()).as_m128i() }
}

// psubd
// __m128i _mm_sub_epi32 (__m128i a, __m128i b)
#[inline]
pub fn mm_sub_epi32(a: m128i, b: m128i) -> m128i {
    unsafe { simd_sub(a.as_i32x4(), b.as_i32x4()).as_m128i() }
}

// psubq
// __m128i _mm_sub_epi64 (__m128i a, __m128i b)
#[inline]
pub fn mm_sub_epi64(a: m128i, b: m128i) -> m128i {
    unsafe { simd_sub(a.as_i64x2(), b.as_i64x2()).as_m128i() }
}

// psubb
// __m128i _mm_sub_epi8 (__m128i a, __m128i b)
#[inline]
pub fn mm_sub_epi8(a: m128i, b: m128i) -> m128i {
    unsafe { simd_sub(a.as_i8x16(), b.as_i8x16()).as_m128i() }
}

// subpd
// __m128d _mm_sub_pd (__m128d a, __m128d b)
#[inline]
pub fn mm_sub_pd(a: m128d, b: m128d) -> m128d {
    unsafe { simd_sub(a.as_f64x2(), b.as_f64x2()).as_m128d() }
}

// subsd
// __m128d _mm_sub_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_sub_sd(a: m128d, b: m128d) -> m128d {
    let v = a.as_f64x2().extract(0) - b.as_f64x2().extract(0);
    a.as_f64x2().insert(0, v).as_m128d()
}

// psubq
// __m64 _mm_sub_si64 (__m64 a, __m64 b)

// psubsw
// __m128i _mm_subs_epi16 (__m128i a, __m128i b)
#[inline]
pub fn mm_subs_epi16(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_subs_epi16(a.as_i16x8(), b.as_i16x8()).as_m128i() }
}

// psubsb
// __m128i _mm_subs_epi8 (__m128i a, __m128i b)
#[inline]
pub fn mm_subs_epi8(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_subs_epi8(a.as_i8x16(), b.as_i8x16()).as_m128i() }
}

// psubusw
// __m128i _mm_subs_epu16 (__m128i a, __m128i b)
#[inline]
pub fn mm_subs_epu16(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_subs_epu16(a.as_u16x8(), b.as_u16x8()).as_m128i() }
}

// psubusb
// __m128i _mm_subs_epu8 (__m128i a, __m128i b)
#[inline]
pub fn mm_subs_epu8(a: m128i, b: m128i) -> m128i {
    unsafe { x86_mm_subs_epu8(a.as_u8x16(), b.as_u8x16()).as_m128i() }
}

// ucomisd
// int _mm_ucomieq_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_ucomieq_sd(a: m128d, b: m128d) -> i32 {
    unsafe { sse2_ucomieq_sd(a, b) }
}

// ucomisd
// int _mm_ucomige_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_ucomige_sd(a: m128d, b: m128d) -> i32 {
    unsafe { sse2_ucomige_sd(a, b) }
}

// ucomisd
// int _mm_ucomigt_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_ucomigt_sd(a: m128d, b: m128d) -> i32 {
    unsafe { sse2_ucomigt_sd(a, b) }
}

// ucomisd
// int _mm_ucomile_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_ucomile_sd(a: m128d, b: m128d) -> i32 {
    unsafe { sse2_ucomile_sd(a, b) }
}

// ucomisd
// int _mm_ucomilt_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_ucomilt_sd(a: m128d, b: m128d) -> i32 {
    unsafe { sse2_ucomilt_sd(a, b) }
}

// ucomisd
// int _mm_ucomineq_sd (__m128d a, __m128d b)
#[inline]
pub fn mm_ucomineq_sd(a: m128d, b: m128d) -> i32 {
    unsafe { sse2_ucomineq_sd(a, b) }
}

// punpckhwd
// __m128i _mm_unpackhi_epi16 (__m128i a, __m128i b)
// punpckhdq
// __m128i _mm_unpackhi_epi32 (__m128i a, __m128i b)
// punpckhqdq
// __m128i _mm_unpackhi_epi64 (__m128i a, __m128i b)
// punpckhbw
// __m128i _mm_unpackhi_epi8 (__m128i a, __m128i b)
// unpckhpd
// __m128d _mm_unpackhi_pd (__m128d a, __m128d b)
// punpcklwd
// __m128i _mm_unpacklo_epi16 (__m128i a, __m128i b)
// punpckldq
// __m128i _mm_unpacklo_epi32 (__m128i a, __m128i b)
// punpcklqdq
// __m128i _mm_unpacklo_epi64 (__m128i a, __m128i b)
// punpcklbw
// __m128i _mm_unpacklo_epi8 (__m128i a, __m128i b)
// unpcklpd
// __m128d _mm_unpacklo_pd (__m128d a, __m128d b)

// xorpd
// __m128d _mm_xor_pd (__m128d a, __m128d b)
#[inline]
pub fn mm_xor_pd(a: m128d, b: m128d) -> m128d {
    mm_xor_si128(a.as_m128i(), b.as_m128i()).as_m128d()
}

// pxor
// __m128i _mm_xor_si128 (__m128i a, __m128i b)
#[inline]
pub fn mm_xor_si128(a: m128i, b: m128i) -> m128i {
    unsafe { simd_xor(a, b) }
}

#[cfg(test)]
mod tests {
    use std;
    use super::super::*;

    #[test]
    fn test_mm_add_epi() {
        let x = mm_setr_epi8(0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF, 0x0);
        let y = mm_setr_epi8(0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1);

        let z8 = mm_add_epi8(x, y);
        let z16 = mm_add_epi16(x, y);
        let z32 = mm_add_epi32(x, y);
        let z64 = mm_add_epi64(x, y);

        assert_eq!(z8.as_i8x16().extract(0), 2);
        assert_eq!(z16.as_i16x8().extract(0), 0x0201 + 0x0101);
        assert_eq!(z32.as_i32x4().extract(0), 0x04030201 + 0x01010101);
        assert_eq!(z64.as_i64x2().extract(0), 0x0807060504030201 + 0x0101010101010101);
    }

    #[test]
    fn test_mm_arith_pd() {
        let x = mm_setr_pd(1.0, 2.0);
        let y = mm_setr_pd(2.0, 4.0);

        assert_eq!(mm_add_pd(x, y).as_f64x2().as_array(), [3.0, 6.0]);
        assert_eq!(mm_sub_pd(x, y).as_f64x2().as_array(), [-1.0, -2.0]);
        assert_eq!(mm_mul_pd(x, y).as_f64x2().as_array(), [2.0, 8.0]);
        assert_eq!(mm_div_pd(x, y).as_f64x2().as_array(), [0.5, 0.5]);
    }

    #[test]
    fn test_mm_arith_sd() {
        let x = mm_setr_pd(1.0, 2.0);
        let y = mm_setr_pd(2.0, 4.0);

        assert_eq!(mm_add_sd(x, y).as_f64x2().as_array(), [3.0, 2.0]);
        assert_eq!(mm_sub_sd(x, y).as_f64x2().as_array(), [-1.0, 2.0]);
        assert_eq!(mm_mul_sd(x, y).as_f64x2().as_array(), [2.0, 2.0]);
        assert_eq!(mm_div_sd(x, y).as_f64x2().as_array(), [0.5, 2.0]);
    }

    #[test]
    fn test_mm_ariths_16() {
        let x = mm_setr_epi16(1, 2, 0, -1, 0x7FFF, 0x7FFF, -0x8000, -0x8000);
        let y = mm_setr_epi16(3, 4, -1, -1, 0x7FFF, -0x8000, 0x7FFF, -0x8000);

        assert_eq!(mm_adds_epi16(x, y).as_i16x8().as_array(),
                   [4, 6, -1, -2, 0x7FFF, -1, -1, -0x8000]);
        assert_eq!(mm_adds_epu16(x, y).as_u16x8().as_array(),
                   [4, 6, 0xFFFF, 0xFFFF, 0xFFFE, 0xFFFF, 0xFFFF, 0xFFFF]);
        assert_eq!(mm_subs_epi16(x, y).as_i16x8().as_array(),
                   [-2, -2, 1, 0, 0, 0x7FFF, -0x8000, 0]);
        assert_eq!(mm_subs_epu16(x, y).as_u16x8().as_array(),
                   [0, 0, 0, 0, 0, 0, 1, 0]);
    }

    #[test]
    fn test_mm_ariths_8() {
        let x = mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, -1, 0x7F, 0x7F, -0x80, -0x80);
        let y = mm_setr_epi8(3, 4, 5, 6, 1, 2, 3, 4, 5, 10, -1, -1, 0x7F, -0x80, 0x7F, -0x80);

        assert_eq!(mm_adds_epi8(x, y).as_i8x16().as_array(),
                   [4, 6, 8, 10, 6, 8, 10, 12, 14, 20, -1, -2, 0x7F, -1, -1, -0x80]);
        assert_eq!(mm_adds_epu8(x, y).as_u8x16().as_array(),
                   [4, 6, 8, 10, 6, 8, 10, 12, 14, 20, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFF]);
        assert_eq!(mm_subs_epi8(x, y).as_i8x16().as_array(),
                   [-2, -2, -2, -2, 4, 4, 4, 4, 4, 0, 1, 0, 0, 0x7F, -0x80, 0]);
        assert_eq!(mm_subs_epu8(x, y).as_u8x16().as_array(),
                   [0, 0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 1, 0]);
    }

    #[test]
    fn test_mm_avg() {
        let x8 = mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let y8 = mm_setr_epi8(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18);
        let x16 = mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let y16 = mm_setr_epi16(3, 4, 5, 6, 7, 8, 9, 10);

        assert_eq!(mm_avg_epu16(x16, y16).as_i16x8().as_array(),
                   [2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(mm_avg_epu8(x8, y8).as_i8x16().as_array(),
                   [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]);
    }

    #[test]
    fn test_cmp_int() {
        let x8 = mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let y8 = mm_setr_epi8(1, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let x16 = mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let y16 = mm_setr_epi16(1, 3, 1, 4, 5, 6, 7, 8);
        let x32 = mm_setr_epi32(1, 2, 3, 4);
        let y32 = mm_setr_epi32(1, 3, 1, 4);

        assert_eq!(mm_cmpeq_epi8(x8, y8).as_i8x16().as_array(),
                   [!0, 0, 0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0]);
        assert_eq!(mm_cmpgt_epi8(x8, y8).as_i8x16().as_array(),
                   [0, 0, !0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(mm_cmplt_epi8(x8, y8).as_i8x16().as_array(),
                   [0, !0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(mm_cmpeq_epi16(x16, y16).as_i16x8().as_array(),
                   [!0, 0, 0, !0, !0, !0, !0, !0]);
        assert_eq!(mm_cmpgt_epi16(x16, y16).as_i16x8().as_array(),
                   [0, 0, !0, 0, 0, 0, 0, 0]);
        assert_eq!(mm_cmplt_epi16(x16, y16).as_i16x8().as_array(),
                   [0, !0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(mm_cmpeq_epi32(x32, y32).as_i32x4().as_array(),
                   [!0, 0, 0, !0]);
        assert_eq!(mm_cmpgt_epi32(x32, y32).as_i32x4().as_array(),
                   [0, 0, !0, 0]);
        assert_eq!(mm_cmplt_epi32(x32, y32).as_i32x4().as_array(),
                   [0, !0, 0, 0]);
    }

    #[test]
    fn test_cmp_pd() {
        let x = mm_setr_pd(1.0, 2.0);
        let y = mm_setr_pd(2.0, 2.0);
        let z = mm_setr_pd(std::f64::NAN, std::f64::NAN);

        let xy_eq = mm_cmpeq_pd(x, y).as_m128i().as_i64x2();
        let xy_ge = mm_cmpge_pd(x, y).as_m128i().as_i64x2();
        let xy_gt = mm_cmpgt_pd(x, y).as_m128i().as_i64x2();
        let xy_le = mm_cmple_pd(x, y).as_m128i().as_i64x2();
        let xy_lt = mm_cmplt_pd(x, y).as_m128i().as_i64x2();
        let xy_ne = mm_cmpneq_pd(x, y).as_m128i().as_i64x2();
        let xy_nge = mm_cmpnge_pd(x, y).as_m128i().as_i64x2();
        let xy_ngt = mm_cmpngt_pd(x, y).as_m128i().as_i64x2();
        let xy_nle = mm_cmpnle_pd(x, y).as_m128i().as_i64x2();
        let xy_nlt = mm_cmpnlt_pd(x, y).as_m128i().as_i64x2();
        let xy_ord = mm_cmpord_pd(x, y).as_m128i().as_i64x2();
        let xy_uno = mm_cmpunord_pd(x, y).as_m128i().as_i64x2();

        assert_eq!(xy_eq.as_array(),  [ 0, !0]);
        assert_eq!(xy_ge.as_array(),  [ 0, !0]);
        assert_eq!(xy_gt.as_array(),  [ 0,  0]);
        assert_eq!(xy_le.as_array(),  [!0, !0]);
        assert_eq!(xy_lt.as_array(),  [!0,  0]);
        assert_eq!(xy_ne.as_array(),  [!0,  0]);
        assert_eq!(xy_nge.as_array(), [!0,  0]);
        assert_eq!(xy_ngt.as_array(), [!0, !0]);
        assert_eq!(xy_nle.as_array(), [ 0,  0]);
        assert_eq!(xy_nlt.as_array(), [ 0, !0]);
        assert_eq!(xy_ord.as_array(), [!0, !0]);
        assert_eq!(xy_uno.as_array(), [ 0,  0]);

        let yx_eq = mm_cmpeq_pd(y, x).as_m128i().as_i64x2();
        let yx_ge = mm_cmpge_pd(y, x).as_m128i().as_i64x2();
        let yx_gt = mm_cmpgt_pd(y, x).as_m128i().as_i64x2();
        let yx_le = mm_cmple_pd(y, x).as_m128i().as_i64x2();
        let yx_lt = mm_cmplt_pd(y, x).as_m128i().as_i64x2();
        let yx_ne = mm_cmpneq_pd(y, x).as_m128i().as_i64x2();
        let yx_nge = mm_cmpnge_pd(y, x).as_m128i().as_i64x2();
        let yx_ngt = mm_cmpngt_pd(y, x).as_m128i().as_i64x2();
        let yx_nle = mm_cmpnle_pd(y, x).as_m128i().as_i64x2();
        let yx_nlt = mm_cmpnlt_pd(y, x).as_m128i().as_i64x2();
        let yx_ord = mm_cmpord_pd(y, x).as_m128i().as_i64x2();
        let yx_uno = mm_cmpunord_pd(y, x).as_m128i().as_i64x2();

        assert_eq!(yx_eq.as_array(), [ 0, !0]);
        assert_eq!(yx_ge.as_array(), [!0, !0]);
        assert_eq!(yx_gt.as_array(), [!0,  0]);
        assert_eq!(yx_le.as_array(), [ 0, !0]);
        assert_eq!(yx_lt.as_array(), [ 0,  0]);
        assert_eq!(yx_ne.as_array(), [!0,  0]);
        assert_eq!(yx_nge.as_array(), [ 0,  0]);
        assert_eq!(yx_ngt.as_array(), [ 0, !0]);
        assert_eq!(yx_nle.as_array(), [!0,  0]);
        assert_eq!(yx_nlt.as_array(), [!0, !0]);
        assert_eq!(yx_ord.as_array(), [!0, !0]);
        assert_eq!(yx_uno.as_array(), [ 0,  0]);

        let xz_eq = mm_cmpeq_pd(x, z).as_m128i().as_i64x2();
        let xz_ge = mm_cmpge_pd(x, z).as_m128i().as_i64x2();
        let xz_gt = mm_cmpgt_pd(x, z).as_m128i().as_i64x2();
        let xz_le = mm_cmple_pd(x, z).as_m128i().as_i64x2();
        let xz_lt = mm_cmplt_pd(x, z).as_m128i().as_i64x2();
        let xz_ne = mm_cmpneq_pd(x, z).as_m128i().as_i64x2();
        let xz_nge = mm_cmpnge_pd(x, z).as_m128i().as_i64x2();
        let xz_ngt = mm_cmpngt_pd(x, z).as_m128i().as_i64x2();
        let xz_nle = mm_cmpnle_pd(x, z).as_m128i().as_i64x2();
        let xz_nlt = mm_cmpnlt_pd(x, z).as_m128i().as_i64x2();
        let xz_ord = mm_cmpord_pd(x, z).as_m128i().as_i64x2();
        let xz_uno = mm_cmpunord_pd(x, z).as_m128i().as_i64x2();

        assert_eq!(xz_eq.as_array(), [ 0,  0]);
        assert_eq!(xz_ge.as_array(), [ 0,  0]);
        assert_eq!(xz_gt.as_array(), [ 0,  0]);
        assert_eq!(xz_le.as_array(), [ 0,  0]);
        assert_eq!(xz_lt.as_array(), [ 0,  0]);
        assert_eq!(xz_ne.as_array(), [!0, !0]);
        assert_eq!(xz_nge.as_array(), [!0, !0]);
        assert_eq!(xz_ngt.as_array(), [!0, !0]);
        assert_eq!(xz_nle.as_array(), [!0, !0]);
        assert_eq!(xz_nlt.as_array(), [!0, !0]);
        assert_eq!(xz_ord.as_array(), [ 0,  0]);
        assert_eq!(xz_uno.as_array(), [!0, !0]);
    }

    #[test]
    fn test_cmp_sd() {
        let x = mm_setr_pd(1.0, 2.0);
        let y = mm_setr_pd(2.0, 2.0);
        let z = mm_setr_pd(std::f64::NAN, std::f64::NAN);

        let x1 = x.as_m128i().as_i64x2().extract(1);
        let y1 = y.as_m128i().as_i64x2().extract(1);

        let xy_eq = mm_cmpeq_sd(x, y).as_m128i().as_i64x2();
        let xy_ge = mm_cmpge_sd(x, y).as_m128i().as_i64x2();
        let xy_gt = mm_cmpgt_sd(x, y).as_m128i().as_i64x2();
        let xy_le = mm_cmple_sd(x, y).as_m128i().as_i64x2();
        let xy_lt = mm_cmplt_sd(x, y).as_m128i().as_i64x2();
        let xy_ne = mm_cmpneq_sd(x, y).as_m128i().as_i64x2();
        let xy_nge = mm_cmpnge_sd(x, y).as_m128i().as_i64x2();
        let xy_ngt = mm_cmpngt_sd(x, y).as_m128i().as_i64x2();
        let xy_nle = mm_cmpnle_sd(x, y).as_m128i().as_i64x2();
        let xy_nlt = mm_cmpnlt_sd(x, y).as_m128i().as_i64x2();
        let xy_ord = mm_cmpord_sd(x, y).as_m128i().as_i64x2();
        let xy_uno = mm_cmpunord_sd(x, y).as_m128i().as_i64x2();

        assert_eq!(xy_eq.as_array(),  [ 0, x1]);
        assert_eq!(xy_ge.as_array(),  [ 0, x1]);
        assert_eq!(xy_gt.as_array(),  [ 0, x1]);
        assert_eq!(xy_le.as_array(),  [!0, x1]);
        assert_eq!(xy_lt.as_array(),  [!0, x1]);
        assert_eq!(xy_ne.as_array(),  [!0, x1]);
        assert_eq!(xy_nge.as_array(), [!0, x1]);
        assert_eq!(xy_ngt.as_array(), [!0, x1]);
        assert_eq!(xy_nle.as_array(), [ 0, x1]);
        assert_eq!(xy_nlt.as_array(), [ 0, x1]);
        assert_eq!(xy_ord.as_array(), [!0, x1]);
        assert_eq!(xy_uno.as_array(), [ 0, x1]);

        let yx_eq = mm_cmpeq_sd(y, x).as_m128i().as_i64x2();
        let yx_ge = mm_cmpge_sd(y, x).as_m128i().as_i64x2();
        let yx_gt = mm_cmpgt_sd(y, x).as_m128i().as_i64x2();
        let yx_le = mm_cmple_sd(y, x).as_m128i().as_i64x2();
        let yx_lt = mm_cmplt_sd(y, x).as_m128i().as_i64x2();
        let yx_ne = mm_cmpneq_sd(y, x).as_m128i().as_i64x2();
        let yx_nge = mm_cmpnge_sd(y, x).as_m128i().as_i64x2();
        let yx_ngt = mm_cmpngt_sd(y, x).as_m128i().as_i64x2();
        let yx_nle = mm_cmpnle_sd(y, x).as_m128i().as_i64x2();
        let yx_nlt = mm_cmpnlt_sd(y, x).as_m128i().as_i64x2();
        let yx_ord = mm_cmpord_sd(y, x).as_m128i().as_i64x2();
        let yx_uno = mm_cmpunord_sd(y, x).as_m128i().as_i64x2();

        assert_eq!(yx_eq.as_array(),  [ 0, y1]);
        assert_eq!(yx_ge.as_array(),  [!0, y1]);
        assert_eq!(yx_gt.as_array(),  [!0, y1]);
        assert_eq!(yx_le.as_array(),  [ 0, y1]);
        assert_eq!(yx_lt.as_array(),  [ 0, y1]);
        assert_eq!(yx_ne.as_array(),  [!0, y1]);
        assert_eq!(yx_nge.as_array(), [ 0, y1]);
        assert_eq!(yx_ngt.as_array(), [ 0, y1]);
        assert_eq!(yx_nle.as_array(), [!0, y1]);
        assert_eq!(yx_nlt.as_array(), [!0, y1]);
        assert_eq!(yx_ord.as_array(), [!0, y1]);
        assert_eq!(yx_uno.as_array(), [ 0, y1]);

        let xz_eq = mm_cmpeq_sd(x, z).as_m128i().as_i64x2();
        let xz_ge = mm_cmpge_sd(x, z).as_m128i().as_i64x2();
        let xz_gt = mm_cmpgt_sd(x, z).as_m128i().as_i64x2();
        let xz_le = mm_cmple_sd(x, z).as_m128i().as_i64x2();
        let xz_lt = mm_cmplt_sd(x, z).as_m128i().as_i64x2();
        let xz_ne = mm_cmpneq_sd(x, z).as_m128i().as_i64x2();
        let xz_nge = mm_cmpnge_sd(x, z).as_m128i().as_i64x2();
        let xz_ngt = mm_cmpngt_sd(x, z).as_m128i().as_i64x2();
        let xz_nle = mm_cmpnle_sd(x, z).as_m128i().as_i64x2();
        let xz_nlt = mm_cmpnlt_sd(x, z).as_m128i().as_i64x2();
        let xz_ord = mm_cmpord_sd(x, z).as_m128i().as_i64x2();
        let xz_uno = mm_cmpunord_sd(x, z).as_m128i().as_i64x2();

        assert_eq!(xz_eq.as_array(),  [ 0, x1]);
        assert_eq!(xz_ge.as_array(),  [ 0, x1]);
        assert_eq!(xz_gt.as_array(),  [ 0, x1]);
        assert_eq!(xz_le.as_array(),  [ 0, x1]);
        assert_eq!(xz_lt.as_array(),  [ 0, x1]);
        assert_eq!(xz_ne.as_array(),  [!0, x1]);
        assert_eq!(xz_nge.as_array(), [!0, x1]);
        assert_eq!(xz_ngt.as_array(), [!0, x1]);
        assert_eq!(xz_nle.as_array(), [!0, x1]);
        assert_eq!(xz_nlt.as_array(), [!0, x1]);
        assert_eq!(xz_ord.as_array(), [ 0, x1]);
        assert_eq!(xz_uno.as_array(), [!0, x1]);
    }

    #[test]
    fn test_mm_comi_sd() {
        let x = mm_setr_pd(1.0, 2.0);
        let y = mm_setr_pd(2.0, 2.0);

        assert_eq!(mm_comieq_sd(x, x), 1);
        assert_eq!(mm_comige_sd(x, x), 1);
        assert_eq!(mm_comigt_sd(x, x), 0);
        assert_eq!(mm_comile_sd(x, x), 1);
        assert_eq!(mm_comilt_sd(x, x), 0);
        assert_eq!(mm_comineq_sd(x, x), 0);

        assert_eq!(mm_ucomieq_sd(x, x), 1);
        assert_eq!(mm_ucomige_sd(x, x), 1);
        assert_eq!(mm_ucomigt_sd(x, x), 0);
        assert_eq!(mm_ucomile_sd(x, x), 1);
        assert_eq!(mm_ucomilt_sd(x, x), 0);
        assert_eq!(mm_ucomineq_sd(x, x), 0);

        assert_eq!(mm_comieq_sd(x, y), 0);
        assert_eq!(mm_comige_sd(x, y), 0);
        assert_eq!(mm_comigt_sd(x, y), 0);
        assert_eq!(mm_comile_sd(x, y), 1);
        assert_eq!(mm_comilt_sd(x, y), 1);
        assert_eq!(mm_comineq_sd(x, y), 1);

        assert_eq!(mm_ucomieq_sd(x, y), 0);
        assert_eq!(mm_ucomige_sd(x, y), 0);
        assert_eq!(mm_ucomigt_sd(x, y), 0);
        assert_eq!(mm_ucomile_sd(x, y), 1);
        assert_eq!(mm_ucomilt_sd(x, y), 1);
        assert_eq!(mm_ucomineq_sd(x, y), 1);

        assert_eq!(mm_comieq_sd(y, x), 0);
        assert_eq!(mm_comige_sd(y, x), 1);
        assert_eq!(mm_comigt_sd(y, x), 1);
        assert_eq!(mm_comile_sd(y, x), 0);
        assert_eq!(mm_comilt_sd(y, x), 0);
        assert_eq!(mm_comineq_sd(y, x), 1);

        assert_eq!(mm_ucomieq_sd(y, x), 0);
        assert_eq!(mm_ucomige_sd(y, x), 1);
        assert_eq!(mm_ucomigt_sd(y, x), 1);
        assert_eq!(mm_ucomile_sd(y, x), 0);
        assert_eq!(mm_ucomilt_sd(y, x), 0);
        assert_eq!(mm_ucomineq_sd(y, x), 1);

        // TODO(mayah): Hmm, hitting this behavior change?
        // https://llvm.org/bugs/show_bug.cgi?id=28510
        //
        // let z = mm_setr_pd(std::f64::NAN, std::f64::NAN);
        // assert_eq!(mm_comieq_sd(x, z), 1);
        // assert_eq!(mm_comige_sd(x, z), 1);
        // assert_eq!(mm_comigt_sd(x, z), 1);
        // assert_eq!(mm_comile_sd(x, z), 1);
        // assert_eq!(mm_comilt_sd(x, z), 1);
        // assert_eq!(mm_comineq_sd(x, z), 1);
        //
        // assert_eq!(mm_ucomieq_sd(x, z), 1);
        // assert_eq!(mm_ucomige_sd(x, z), 1);
        // assert_eq!(mm_ucomigt_sd(x, z), 1);
        // assert_eq!(mm_ucomile_sd(x, z), 1);
        // assert_eq!(mm_ucomilt_sd(x, z), 1);
        // assert_eq!(mm_ucomineq_sd(x, z), 1);
    }

    #[test]
    fn test_mm_sub_epi() {
        let x = mm_setr_epi8(0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF, 0x0);
        let y = mm_setr_epi8(0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1);

        let z8 = mm_sub_epi8(x, y);
        let z16 = mm_sub_epi16(x, y);
        let z32 = mm_sub_epi32(x, y);
        let z64 = mm_sub_epi64(x, y);

        assert_eq!(z8.as_i8x16().extract(0), 0);
        assert_eq!(z16.as_i16x8().extract(0), 0x0201 - 0x0101);
        assert_eq!(z32.as_i32x4().extract(0), 0x04030201 - 0x01010101);
        assert_eq!(z64.as_i64x2().extract(0), 0x0807060504030201 - 0x0101010101010101);
    }

    #[test]
    fn test_mm_set_int() {
        assert_eq!(mm_setzero_si128().as_i64x2().as_array(), [0, 0]);

        assert_eq!(mm_set_epi32(1, 2, 3, 4).as_i32x4().as_array(), [4, 3, 2, 1]);
        assert_eq!(mm_setr_epi32(1, 2, 3, 4).as_i32x4().as_array(), [1, 2, 3, 4]);
        assert_eq!(mm_set_epi16(1, 2, 3, 4, 5, 6, 7, 8).as_i16x8().as_array(), [8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8).as_i16x8().as_array(), [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(mm_set_epi64x(0x3, 0xF).as_i64x2().as_array(), [0xF, 0x3]);

        assert_eq!(mm_set1_epi16(1).as_i16x8().as_array(), [1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(mm_set1_epi32(1).as_i32x4().as_array(), [1, 1, 1, 1]);
        assert_eq!(mm_set1_epi64(1).as_i64x2().as_array(), [1, 1]);
        assert_eq!(mm_set1_epi8(1).as_i8x16().as_array(),
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_mm_set_pd() {
        assert_eq!(mm_setzero_pd().as_f64x2().as_array(), [0.0, 0.0]);

        assert_eq!(mm_set_pd(1.0, 2.0).as_f64x2().as_array(), [2.0, 1.0]);
        assert_eq!(mm_setr_pd(1.0, 2.0).as_f64x2().as_array(), [1.0, 2.0]);
        assert_eq!(mm_set_pd1(1.0).as_f64x2().as_array(), [1.0, 1.0]);
        assert_eq!(mm_set_sd(1.0).as_f64x2().as_array(), [1.0, 0.0]);
        assert_eq!(mm_set1_pd(1.0).as_f64x2().as_array(), [1.0, 1.0]);
    }

    #[test]
    fn test_mm_logic_si128() {
        let x = mm_setr_epi32(0x3F, 0x7E, 0x13, 0xFF);
        let y = mm_setr_epi32(0x53, 0x8C, 0xFF, 0x17);

        assert_eq!(mm_and_si128(x, y).as_i32x4().as_array(), [0x3F & 0x53, 0x7E & 0x8C, 0x13 & 0xFF, 0xFF & 0x17]);
        assert_eq!(mm_or_si128(x, y).as_i32x4().as_array(), [0x3F | 0x53, 0x7E | 0x8C, 0x13 | 0xFF, 0xFF | 0x17]);
        assert_eq!(mm_xor_si128(x, y).as_i32x4().as_array(), [0x3F ^ 0x53, 0x7E ^ 0x8C, 0x13 ^ 0xFF, 0xFF ^ 0x17]);
        assert_eq!(mm_andnot_si128(x, y).as_i32x4().as_array(), [!0x3F & 0x53, !0x7E & 0x8C, !0x13 & 0xFF, !0xFF & 0x17]);
    }

    #[test]
    fn test_mm_logic_pd() {
        let x = mm_setr_epi32(0x3F, 0x7E, 0x13, 0xFF).as_m128d();
        let y = mm_setr_epi32(0x53, 0x8C, 0xFF, 0x17).as_m128d();

        assert_eq!(mm_and_pd(x, y).as_m128i().as_i32x4().as_array(), [0x3F & 0x53, 0x7E & 0x8C, 0x13 & 0xFF, 0xFF & 0x17]);
        assert_eq!(mm_or_pd(x, y).as_m128i().as_i32x4().as_array(), [0x3F | 0x53, 0x7E | 0x8C, 0x13 | 0xFF, 0xFF | 0x17]);
        assert_eq!(mm_xor_pd(x, y).as_m128i().as_i32x4().as_array(), [0x3F ^ 0x53, 0x7E ^ 0x8C, 0x13 ^ 0xFF, 0xFF ^ 0x17]);
        assert_eq!(mm_andnot_pd(x, y).as_m128i().as_i32x4().as_array(), [!0x3F & 0x53, !0x7E & 0x8C, !0x13 & 0xFF, !0xFF & 0x17]);
    }

    #[test]
    fn test_mm_slli_epi16() {
        let x = mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let x0 = mm_slli_epi16(x, 0).as_i16x8();
        let x1 = mm_slli_epi16(x, 1).as_i16x8();
        let x2 = mm_slli_epi16(x, 2).as_i16x8();

        for i in 0 .. 8 {
            assert_eq!(x0.extract(i) as usize, (i + 1) << 0);
            assert_eq!(x1.extract(i) as usize, (i + 1) << 1);
            assert_eq!(x2.extract(i) as usize, (i + 1) << 2);
        }
    }

    #[test]
    fn test_mm_srli_epi16() {
        let x = mm_setr_epi16(11, 12, 13, 14, 15, 16, 17, 18);
        let x0 = mm_srli_epi16(x, 0).as_i16x8();
        let x1 = mm_srli_epi16(x, 1).as_i16x8();
        let x2 = mm_srli_epi16(x, 2).as_i16x8();

        for i in 0 .. 8 {
            assert_eq!(x0.extract(i) as usize, (i + 11) >> 0);
            assert_eq!(x1.extract(i) as usize, (i + 11) >> 1);
            assert_eq!(x2.extract(i) as usize, (i + 11) >> 2);
        }
    }

    #[test]
    fn test_mm_slli_si128() {
        let x = mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let x0 = mm_slli_si128(x, 0).as_i8x16();
        let x1 = mm_slli_si128(x, 1).as_i8x16();
        let x2 = mm_slli_si128(x, 2).as_i8x16();

        let bx0 = mm_bslli_si128(x, 0).as_i8x16();
        let bx1 = mm_bslli_si128(x, 1).as_i8x16();
        let bx2 = mm_bslli_si128(x, 2).as_i8x16();

        for i in 0 .. 16 {
            assert_eq!(x0.extract(i) as usize, i + 1);
            assert_eq!(x1.extract(i) as usize, i);
            assert_eq!(x2.extract(i) as usize, if i >= 1 { i - 1 } else { 0 });
            assert_eq!(bx0.extract(i) as usize, i + 1);
            assert_eq!(bx1.extract(i) as usize, i);
            assert_eq!(bx2.extract(i) as usize, if i >= 1 { i - 1 } else { 0 });
        }
    }

    #[test]
    fn test_mm_srli_si128() {
        let x = mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let x0 = mm_srli_si128(x, 0).as_i8x16();
        let x1 = mm_srli_si128(x, 1).as_i8x16();
        let x2 = mm_srli_si128(x, 2).as_i8x16();
        let bx0 = mm_bsrli_si128(x, 0).as_i8x16();
        let bx1 = mm_bsrli_si128(x, 1).as_i8x16();
        let bx2 = mm_bsrli_si128(x, 2).as_i8x16();
        for i in 0 .. 16 {
            assert_eq!(x0.extract(i) as usize, i + 1);
            assert_eq!(x1.extract(i) as usize, if i + 2 >= 17 { 0 } else { i + 2 } );
            assert_eq!(x2.extract(i) as usize, if i + 3 >= 17 { 0 } else { i + 3 } );
            assert_eq!(bx0.extract(i) as usize, i + 1);
            assert_eq!(bx1.extract(i) as usize, if i + 2 >= 17 { 0 } else { i + 2 } );
            assert_eq!(bx2.extract(i) as usize, if i + 3 >= 17 { 0 } else { i + 3 } );
        }
    }
}
