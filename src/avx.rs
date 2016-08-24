use std;
use super::*;
use super::{simd_add, simd_sub, simd_mul, simd_div,
            simd_and, simd_or, simd_xor,
            simd_shuffle2, simd_shuffle4, simd_shuffle8};

extern {
    #[link_name = "llvm.x86.avx.blendv.pd.256"]
    fn avx_blendv_pd_256(a: m256d, b: m256d, c: m256d) -> m256d;
    #[link_name = "llvm.x86.avx.blendv.ps.256"]
    fn avx_blendv_ps_256(a: m256, b: m256, c: m256) -> m256;

    #[link_name = "llvm.x86.avx.vbroadcastf128.pd.256"]
    fn avx_vbroadcastf128_pd_256(a: *const u8) -> m256d;
    #[link_name = "llvm.x86.avx.vbroadcastf128.ps.256"]
    fn avx_vbroadcastf128_ps_256(a: *const u8) -> m256;

    // #[link_name = "llvm.x86.sse2.cmp.ps"]
    // fn sse2_cmp_ps(a: m128, b: m128, c: i8) -> m128;
    // #[link_name = "llvm.x86.sse2.cmp.pd"]
    // fn sse2_cmp_pd(a: m128d, b: m128d, c: i8) -> m128d;

    #[link_name = "llvm.x86.avx.dp.ps.256"]
    fn avx_dp_ps_256(a: m256, b: m256, c: u8) -> m256;

    #[link_name = "llvm.x86.avx.ldu.dq.256"]
    fn avx_ldu_dq_256(a: *mut u8) -> i8x32;

    #[link_name = "llvm.x86.avx.maskload.pd"]
    fn avx_maskload_pd(a: *const i8, b: i64x2) -> m128d;
    #[link_name = "llvm.x86.avx.maskload.ps"]
    fn avx_maskload_ps(a: *const i8, b: i32x4) -> m128;
    #[link_name = "llvm.x86.avx.maskload.pd.256"]
    fn avx_maskload_pd_256(a: *const i8, b: i64x4) -> m256d;
    #[link_name = "llvm.x86.avx.maskload.ps.256"]
    fn avx_maskload_ps_256(a: *const i8, b: i32x8) -> m256;
    #[link_name = "llvm.x86.avx.maskstore.pd"]
    fn avx_maskstore_pd(a: *mut i8, b: i64x2, c: m128d) -> ();
    #[link_name = "llvm.x86.avx.maskstore.ps"]
    fn avx_maskstore_ps(a: *mut i8, b: i32x4, c: m128) -> ();
    #[link_name = "llvm.x86.avx.maskstore.pd.256"]
    fn avx_maskstore_pd_256(a: *mut i8, b: i64x4, c: m256d) -> ();
    #[link_name = "llvm.x86.avx.maskstore.ps.256"]
    fn avx_maskstore_ps_256(a: *mut i8, b: i32x8, c: m256) -> ();

    #[link_name = "llvm.x86.avx.movmsk.pd.256"]
    fn avx_movmsk_pd_256(a: m256d) -> i32;
    #[link_name = "llvm.x86.avx.movmsk.ps.256"]
    fn avx_movmsk_ps_256(a: m256) -> i32;

    #[link_name = "llvm.x86.avx.round.ps.256"]
    fn avx_round_ps(a: m256, b: i32) -> m256;
    #[link_name = "llvm.x86.avx.round.pd.256"]
    fn avx_round_pd(a: m256d, b: i32) -> m256d;

    #[link_name = "llvm.x86.avx.vzeroall"]
    fn avx_vzeroall();
    #[link_name = "llvm.x86.avx.vzeroupper"]
    fn avx_vzeroupper();
}

extern "platform-intrinsic" {
    fn x86_mm256_addsub_pd(x: m256d, y: m256d) -> m256d;
    fn x86_mm256_addsub_ps(x: m256, y: m256) -> m256;
    fn x86_mm256_hadd_pd(x: m256d, y: m256d) -> m256d;
    fn x86_mm256_hadd_ps(x: m256, y: m256) -> m256;
    fn x86_mm256_hsub_pd(x: m256d, y: m256d) -> m256d;
    fn x86_mm256_hsub_ps(x: m256, y: m256) -> m256;

    // fn x86_mm256_cmp_pd(a: m256d, b: m256d, c: i8) -> m256d;
    // fn x86_mm256_cmp_ps(a: m256, b: m256, c: i8) -> m256;

    // TODO(mayah): It looks the type of these functions in rust
    // and llvm are mismatching? If I use i64x2, LLVM shows an error.
    // If I use f64x2, rust shows an error.
    // fn x86_mm_maskload_pd(x: *const f64, y: i64x2) -> m128d;
    // fn x86_mm_maskload_ps(x: *const f32, y: i32x4) -> m128;
    // fn x86_mm256_maskload_pd(x: *const f64, y: i64x4) -> m256d;
    // fn x86_mm256_maskload_ps(x: *const f32, y: i32x8) -> m256;
    // fn x86_mm_maskstore_pd(x: *mut f64, y: i64x2, z: m128d);
    // fn x86_mm_maskstore_ps(x: *mut f32, y: i32x4, z: m128);
    // fn x86_mm256_maskstore_pd(x: *mut f64, y: i64x4, z: m256d);
    // fn x86_mm256_maskstore_ps(x: *mut f32, y: i32x8, z: m256);

    fn x86_mm256_max_ps(x: m256, y: m256) -> m256;
    fn x86_mm256_max_pd(x: m256d, y: m256d) -> m256d;
    fn x86_mm256_min_ps(x: m256, y: m256) -> m256;
    fn x86_mm256_min_pd(x: m256d, y: m256d) -> m256d;

    fn x86_mm256_rsqrt_ps(x: m256) -> m256;
    fn x86_mm256_sqrt_ps(x: m256) -> m256;
    fn x86_mm256_sqrt_pd(x: m256d) -> m256d;
    fn x86_mm256_rcp_ps(x: m256) -> m256;

    fn x86_mm_testc_ps(x: m128, y: m128) -> i32;
    fn x86_mm256_testc_ps(x: m256, y: m256) -> i32;
    fn x86_mm_testc_pd(x: m128d, y: m128d) -> i32;
    fn x86_mm256_testc_pd(x: m256d, y: m256d) -> i32;
    fn x86_mm256_testc_si256(x: u64x4, y: u64x4) -> i32;
    fn x86_mm_testnzc_ps(x: m128, y: m128) -> i32;
    fn x86_mm256_testnzc_ps(x: m256, y: m256) -> i32;
    fn x86_mm_testnzc_pd(x: m128d, y: m128d) -> i32;
    fn x86_mm256_testnzc_pd(x: m256d, y: m256d) -> i32;
    fn x86_mm256_testnzc_si256(x: u64x4, y: u64x4) -> i32;
    fn x86_mm_testz_ps(x: m128, y: m128) -> i32;
    fn x86_mm256_testz_ps(x: m256, y: m256) -> i32;
    fn x86_mm_testz_pd(x: m128d, y: m128d) -> i32;
    fn x86_mm256_testz_pd(x: m256d, y: m256d) -> i32;
    fn x86_mm256_testz_si256(x: u64x4, y: u64x4) -> i32;
}

pub const CMP_EQ_OQ: i32 = 0x00;
pub const CMP_LT_OS: i32 = 0x01;
pub const CMP_LE_OS: i32 = 0x02;
pub const CMP_UNORD_Q: i32 = 0x03;
pub const CMP_NEQ_UQ: i32 = 0x04;
pub const CMP_NLT_US: i32 = 0x05;
pub const CMP_NLE_US: i32 = 0x06;
pub const CMP_ORD_Q: i32 = 0x07;
pub const CMP_EQ_UQ: i32 = 0x08;
pub const CMP_NGE_US: i32 = 0x09;
pub const CMP_NGT_US: i32 = 0x0a;
pub const CMP_FALSE_OQ: i32 = 0x0b;
pub const CMP_NEQ_OQ: i32 = 0x0c;
pub const CMP_GE_OS: i32 = 0x0d;
pub const CMP_GT_OS: i32 = 0x0e;
pub const CMP_TRUE_UQ: i32 = 0x0f;
pub const CMP_EQ_OS: i32 = 0x10;
pub const CMP_LT_OQ: i32 = 0x11;
pub const CMP_LE_OQ: i32 = 0x12;
pub const CMP_UNORD_S: i32 = 0x13;
pub const CMP_NEQ_US: i32 = 0x14;
pub const CMP_NLT_UQ: i32 = 0x15;
pub const CMP_NLE_UQ: i32 = 0x16;
pub const CMP_ORD_S: i32 = 0x17;
pub const CMP_EQ_US: i32 = 0x18;
pub const CMP_NGE_UQ: i32 = 0x19;
pub const CMP_NGT_UQ: i32 = 0x1a;
pub const CMP_FALSE_OS: i32 = 0x1b;
pub const CMP_NEQ_OS: i32 = 0x1c;
pub const CMP_GE_OQ: i32 = 0x1d;
pub const CMP_GT_OQ: i32 = 0x1e;
pub const CMP_TRUE_US: i32 = 0x1f;

// vaddpd
// __m256d _mm256_add_pd (__m256d a, __m256d b)
#[inline]
pub fn mm256_add_pd(a: m256d, b: m256d) -> m256d {
    unsafe { simd_add(a, b) }
}

// vaddps
// __m256 _mm256_add_ps (__m256 a, __m256 b)
#[inline]
pub fn mm256_add_ps(a: m256, b: m256) -> m256 {
    unsafe { simd_add(a, b) }
}

// vaddsubpd
// __m256d _mm256_addsub_pd (__m256d a, __m256d b)
#[inline]
pub fn mm256_addsub_pd(a: m256d, b: m256d) -> m256d {
    unsafe { x86_mm256_addsub_pd(a, b) }
}

// vaddsubps
// __m256 _mm256_addsub_ps (__m256 a, __m256 b)
#[inline]
pub fn mm256_addsub_ps(a: m256, b: m256) -> m256 {
    unsafe { x86_mm256_addsub_ps(a, b) }
}

// vandpd
// __m256d _mm256_and_pd (__m256d a, __m256d b)
#[inline]
pub fn mm256_and_pd(a: m256d, b: m256d) -> m256d {
    let ai = a.as_m256i();
    let bi = b.as_m256i();
    unsafe { simd_and(ai, bi).as_m256d() }
}

// vandps
// __m256 _mm256_and_ps (__m256 a, __m256 b)
#[inline]
pub fn mm256_and_ps(a: m256, b: m256) -> m256 {
    let ai = a.as_m256i();
    let bi = b.as_m256i();
    unsafe { simd_and(ai, bi).as_m256() }
}

// vandnpd
// __m256d _mm256_andnot_pd (__m256d a, __m256d b)
#[inline]
pub fn mm256_andnot_pd(a: m256d, b: m256d) -> m256d {
    let ones = i64x4(!0, !0, !0, !0).as_m256i();
    let ai = a.as_m256i();
    let bi = b.as_m256i();
    unsafe { simd_and(simd_xor(ai, ones), bi).as_m256d() }
}

// vandnps
// __m256 _mm256_andnot_ps (__m256 a, __m256 b)
#[inline]
pub fn mm256_andnot_ps(a: m256, b: m256) -> m256 {
    let ones = i64x4(!0, !0, !0, !0).as_m256i();
    let ai = a.as_m256i();
    let bi = b.as_m256i();
    unsafe { simd_and(simd_xor(ai, ones), bi).as_m256() }
}

// vblendpd
// __m256d _mm256_blend_pd (__m256d a, __m256d b, const int imm8)
#[inline]
pub fn mm256_blend_pd(a: m256d, b: m256d, imm8: i32) -> m256d {
    blend_shuffle4!(a, b, imm8)
}

// vblendps
// __m256 _mm256_blend_ps (__m256 a, __m256 b, const int imm8)
#[inline]
pub fn mm256_blend_ps(a: m256, b: m256, imm8: i32) -> m256 {
    blend_shuffle8!(a, b, imm8)
}

// vblendvpd
// __m256d _mm256_blendv_pd (__m256d a, __m256d b, __m256d mask)
#[inline]
pub fn mm256_blendv_pd(a: m256d, b: m256d, mask: m256d) -> m256d {
    unsafe { avx_blendv_pd_256(a, b, mask) }
}

// vblendvps
// __m256 _mm256_blendv_ps (__m256 a, __m256 b, __m256 mask)
#[inline]
pub fn mm256_blendv_ps(a: m256, b: m256, mask: m256) -> m256 {
    unsafe { avx_blendv_ps_256(a, b, mask) }
}

// vbroadcastf128
// __m256d _mm256_broadcast_pd (__m128d const * mem_addr)
#[inline]
pub unsafe fn mm256_broadcast_pd(mem_addr: *const m128d) -> m256d {
    avx_vbroadcastf128_pd_256(mem_addr as *const u8)
}

// vbroadcastf128
// __m256 _mm256_broadcast_ps (__m128 const * mem_addr)
#[inline]
pub unsafe fn mm256_broadcast_ps(mem_addr: *const m128) -> m256 {
    avx_vbroadcastf128_ps_256(mem_addr as *const u8)
}

// vbroadcastsd
// __m256d _mm256_broadcast_sd (double const * mem_addr)
#[inline]
pub unsafe fn mm256_broadcast_sd(mem_addr: *const f64) -> m256d {
    let d = *mem_addr;
    f64x4(d, d, d, d).as_m256d()
}

// vbroadcastss
// __m128 _mm_broadcast_ss (float const * mem_addr)
#[inline]
pub unsafe fn mm_broadcast_ss(mem_addr: *const f32) -> m128 {
    let d = *mem_addr;
    f32x4(d, d, d, d).as_m128()
}

// vbroadcastss
// __m256 _mm256_broadcast_ss (float const * mem_addr)
#[inline]
pub unsafe fn mm256_broadcast_ss(mem_addr: *const f32) -> m256 {
    let d = *mem_addr;
    f32x8(d, d, d, d, d, d, d, d).as_m256()
}

// __m256 _mm256_castpd_ps (__m256d a)
#[inline]
pub fn mm256_castpd_ps(a: m256d) -> m256 {
    a.as_m256()
}

// __m256i _mm256_castpd_si256 (__m256d a)
#[inline]
pub fn mm256_castpd_si256(a: m256d) -> m256i {
    a.as_m256i()
}

// __m256d _mm256_castpd128_pd256 (__m128d a)
#[inline]
pub fn mm256_castpd128_pd256(a: m128d) -> m256d {
    // TODO(mayah): Why simd_shuffle takes u32? It should be i32?
    // TODO(mayah): Uguu, simd_shuffe4 takes only 0-3 as index?
    // return __builtin_shufflevector(__a, __a, 0, 1, -1, -1);
    // unsafe { simd_shuffle4(a, a, [0, 1, -1i32 as u32, -1i32 as u32]) }

    let b = mm_undefined_pd();
    unsafe { simd_shuffle4(a, b, [0, 1, 2, 3]) }
}

// __m128d _mm256_castpd256_pd128 (__m256d a)
#[inline]
pub fn mm256_castpd256_pd128(a: m256d) -> m128d {
    unsafe { simd_shuffle2(a, a, [0, 1]) }
}

// __m256d _mm256_castps_pd (__m256 a)
#[inline]
pub fn mm256_castps_pd(a: m256) -> m256d {
    a.as_m256d()
}

// __m256i _mm256_castps_si256 (__m256 a)
#[inline]
pub fn mm256_castps_si256(a: m256) -> m256i {
    a.as_m256i()
}

// __m256 _mm256_castps128_ps256 (__m128 a)
#[inline]
pub fn mm256_castps128_ps256(a: m128) -> m256 {
    // TODO(mayah): Use return __builtin_shufflevector(__a, __a, 0, 1, 2, 3, -1, -1, -1, -1);

    let b = mm_undefined_ps();
    unsafe { simd_shuffle8(a, b, [0, 1, 2, 3, 4, 5, 6, 7]) }
}

// __m128 _mm256_castps256_ps128 (__m256 a)
#[inline]
#[allow(unused_variables)]
pub fn mm256_castps256_ps128(a: m256) -> m128 {
    unimplemented!()
}

// __m256i _mm256_castsi128_si256 (__m128i a)
#[inline]
#[allow(unused_variables)]
pub fn mm256_castsi128_si256(a: m128i) -> m256i {
    // TODO(mayah): Use return __builtin_shufflevector(__a, __a, 0, 1, -1, -1);

    let b = mm_undefined_si128();
    let c: i64x4 = unsafe { simd_shuffle4(a, b, [0, 1, 2, 3]) };
    c.as_m256i()
}

// __m256d _mm256_castsi256_pd (__m256i a)
#[inline]
pub fn mm256_castsi256_pd(a: m256i) -> m256d {
    a.as_m256d()
}

// __m256 _mm256_castsi256_ps (__m256i a)
#[inline]
pub fn mm256_castsi256_ps(a: m256i) -> m256 {
    a.as_m256()
}

// __m128i _mm256_castsi256_si128 (__m256i a)
#[inline]
#[allow(unused_variables)]
pub fn mm256_castsi256_si128(a: m256i) -> m128i {
    unimplemented!()
}

// vroundpd
// __m256d _mm256_ceil_pd (__m256d a)
#[inline]
pub fn mm256_ceil_pd(a: m256d) -> m256d {
    mm256_round_pd(a, MM_FROUND_CEIL)
}

// vroundps
// __m256 _mm256_ceil_ps (__m256 a)
#[inline]
pub fn mm256_ceil_ps(a: m256) -> m256 {
    mm256_round_ps(a, MM_FROUND_CEIL)
}

// vcmppd
// __m128d _mm_cmp_pd (__m128d a, __m128d b, const int imm8)
#[inline]
#[allow(unused_variables)]
pub fn mm_cmp_pd(a: m128d, b: m128d, imm8: i32) -> m128d {
    unimplemented!()
    // unsafe { sse2_cmp_pd(a, b, imm8 as i8) }
}

// vcmppd
// __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
#[inline]
#[allow(unused_variables)]
pub fn mm256_cmp_pd(a: m256d, b: m256d, imm8: i32) -> m256d {
    unimplemented!()
    // fn_imm8_arg2!(x86_mm256_cmp_pd, a, b, imm8)
    // unsafe { x86_mm256_cmp_pd(a, b, imm8 as i8) }
}

// vcmpps
// __m128 _mm_cmp_ps (__m128 a, __m128 b, const int imm8)
#[inline]
#[allow(unused_variables)]
pub fn mm_cmp_ps(a: m128, b: m128, imm8: i32) -> m128 {
    unimplemented!()
    // unsafe { sse2_cmp_ps(a, b, imm8 as i8) }
}

// vcmpps
// __m256 _mm256_cmp_ps (__m256 a, __m256 b, const int imm8)
#[inline]
#[allow(unused_variables)]
pub fn mm256_cmp_ps(a: m256, b: m256, imm8: i32) -> m256 {
    unimplemented!()
    // unsafe { x86_mm256_cmp_ps(a, b, imm8 as i8) }
}

// vcmpsd
// __m128d _mm_cmp_sd (__m128d a, __m128d b, const int imm8)
#[inline]
#[allow(unused_variables)]
pub fn mm256_cmp_sd(a: m128d, b: m128d, imm8: i32) -> m128d {
    unimplemented!()
    // unsafe { sse2_cmp_sd(a, b, imm8 as i8) }
}

// vcmpss
// __m128 _mm_cmp_ss (__m128 a, __m128 b, const int imm8)
#[inline]
#[allow(unused_variables)]
pub fn mm256_cmp_ss(a: m128, b: m128, imm8: i32) -> m128 {
    unimplemented!()
    // unsafe { sse2_cmp_ss(a, b, imm8 as i8) }
}

// TODO(mayah): Implement these.
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
#[inline]
pub fn mm256_div_pd(a: m256d, b: m256d) -> m256d {
    unsafe { simd_div(a, b) }
}

// vdivps
// __m256 _mm256_div_ps (__m256 a, __m256 b)
#[inline]
pub fn mm256_div_ps(a: m256, b: m256) -> m256 {
    unsafe { simd_div(a, b) }
}

// vdpps
// __m256 _mm256_dp_ps (__m256 a, __m256 b, const int imm8)
#[inline]
pub fn mm256_dp_ps(a: m256, b: m256, imm8: i32) -> m256 {
    fn_imm8_arg2!(avx_dp_ps_256, a, b, imm8)
}

// ...
// __int16 _mm256_extract_epi16 (__m256i a, const int index)
#[inline]
pub fn mm256_extract_epi16(a: m256i, index: i32) -> i16 {
    a.as_i16x16().extract((index as usize) & 0xF)
}

// ...
// __int32 _mm256_extract_epi32 (__m256i a, const int index)
#[inline]
pub fn mm256_extract_epi32(a: m256i, index: i32) -> i32 {
    a.as_i32x8().extract((index as usize) & 0x7)
}

// ...
// __int64 _mm256_extract_epi64 (__m256i a, const int index)
#[inline]
pub fn mm256_extract_epi64(a: m256i, index: i32) -> i64 {
    a.as_i64x4().extract((index as usize) & 0x3)
}

// ...
// __int8 _mm256_extract_epi8 (__m256i a, const int index)
#[inline]
pub fn mm256_extract_epi8(a: m256i, index: i32) -> i8 {
    a.as_i8x32().extract((index as usize) & 31)
}

// vextractf128
// __m128d _mm256_extractf128_pd (__m256d a, const int imm8)
#[inline]
pub fn mm256_extractf128_pd(a: m256d, imm8: i32) -> m128d {
    unsafe {
        match imm8 & 0x1 {
            0 => simd_shuffle2(a, a, [0, 1]),
            1 => simd_shuffle2(a, a, [2, 3]),
            _ => unreachable!()
        }
    }
}

// vextractf128
// __m128 _mm256_extractf128_ps (__m256 a, const int imm8)
#[inline]
pub fn mm256_extractf128_ps(a: m256, imm8: i32) -> m128 {
    unsafe {
        match imm8 & 0x1 {
            0 => simd_shuffle4(a, a, [0, 1, 2, 3]),
            1 => simd_shuffle4(a, a, [4, 5, 6, 7]),
            _ => unreachable!()
        }
    }
}

// vextractf128
// __m128i _mm256_extractf128_si256 (__m256i a, const int imm8)
pub fn mm256_extractf128_si256(a: m256i, imm8: i32) -> m128i {
    let ai = a.as_i64x4();
    unsafe {
        match imm8 & 0x1 {
            0 => simd_shuffle2(ai, ai, [0, 1]),
            1 => simd_shuffle2(ai, ai, [2, 3]),
            _ => unreachable!()
        }
    }
}

// vroundpd
// __m256d _mm256_floor_pd (__m256d a)
#[inline]
pub fn mm256_floor_pd(a: m256d) -> m256d {
    mm256_round_pd(a, MM_FROUND_FLOOR)
}

// vroundps
// __m256 _mm256_floor_ps (__m256 a)
#[inline]
pub fn mm256_floor_ps(a: m256) -> m256 {
    mm256_round_ps(a, MM_FROUND_FLOOR)
}

// vhaddpd
// __m256d _mm256_hadd_pd (__m256d a, __m256d b)
#[inline]
pub fn mm256_hadd_pd(a: m256d, b: m256d) -> m256d {
    unsafe { x86_mm256_hadd_pd(a, b) }
}

// vhaddps
// __m256 _mm256_hadd_ps (__m256 a, __m256 b)
#[inline]
pub fn mm256_hadd_ps(a: m256, b: m256) -> m256 {
    unsafe { x86_mm256_hadd_ps(a, b) }
}

// vhsubpd
// __m256d _mm256_hsub_pd (__m256d a, __m256d b)
#[inline]
pub fn mm256_hsub_pd(a: m256d, b: m256d) -> m256d {
    unsafe { x86_mm256_hsub_pd(a, b) }
}

// vhsubps
// __m256 _mm256_hsub_ps (__m256 a, __m256 b)
#[inline]
pub fn mm256_hsub_ps(a: m256, b: m256) -> m256 {
    unsafe { x86_mm256_hsub_ps(a, b) }
}

// ...
// __m256i _mm256_insert_epi16 (__m256i a, __int16 i, const int index)
#[inline]
pub fn mm256_insert_epi16(a: m256i, i: i16, index: i32) -> m256i {
    a.as_i16x16().insert(index as usize, i).as_m256i()
}

// ...
// __m256i _mm256_insert_epi32 (__m256i a, __int32 i, const int index)
#[inline]
pub fn mm256_insert_epi32(a: m256i, i: i32, index: i32) -> m256i {
    a.as_i32x8().insert(index as usize, i).as_m256i()
}

// ...
// __m256i _mm256_insert_epi64 (__m256i a, __int64 i, const int index)
#[inline]
pub fn mm256_insert_epi64(a: m256i, i: i64, index: i32) -> m256i {
    a.as_i64x4().insert(index as usize, i).as_m256i()
}

// ...
// __m256i _mm256_insert_epi8 (__m256i a, __int8 i, const int index)
#[inline]
pub fn mm256_insert_epi8(a: m256i, i: i8, index: i32) -> m256i {
    a.as_i8x32().insert(index as usize, i).as_m256i()
}

// vinsertf128
// __m256d _mm256_insertf128_pd (__m256d a, __m128d b, int imm8)
#[inline]
pub fn mm256_insertf128_pd(a: m256d, b: m128d, imm8: i32) -> m256d {
    unsafe {
        match imm8 & 1 {
            0 => simd_shuffle4(a, mm256_castpd128_pd256(b), [4, 5, 2, 3]),
            1 => simd_shuffle4(a, mm256_castpd128_pd256(b), [0, 1, 4, 5]),
            _ => unreachable!()
        }
    }
}

// vinsertf128
// __m256 _mm256_insertf128_ps (__m256 a, __m128 b, int imm8)
#[inline]
pub fn mm256_insertf128_ps(a: m256, b: m128, imm8: i32) -> m256 {
    unsafe {
        match imm8 & 1 {
            0 => simd_shuffle8(a, mm256_castps128_ps256(b), [8, 9, 10, 11, 4, 5, 6, 7]),
            1 => simd_shuffle8(a, mm256_castps128_ps256(b), [0, 1, 2, 3, 8, 9, 10, 11]),
            _ => unreachable!()
        }
    }
}

// vinsertf128
// __m256i _mm256_insertf128_si256 (__m256i a, __m128i b, int imm8)
#[inline]
pub fn mm256_insertf128_si256(a: m256i, b: m128i, imm8: i32) -> m256i {
    unsafe {
        let x: i64x4 = match imm8 & 1 {
            0 => simd_shuffle4(a.as_i64x4(), mm256_castsi128_si256(b).as_i64x4(), [4, 5, 2, 3]),
            1 => simd_shuffle4(a.as_i64x4(), mm256_castsi128_si256(b).as_i64x4(), [0, 1, 4, 5]),
            _ => unreachable!()
        };
        x.as_m256i()
    }
}

// vlddqu
// __m256i _mm256_lddqu_si256 (__m256i const * mem_addr)
#[inline]
pub unsafe fn mm256_lddqu_si256(mem_addr: *const m256i) -> m256i {
    avx_ldu_dq_256(mem_addr as *mut u8).as_m256i()
}

// vmovapd
// __m256d _mm256_load_pd (double const * mem_addr)
#[inline]
pub unsafe fn mm256_load_pd(mem_addr: *const f64) -> m256d {
    // mem_addr should be 32byte aligned
    *(mem_addr as *const m256d)
}

// vmovaps
// __m256 _mm256_load_ps (float const * mem_addr)
#[inline]
pub unsafe fn mm256_load_ps(mem_addr: *const f32) -> m256 {
    *(mem_addr as *const m256)
}

// vmovdqa
// __m256i _mm256_load_si256 (__m256i const * mem_addr)
#[inline]
pub unsafe fn mm256_load_si256(mem_addr: *const m256i) -> m256i {
    *mem_addr
}

// vmovupd
// __m256d _mm256_loadu_pd (double const * mem_addr)
#[inline]
pub unsafe fn mm256_loadu_pd(mem_addr: *const f64) -> m256d {
    let mut result: m256d = std::mem::uninitialized();

    let src_p = mem_addr as *const u8;
    let dst_p = &mut result as *mut m256d as *mut u8;
    std::ptr::copy_nonoverlapping(src_p, dst_p, 32);

    result
}

// vmovups
// __m256 _mm256_loadu_ps (float const * mem_addr)
#[inline]
pub unsafe fn mm256_loadu_ps(mem_addr: *const f32) -> m256 {
    let mut result: m256 = std::mem::uninitialized();

    let src_p = mem_addr as *const u8;
    let dst_p = &mut result as *mut m256 as *mut u8;
    std::ptr::copy_nonoverlapping(src_p, dst_p, 32);

    result
}

// vmovdqu
// __m256i _mm256_loadu_si256 (__m256i const * mem_addr)
#[inline]
pub unsafe fn mm256_loadu_si256(mem_addr: *const m256i) -> m256i {
    let mut result: m256i = std::mem::uninitialized();

    let src_p = mem_addr as *const u8;
    let dst_p = &mut result as *mut m256i as *mut u8;
    std::ptr::copy_nonoverlapping(src_p, dst_p, 32);

    result
}

// ...
// __m256 _mm256_loadu2_m128 (float const* hiaddr, float const* loaddr)
#[inline]
pub unsafe fn mm256_loadu2_m128(hiaddr: *const f32, loaddr: *const f32) -> m256 {
    // __m256 __v256 = _mm256_castps128_ps256(((struct __loadu_ps*)__addr_lo)->__v);
    // return _mm256_insertf128_ps(__v256, ((struct __loadu_ps*)__addr_hi)->__v, 1);

    // TODO(mayah): correct?
    let lo = mm_loadu_ps(loaddr);
    let hi = mm_loadu_ps(hiaddr);
    mm256_insertf128_ps(mm256_castps128_ps256(lo), hi, 1)
}

// ...
// __m256d _mm256_loadu2_m128d (double const* hiaddr, double const* loaddr)
#[inline]
pub unsafe fn mm256_loadu2_m128d(hiaddr: *const f64, loaddr: *const f64) -> m256d {
    // __m256d __v256 = _mm256_castpd128_pd256(((struct __loadu_pd*)__addr_lo)->__v);
    // return _mm256_insertf128_pd(__v256, ((struct __loadu_pd*)__addr_hi)->__v, 1);

    // TODO(mayah): correct?
    let lo = mm_loadu_pd(loaddr);
    let hi = mm_loadu_pd(hiaddr);
    mm256_insertf128_pd(mm256_castpd128_pd256(lo), hi, 1)
}

// ...
// __m256i _mm256_loadu2_m128i (__m128i const* hiaddr, __m128i const* loaddr)
#[inline]
pub unsafe fn mm256_loadu2_m128i(hiaddr: *const m128i, loaddr: *const m128i) -> m256i {
    //__m256i __v256 = _mm256_castsi128_si256(((struct __loadu_si128*)__addr_lo)->__v);
    // return _mm256_insertf128_si256(__v256, ((struct __loadu_si128*)__addr_hi)->__v, 1);

    // TODO(mayah): correct?
    let lo = mm_loadu_si128(loaddr);
    let hi = mm_loadu_si128(hiaddr);
    mm256_insertf128_si256(mm256_castsi128_si256(lo), hi, 1)
}

// vmaskmovpd
// __m128d _mm_maskload_pd (double const * mem_addr, __m128i mask)
#[inline]
pub unsafe fn mm_maskload_pd(mem_addr: *const f64, mask: m128i) -> m128d {
     avx_maskload_pd(mem_addr as *const i8, mask.as_i64x2())
}

// vmaskmovpd
// __m256d _mm256_maskload_pd (double const * mem_addr, __m256i mask)
#[inline]
pub unsafe fn mm256_maskload_pd(mem_addr: *const f64, mask: m256i) -> m256d {
    avx_maskload_pd_256(mem_addr as *const i8, mask.as_i64x4())
}

// vmaskmovps
// __m128 _mm_maskload_ps (float const * mem_addr, __m128i mask)
#[inline]
pub unsafe fn mm_maskload_ps(mem_addr: *const f32, mask: m128i) -> m128 {
    avx_maskload_ps(mem_addr as *const i8, mask.as_i32x4())
}

// vmaskmovps
// __m256 _mm256_maskload_ps (float const * mem_addr, __m256i mask)
#[inline]
pub unsafe fn mm256_maskload_ps(mem_addr: *const f32, mask: m256i) -> m256 {
    avx_maskload_ps_256(mem_addr as *const i8, mask.as_i32x8())
}

// vmaskmovpd
// void _mm_maskstore_pd (double * mem_addr, __m128i mask, __m128d a)
#[inline]
pub unsafe fn mm_maskstore_pd(mem_addr: *mut f64, mask: m128i, a: m128d) {
    avx_maskstore_pd(mem_addr as *mut i8, mask.as_i64x2(), a)
}

// vmaskmovpd
// void _mm256_maskstore_pd (double * mem_addr, __m256i mask, __m256d a)
#[inline]
pub unsafe fn mm256_maskstore_pd(mem_addr: *mut f64, mask: m256i, a: m256d) {
    avx_maskstore_pd_256(mem_addr as *mut i8, mask.as_i64x4(), a)
}

// vmaskmovps
// void _mm_maskstore_ps (float * mem_addr, __m128i mask, __m128 a)
#[inline]
pub unsafe fn mm_maskstore_ps(mem_addr: *mut f32, mask: m128i, a: m128) {
    avx_maskstore_ps(mem_addr as *mut i8, mask.as_i32x4(), a)
}

// vmaskmovps
// void _mm256_maskstore_ps (float * mem_addr, __m256i mask, __m256 a)
#[inline]
pub unsafe fn mm256_maskstore_ps(mem_addr: *mut f32, mask: m256i, a: m256) {
    avx_maskstore_ps_256(mem_addr as *mut i8, mask.as_i32x8(), a)
}

// TODO(mayah): This doc test doesn't work?
/// vmaxpd
/// `__m256d _mm256_max_pd(__m256d a, __m256d b);`
///
/// # Examples
///
/// ```
/// use x86intrin::*;
///
/// let a = mm256_setr_pd(1.0, 2.0, 3.0, 4.0);
/// let b = mm256_setr_pd(3.0, 2.0, 1.0, 2.0);
/// assert_eq!(mm256_max_pd(a, b).as_f64x4().as_array(), [3.0, 2.0, 3.0, 4.0]);
/// ```
#[inline]
pub fn mm256_max_pd(a: m256d, b: m256d) -> m256d {
    unsafe { x86_mm256_max_pd(a, b) }
}

// vmaxps
// __m256 _mm256_max_ps (__m256 a, __m256 b)
#[inline]
pub fn mm256_max_ps(a: m256, b: m256) -> m256 {
    unsafe { x86_mm256_max_ps(a, b) }
}

// vminpd
// __m256d _mm256_min_pd (__m256d a, __m256d b)
#[inline]
pub fn mm256_min_pd(a: m256d, b: m256d) -> m256d {
    unsafe { x86_mm256_min_pd(a, b) }
}

// vminps
// __m256 _mm256_min_ps (__m256 a, __m256 b)
#[inline]
pub fn mm256_min_ps(a: m256, b: m256) -> m256 {
    unsafe { x86_mm256_min_ps(a, b) }
}

// vmovddup
// __m256d _mm256_movedup_pd (__m256d a)
#[inline]
pub fn mm256_movedup_pd(a: m256d) -> m256d {
    unsafe { simd_shuffle4(a, a, [0, 0, 2, 2]) }
}

// vmovshdup
// __m256 _mm256_movehdup_ps (__m256 a)
#[inline]
pub fn mm256_movehdup_ps(a: m256) -> m256 {
    unsafe { simd_shuffle8(a, a, [1, 1, 3, 3, 5, 5, 7, 7]) }
}

// vmovsldup
// __m256 _mm256_moveldup_ps (__m256 a)
#[inline]
pub fn mm256_moveldup_ps(a: m256) -> m256 {
    unsafe { simd_shuffle8(a, a, [0, 0, 2, 2, 4, 4, 6, 6]) }
}

// vmovmskpd
// int _mm256_movemask_pd (__m256d a)
#[inline]
pub fn mm256_movemask_pd(a: m256d) -> i32 {
    unsafe { avx_movmsk_pd_256(a) }
}

// vmovmskps
// int _mm256_movemask_ps (__m256 a)
#[inline]
pub fn mm256_movemask_ps(a: m256) -> i32 {
    unsafe { avx_movmsk_ps_256(a) }
}

// vmulpd
// __m256d _mm256_mul_pd (__m256d a, __m256d b)
#[inline]
pub fn mm256_mul_pd(a: m256d, b: m256d) -> m256d {
    unsafe { simd_mul(a, b) }
}

// vmulps
// __m256 _mm256_mul_ps (__m256 a, __m256 b)
#[inline]
pub fn mm256_mul_ps(a: m256, b: m256) -> m256 {
    unsafe { simd_mul(a, b) }
}

// vorpd
// __m256d _mm256_or_pd (__m256d a, __m256d b)
#[inline]
pub fn mm256_or_pd(a: m256d, b: m256d) -> m256d {
    let ai = a.as_m256i();
    let bi = b.as_m256i();
    unsafe { simd_or(ai, bi).as_m256d() }
}

// vorps
// __m256 _mm256_or_ps (__m256 a, __m256 b)
#[inline]
pub fn mm256_or_ps(a: m256, b: m256) -> m256 {
    let ai = a.as_m256i();
    let bi = b.as_m256i();
    unsafe { simd_or(ai, bi).as_m256() }
}

// TODO(mayah): Implement these.
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
#[inline]
pub fn mm256_rcp_ps(a: m256) -> m256 {
    unsafe { x86_mm256_rcp_ps(a) }
}

// vroundpd
// __m256d _mm256_round_pd (__m256d a, int rounding)
#[inline]
pub fn mm256_round_pd(a: m256d, rounding: i32) -> m256d {
    fn_imm8_arg1!(avx_round_pd, a, rounding)
}

// vroundps
// __m256 _mm256_round_ps (__m256 a, int rounding)
#[inline]
pub fn mm256_round_ps(a: m256, rounding: i32) -> m256 {
    fn_imm8_arg1!(avx_round_ps, a, rounding)
}

// vrsqrtps
// __m256 _mm256_rsqrt_ps (__m256 a)
#[inline]
pub fn mm256_rsqrt_ps(a: m256) -> m256 {
    unsafe { x86_mm256_rsqrt_ps(a) }
}

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
#[inline]
pub fn mm256_shuffle_pd(a: m256d, b: m256d, imm8: i32) -> m256d {
    // #define _mm256_shuffle_pd(a, b, mask) __extension__ ({ \
    //     __m256d __a = (a); \
    //     __m256d __b = (b); \
    //     (__m256d)__builtin_shufflevector((__v4df)__a, (__v4df)__b, \
    //         (mask) & 0x1, \
    //         (((mask) & 0x2) >> 1) + 4, \
    //         (((mask) & 0x4) >> 2) + 2, \
    //         (((mask) & 0x8) >> 3) + 6); })
    unsafe {
        match imm8 & 0x0F {
            0x00 => simd_shuffle4(a, b, [0, 4, 2, 6]),
            0x01 => simd_shuffle4(a, b, [1, 4, 2, 6]),
            0x02 => simd_shuffle4(a, b, [0, 5, 2, 6]),
            0x03 => simd_shuffle4(a, b, [1, 5, 2, 6]),
            0x04 => simd_shuffle4(a, b, [0, 4, 3, 6]),
            0x05 => simd_shuffle4(a, b, [1, 4, 3, 6]),
            0x06 => simd_shuffle4(a, b, [0, 5, 3, 6]),
            0x07 => simd_shuffle4(a, b, [1, 5, 3, 6]),
            0x08 => simd_shuffle4(a, b, [0, 4, 2, 7]),
            0x09 => simd_shuffle4(a, b, [1, 4, 2, 7]),
            0x0A => simd_shuffle4(a, b, [0, 5, 2, 7]),
            0x0B => simd_shuffle4(a, b, [1, 5, 2, 7]),
            0x0C => simd_shuffle4(a, b, [0, 4, 3, 7]),
            0x0D => simd_shuffle4(a, b, [1, 4, 3, 7]),
            0x0E => simd_shuffle4(a, b, [0, 5, 3, 7]),
            0x0F => simd_shuffle4(a, b, [1, 5, 3, 7]),
            _ => unreachable!(),
        }
    }
}

// vshufps
// __m256 _mm256_shuffle_ps (__m256 a, __m256 b, const int imm8)
#[inline]
pub fn mm256_shuffle_ps(a: m256, b: m256, imm8: i32) -> m256 {
    /* Vector shuffle */
    // #define _mm256_shuffle_ps(a, b, mask) __extension__ ({ \
    //     __m256 __a = (a); \
    //     __m256 __b = (b); \
    //     (__m256)__builtin_shufflevector((__v8sf)__a, (__v8sf)__b, \
    //         (mask) & 0x3,                ((mask) & 0xc) >> 2, \
    //         (((mask) & 0x30) >> 4) + 8,  (((mask) & 0xc0) >> 6) + 8, \
    //         ((mask) & 0x3) + 4,          (((mask) & 0xc) >> 2) + 4, \
    //         (((mask) & 0x30) >> 4) + 12, (((mask) & 0xc0) >> 6) + 12); })

    unsafe {
        match imm8 & 0xFF {
            0x00 => simd_shuffle8(a, b, [0, 0, 8, 8, 4, 4, 12, 12]),
            0x01 => simd_shuffle8(a, b, [1, 0, 8, 8, 5, 4, 12, 12]),
            0x02 => simd_shuffle8(a, b, [2, 0, 8, 8, 6, 4, 12, 12]),
            0x03 => simd_shuffle8(a, b, [3, 0, 8, 8, 7, 4, 12, 12]),
            0x04 => simd_shuffle8(a, b, [0, 1, 8, 8, 4, 5, 12, 12]),
            0x05 => simd_shuffle8(a, b, [1, 1, 8, 8, 5, 5, 12, 12]),
            0x06 => simd_shuffle8(a, b, [2, 1, 8, 8, 6, 5, 12, 12]),
            0x07 => simd_shuffle8(a, b, [3, 1, 8, 8, 7, 5, 12, 12]),
            0x08 => simd_shuffle8(a, b, [0, 2, 8, 8, 4, 6, 12, 12]),
            0x09 => simd_shuffle8(a, b, [1, 2, 8, 8, 5, 6, 12, 12]),
            0x0A => simd_shuffle8(a, b, [2, 2, 8, 8, 6, 6, 12, 12]),
            0x0B => simd_shuffle8(a, b, [3, 2, 8, 8, 7, 6, 12, 12]),
            0x0C => simd_shuffle8(a, b, [0, 3, 8, 8, 4, 7, 12, 12]),
            0x0D => simd_shuffle8(a, b, [1, 3, 8, 8, 5, 7, 12, 12]),
            0x0E => simd_shuffle8(a, b, [2, 3, 8, 8, 6, 7, 12, 12]),
            0x0F => simd_shuffle8(a, b, [3, 3, 8, 8, 7, 7, 12, 12]),
            0x10 => simd_shuffle8(a, b, [0, 0, 9, 8, 4, 4, 13, 12]),
            0x11 => simd_shuffle8(a, b, [1, 0, 9, 8, 5, 4, 13, 12]),
            0x12 => simd_shuffle8(a, b, [2, 0, 9, 8, 6, 4, 13, 12]),
            0x13 => simd_shuffle8(a, b, [3, 0, 9, 8, 7, 4, 13, 12]),
            0x14 => simd_shuffle8(a, b, [0, 1, 9, 8, 4, 5, 13, 12]),
            0x15 => simd_shuffle8(a, b, [1, 1, 9, 8, 5, 5, 13, 12]),
            0x16 => simd_shuffle8(a, b, [2, 1, 9, 8, 6, 5, 13, 12]),
            0x17 => simd_shuffle8(a, b, [3, 1, 9, 8, 7, 5, 13, 12]),
            0x18 => simd_shuffle8(a, b, [0, 2, 9, 8, 4, 6, 13, 12]),
            0x19 => simd_shuffle8(a, b, [1, 2, 9, 8, 5, 6, 13, 12]),
            0x1A => simd_shuffle8(a, b, [2, 2, 9, 8, 6, 6, 13, 12]),
            0x1B => simd_shuffle8(a, b, [3, 2, 9, 8, 7, 6, 13, 12]),
            0x1C => simd_shuffle8(a, b, [0, 3, 9, 8, 4, 7, 13, 12]),
            0x1D => simd_shuffle8(a, b, [1, 3, 9, 8, 5, 7, 13, 12]),
            0x1E => simd_shuffle8(a, b, [2, 3, 9, 8, 6, 7, 13, 12]),
            0x1F => simd_shuffle8(a, b, [3, 3, 9, 8, 7, 7, 13, 12]),
            0x20 => simd_shuffle8(a, b, [0, 0, 10, 8, 4, 4, 14, 12]),
            0x21 => simd_shuffle8(a, b, [1, 0, 10, 8, 5, 4, 14, 12]),
            0x22 => simd_shuffle8(a, b, [2, 0, 10, 8, 6, 4, 14, 12]),
            0x23 => simd_shuffle8(a, b, [3, 0, 10, 8, 7, 4, 14, 12]),
            0x24 => simd_shuffle8(a, b, [0, 1, 10, 8, 4, 5, 14, 12]),
            0x25 => simd_shuffle8(a, b, [1, 1, 10, 8, 5, 5, 14, 12]),
            0x26 => simd_shuffle8(a, b, [2, 1, 10, 8, 6, 5, 14, 12]),
            0x27 => simd_shuffle8(a, b, [3, 1, 10, 8, 7, 5, 14, 12]),
            0x28 => simd_shuffle8(a, b, [0, 2, 10, 8, 4, 6, 14, 12]),
            0x29 => simd_shuffle8(a, b, [1, 2, 10, 8, 5, 6, 14, 12]),
            0x2A => simd_shuffle8(a, b, [2, 2, 10, 8, 6, 6, 14, 12]),
            0x2B => simd_shuffle8(a, b, [3, 2, 10, 8, 7, 6, 14, 12]),
            0x2C => simd_shuffle8(a, b, [0, 3, 10, 8, 4, 7, 14, 12]),
            0x2D => simd_shuffle8(a, b, [1, 3, 10, 8, 5, 7, 14, 12]),
            0x2E => simd_shuffle8(a, b, [2, 3, 10, 8, 6, 7, 14, 12]),
            0x2F => simd_shuffle8(a, b, [3, 3, 10, 8, 7, 7, 14, 12]),
            0x30 => simd_shuffle8(a, b, [0, 0, 11, 8, 4, 4, 15, 12]),
            0x31 => simd_shuffle8(a, b, [1, 0, 11, 8, 5, 4, 15, 12]),
            0x32 => simd_shuffle8(a, b, [2, 0, 11, 8, 6, 4, 15, 12]),
            0x33 => simd_shuffle8(a, b, [3, 0, 11, 8, 7, 4, 15, 12]),
            0x34 => simd_shuffle8(a, b, [0, 1, 11, 8, 4, 5, 15, 12]),
            0x35 => simd_shuffle8(a, b, [1, 1, 11, 8, 5, 5, 15, 12]),
            0x36 => simd_shuffle8(a, b, [2, 1, 11, 8, 6, 5, 15, 12]),
            0x37 => simd_shuffle8(a, b, [3, 1, 11, 8, 7, 5, 15, 12]),
            0x38 => simd_shuffle8(a, b, [0, 2, 11, 8, 4, 6, 15, 12]),
            0x39 => simd_shuffle8(a, b, [1, 2, 11, 8, 5, 6, 15, 12]),
            0x3A => simd_shuffle8(a, b, [2, 2, 11, 8, 6, 6, 15, 12]),
            0x3B => simd_shuffle8(a, b, [3, 2, 11, 8, 7, 6, 15, 12]),
            0x3C => simd_shuffle8(a, b, [0, 3, 11, 8, 4, 7, 15, 12]),
            0x3D => simd_shuffle8(a, b, [1, 3, 11, 8, 5, 7, 15, 12]),
            0x3E => simd_shuffle8(a, b, [2, 3, 11, 8, 6, 7, 15, 12]),
            0x3F => simd_shuffle8(a, b, [3, 3, 11, 8, 7, 7, 15, 12]),
            0x40 => simd_shuffle8(a, b, [0, 0, 8, 9, 4, 4, 12, 13]),
            0x41 => simd_shuffle8(a, b, [1, 0, 8, 9, 5, 4, 12, 13]),
            0x42 => simd_shuffle8(a, b, [2, 0, 8, 9, 6, 4, 12, 13]),
            0x43 => simd_shuffle8(a, b, [3, 0, 8, 9, 7, 4, 12, 13]),
            0x44 => simd_shuffle8(a, b, [0, 1, 8, 9, 4, 5, 12, 13]),
            0x45 => simd_shuffle8(a, b, [1, 1, 8, 9, 5, 5, 12, 13]),
            0x46 => simd_shuffle8(a, b, [2, 1, 8, 9, 6, 5, 12, 13]),
            0x47 => simd_shuffle8(a, b, [3, 1, 8, 9, 7, 5, 12, 13]),
            0x48 => simd_shuffle8(a, b, [0, 2, 8, 9, 4, 6, 12, 13]),
            0x49 => simd_shuffle8(a, b, [1, 2, 8, 9, 5, 6, 12, 13]),
            0x4A => simd_shuffle8(a, b, [2, 2, 8, 9, 6, 6, 12, 13]),
            0x4B => simd_shuffle8(a, b, [3, 2, 8, 9, 7, 6, 12, 13]),
            0x4C => simd_shuffle8(a, b, [0, 3, 8, 9, 4, 7, 12, 13]),
            0x4D => simd_shuffle8(a, b, [1, 3, 8, 9, 5, 7, 12, 13]),
            0x4E => simd_shuffle8(a, b, [2, 3, 8, 9, 6, 7, 12, 13]),
            0x4F => simd_shuffle8(a, b, [3, 3, 8, 9, 7, 7, 12, 13]),
            0x50 => simd_shuffle8(a, b, [0, 0, 9, 9, 4, 4, 13, 13]),
            0x51 => simd_shuffle8(a, b, [1, 0, 9, 9, 5, 4, 13, 13]),
            0x52 => simd_shuffle8(a, b, [2, 0, 9, 9, 6, 4, 13, 13]),
            0x53 => simd_shuffle8(a, b, [3, 0, 9, 9, 7, 4, 13, 13]),
            0x54 => simd_shuffle8(a, b, [0, 1, 9, 9, 4, 5, 13, 13]),
            0x55 => simd_shuffle8(a, b, [1, 1, 9, 9, 5, 5, 13, 13]),
            0x56 => simd_shuffle8(a, b, [2, 1, 9, 9, 6, 5, 13, 13]),
            0x57 => simd_shuffle8(a, b, [3, 1, 9, 9, 7, 5, 13, 13]),
            0x58 => simd_shuffle8(a, b, [0, 2, 9, 9, 4, 6, 13, 13]),
            0x59 => simd_shuffle8(a, b, [1, 2, 9, 9, 5, 6, 13, 13]),
            0x5A => simd_shuffle8(a, b, [2, 2, 9, 9, 6, 6, 13, 13]),
            0x5B => simd_shuffle8(a, b, [3, 2, 9, 9, 7, 6, 13, 13]),
            0x5C => simd_shuffle8(a, b, [0, 3, 9, 9, 4, 7, 13, 13]),
            0x5D => simd_shuffle8(a, b, [1, 3, 9, 9, 5, 7, 13, 13]),
            0x5E => simd_shuffle8(a, b, [2, 3, 9, 9, 6, 7, 13, 13]),
            0x5F => simd_shuffle8(a, b, [3, 3, 9, 9, 7, 7, 13, 13]),
            0x60 => simd_shuffle8(a, b, [0, 0, 10, 9, 4, 4, 14, 13]),
            0x61 => simd_shuffle8(a, b, [1, 0, 10, 9, 5, 4, 14, 13]),
            0x62 => simd_shuffle8(a, b, [2, 0, 10, 9, 6, 4, 14, 13]),
            0x63 => simd_shuffle8(a, b, [3, 0, 10, 9, 7, 4, 14, 13]),
            0x64 => simd_shuffle8(a, b, [0, 1, 10, 9, 4, 5, 14, 13]),
            0x65 => simd_shuffle8(a, b, [1, 1, 10, 9, 5, 5, 14, 13]),
            0x66 => simd_shuffle8(a, b, [2, 1, 10, 9, 6, 5, 14, 13]),
            0x67 => simd_shuffle8(a, b, [3, 1, 10, 9, 7, 5, 14, 13]),
            0x68 => simd_shuffle8(a, b, [0, 2, 10, 9, 4, 6, 14, 13]),
            0x69 => simd_shuffle8(a, b, [1, 2, 10, 9, 5, 6, 14, 13]),
            0x6A => simd_shuffle8(a, b, [2, 2, 10, 9, 6, 6, 14, 13]),
            0x6B => simd_shuffle8(a, b, [3, 2, 10, 9, 7, 6, 14, 13]),
            0x6C => simd_shuffle8(a, b, [0, 3, 10, 9, 4, 7, 14, 13]),
            0x6D => simd_shuffle8(a, b, [1, 3, 10, 9, 5, 7, 14, 13]),
            0x6E => simd_shuffle8(a, b, [2, 3, 10, 9, 6, 7, 14, 13]),
            0x6F => simd_shuffle8(a, b, [3, 3, 10, 9, 7, 7, 14, 13]),
            0x70 => simd_shuffle8(a, b, [0, 0, 11, 9, 4, 4, 15, 13]),
            0x71 => simd_shuffle8(a, b, [1, 0, 11, 9, 5, 4, 15, 13]),
            0x72 => simd_shuffle8(a, b, [2, 0, 11, 9, 6, 4, 15, 13]),
            0x73 => simd_shuffle8(a, b, [3, 0, 11, 9, 7, 4, 15, 13]),
            0x74 => simd_shuffle8(a, b, [0, 1, 11, 9, 4, 5, 15, 13]),
            0x75 => simd_shuffle8(a, b, [1, 1, 11, 9, 5, 5, 15, 13]),
            0x76 => simd_shuffle8(a, b, [2, 1, 11, 9, 6, 5, 15, 13]),
            0x77 => simd_shuffle8(a, b, [3, 1, 11, 9, 7, 5, 15, 13]),
            0x78 => simd_shuffle8(a, b, [0, 2, 11, 9, 4, 6, 15, 13]),
            0x79 => simd_shuffle8(a, b, [1, 2, 11, 9, 5, 6, 15, 13]),
            0x7A => simd_shuffle8(a, b, [2, 2, 11, 9, 6, 6, 15, 13]),
            0x7B => simd_shuffle8(a, b, [3, 2, 11, 9, 7, 6, 15, 13]),
            0x7C => simd_shuffle8(a, b, [0, 3, 11, 9, 4, 7, 15, 13]),
            0x7D => simd_shuffle8(a, b, [1, 3, 11, 9, 5, 7, 15, 13]),
            0x7E => simd_shuffle8(a, b, [2, 3, 11, 9, 6, 7, 15, 13]),
            0x7F => simd_shuffle8(a, b, [3, 3, 11, 9, 7, 7, 15, 13]),
            0x80 => simd_shuffle8(a, b, [0, 0, 8, 10, 4, 4, 12, 14]),
            0x81 => simd_shuffle8(a, b, [1, 0, 8, 10, 5, 4, 12, 14]),
            0x82 => simd_shuffle8(a, b, [2, 0, 8, 10, 6, 4, 12, 14]),
            0x83 => simd_shuffle8(a, b, [3, 0, 8, 10, 7, 4, 12, 14]),
            0x84 => simd_shuffle8(a, b, [0, 1, 8, 10, 4, 5, 12, 14]),
            0x85 => simd_shuffle8(a, b, [1, 1, 8, 10, 5, 5, 12, 14]),
            0x86 => simd_shuffle8(a, b, [2, 1, 8, 10, 6, 5, 12, 14]),
            0x87 => simd_shuffle8(a, b, [3, 1, 8, 10, 7, 5, 12, 14]),
            0x88 => simd_shuffle8(a, b, [0, 2, 8, 10, 4, 6, 12, 14]),
            0x89 => simd_shuffle8(a, b, [1, 2, 8, 10, 5, 6, 12, 14]),
            0x8A => simd_shuffle8(a, b, [2, 2, 8, 10, 6, 6, 12, 14]),
            0x8B => simd_shuffle8(a, b, [3, 2, 8, 10, 7, 6, 12, 14]),
            0x8C => simd_shuffle8(a, b, [0, 3, 8, 10, 4, 7, 12, 14]),
            0x8D => simd_shuffle8(a, b, [1, 3, 8, 10, 5, 7, 12, 14]),
            0x8E => simd_shuffle8(a, b, [2, 3, 8, 10, 6, 7, 12, 14]),
            0x8F => simd_shuffle8(a, b, [3, 3, 8, 10, 7, 7, 12, 14]),
            0x90 => simd_shuffle8(a, b, [0, 0, 9, 10, 4, 4, 13, 14]),
            0x91 => simd_shuffle8(a, b, [1, 0, 9, 10, 5, 4, 13, 14]),
            0x92 => simd_shuffle8(a, b, [2, 0, 9, 10, 6, 4, 13, 14]),
            0x93 => simd_shuffle8(a, b, [3, 0, 9, 10, 7, 4, 13, 14]),
            0x94 => simd_shuffle8(a, b, [0, 1, 9, 10, 4, 5, 13, 14]),
            0x95 => simd_shuffle8(a, b, [1, 1, 9, 10, 5, 5, 13, 14]),
            0x96 => simd_shuffle8(a, b, [2, 1, 9, 10, 6, 5, 13, 14]),
            0x97 => simd_shuffle8(a, b, [3, 1, 9, 10, 7, 5, 13, 14]),
            0x98 => simd_shuffle8(a, b, [0, 2, 9, 10, 4, 6, 13, 14]),
            0x99 => simd_shuffle8(a, b, [1, 2, 9, 10, 5, 6, 13, 14]),
            0x9A => simd_shuffle8(a, b, [2, 2, 9, 10, 6, 6, 13, 14]),
            0x9B => simd_shuffle8(a, b, [3, 2, 9, 10, 7, 6, 13, 14]),
            0x9C => simd_shuffle8(a, b, [0, 3, 9, 10, 4, 7, 13, 14]),
            0x9D => simd_shuffle8(a, b, [1, 3, 9, 10, 5, 7, 13, 14]),
            0x9E => simd_shuffle8(a, b, [2, 3, 9, 10, 6, 7, 13, 14]),
            0x9F => simd_shuffle8(a, b, [3, 3, 9, 10, 7, 7, 13, 14]),
            0xA0 => simd_shuffle8(a, b, [0, 0, 10, 10, 4, 4, 14, 14]),
            0xA1 => simd_shuffle8(a, b, [1, 0, 10, 10, 5, 4, 14, 14]),
            0xA2 => simd_shuffle8(a, b, [2, 0, 10, 10, 6, 4, 14, 14]),
            0xA3 => simd_shuffle8(a, b, [3, 0, 10, 10, 7, 4, 14, 14]),
            0xA4 => simd_shuffle8(a, b, [0, 1, 10, 10, 4, 5, 14, 14]),
            0xA5 => simd_shuffle8(a, b, [1, 1, 10, 10, 5, 5, 14, 14]),
            0xA6 => simd_shuffle8(a, b, [2, 1, 10, 10, 6, 5, 14, 14]),
            0xA7 => simd_shuffle8(a, b, [3, 1, 10, 10, 7, 5, 14, 14]),
            0xA8 => simd_shuffle8(a, b, [0, 2, 10, 10, 4, 6, 14, 14]),
            0xA9 => simd_shuffle8(a, b, [1, 2, 10, 10, 5, 6, 14, 14]),
            0xAA => simd_shuffle8(a, b, [2, 2, 10, 10, 6, 6, 14, 14]),
            0xAB => simd_shuffle8(a, b, [3, 2, 10, 10, 7, 6, 14, 14]),
            0xAC => simd_shuffle8(a, b, [0, 3, 10, 10, 4, 7, 14, 14]),
            0xAD => simd_shuffle8(a, b, [1, 3, 10, 10, 5, 7, 14, 14]),
            0xAE => simd_shuffle8(a, b, [2, 3, 10, 10, 6, 7, 14, 14]),
            0xAF => simd_shuffle8(a, b, [3, 3, 10, 10, 7, 7, 14, 14]),
            0xB0 => simd_shuffle8(a, b, [0, 0, 11, 10, 4, 4, 15, 14]),
            0xB1 => simd_shuffle8(a, b, [1, 0, 11, 10, 5, 4, 15, 14]),
            0xB2 => simd_shuffle8(a, b, [2, 0, 11, 10, 6, 4, 15, 14]),
            0xB3 => simd_shuffle8(a, b, [3, 0, 11, 10, 7, 4, 15, 14]),
            0xB4 => simd_shuffle8(a, b, [0, 1, 11, 10, 4, 5, 15, 14]),
            0xB5 => simd_shuffle8(a, b, [1, 1, 11, 10, 5, 5, 15, 14]),
            0xB6 => simd_shuffle8(a, b, [2, 1, 11, 10, 6, 5, 15, 14]),
            0xB7 => simd_shuffle8(a, b, [3, 1, 11, 10, 7, 5, 15, 14]),
            0xB8 => simd_shuffle8(a, b, [0, 2, 11, 10, 4, 6, 15, 14]),
            0xB9 => simd_shuffle8(a, b, [1, 2, 11, 10, 5, 6, 15, 14]),
            0xBA => simd_shuffle8(a, b, [2, 2, 11, 10, 6, 6, 15, 14]),
            0xBB => simd_shuffle8(a, b, [3, 2, 11, 10, 7, 6, 15, 14]),
            0xBC => simd_shuffle8(a, b, [0, 3, 11, 10, 4, 7, 15, 14]),
            0xBD => simd_shuffle8(a, b, [1, 3, 11, 10, 5, 7, 15, 14]),
            0xBE => simd_shuffle8(a, b, [2, 3, 11, 10, 6, 7, 15, 14]),
            0xBF => simd_shuffle8(a, b, [3, 3, 11, 10, 7, 7, 15, 14]),
            0xC0 => simd_shuffle8(a, b, [0, 0, 8, 11, 4, 4, 12, 15]),
            0xC1 => simd_shuffle8(a, b, [1, 0, 8, 11, 5, 4, 12, 15]),
            0xC2 => simd_shuffle8(a, b, [2, 0, 8, 11, 6, 4, 12, 15]),
            0xC3 => simd_shuffle8(a, b, [3, 0, 8, 11, 7, 4, 12, 15]),
            0xC4 => simd_shuffle8(a, b, [0, 1, 8, 11, 4, 5, 12, 15]),
            0xC5 => simd_shuffle8(a, b, [1, 1, 8, 11, 5, 5, 12, 15]),
            0xC6 => simd_shuffle8(a, b, [2, 1, 8, 11, 6, 5, 12, 15]),
            0xC7 => simd_shuffle8(a, b, [3, 1, 8, 11, 7, 5, 12, 15]),
            0xC8 => simd_shuffle8(a, b, [0, 2, 8, 11, 4, 6, 12, 15]),
            0xC9 => simd_shuffle8(a, b, [1, 2, 8, 11, 5, 6, 12, 15]),
            0xCA => simd_shuffle8(a, b, [2, 2, 8, 11, 6, 6, 12, 15]),
            0xCB => simd_shuffle8(a, b, [3, 2, 8, 11, 7, 6, 12, 15]),
            0xCC => simd_shuffle8(a, b, [0, 3, 8, 11, 4, 7, 12, 15]),
            0xCD => simd_shuffle8(a, b, [1, 3, 8, 11, 5, 7, 12, 15]),
            0xCE => simd_shuffle8(a, b, [2, 3, 8, 11, 6, 7, 12, 15]),
            0xCF => simd_shuffle8(a, b, [3, 3, 8, 11, 7, 7, 12, 15]),
            0xD0 => simd_shuffle8(a, b, [0, 0, 9, 11, 4, 4, 13, 15]),
            0xD1 => simd_shuffle8(a, b, [1, 0, 9, 11, 5, 4, 13, 15]),
            0xD2 => simd_shuffle8(a, b, [2, 0, 9, 11, 6, 4, 13, 15]),
            0xD3 => simd_shuffle8(a, b, [3, 0, 9, 11, 7, 4, 13, 15]),
            0xD4 => simd_shuffle8(a, b, [0, 1, 9, 11, 4, 5, 13, 15]),
            0xD5 => simd_shuffle8(a, b, [1, 1, 9, 11, 5, 5, 13, 15]),
            0xD6 => simd_shuffle8(a, b, [2, 1, 9, 11, 6, 5, 13, 15]),
            0xD7 => simd_shuffle8(a, b, [3, 1, 9, 11, 7, 5, 13, 15]),
            0xD8 => simd_shuffle8(a, b, [0, 2, 9, 11, 4, 6, 13, 15]),
            0xD9 => simd_shuffle8(a, b, [1, 2, 9, 11, 5, 6, 13, 15]),
            0xDA => simd_shuffle8(a, b, [2, 2, 9, 11, 6, 6, 13, 15]),
            0xDB => simd_shuffle8(a, b, [3, 2, 9, 11, 7, 6, 13, 15]),
            0xDC => simd_shuffle8(a, b, [0, 3, 9, 11, 4, 7, 13, 15]),
            0xDD => simd_shuffle8(a, b, [1, 3, 9, 11, 5, 7, 13, 15]),
            0xDE => simd_shuffle8(a, b, [2, 3, 9, 11, 6, 7, 13, 15]),
            0xDF => simd_shuffle8(a, b, [3, 3, 9, 11, 7, 7, 13, 15]),
            0xE0 => simd_shuffle8(a, b, [0, 0, 10, 11, 4, 4, 14, 15]),
            0xE1 => simd_shuffle8(a, b, [1, 0, 10, 11, 5, 4, 14, 15]),
            0xE2 => simd_shuffle8(a, b, [2, 0, 10, 11, 6, 4, 14, 15]),
            0xE3 => simd_shuffle8(a, b, [3, 0, 10, 11, 7, 4, 14, 15]),
            0xE4 => simd_shuffle8(a, b, [0, 1, 10, 11, 4, 5, 14, 15]),
            0xE5 => simd_shuffle8(a, b, [1, 1, 10, 11, 5, 5, 14, 15]),
            0xE6 => simd_shuffle8(a, b, [2, 1, 10, 11, 6, 5, 14, 15]),
            0xE7 => simd_shuffle8(a, b, [3, 1, 10, 11, 7, 5, 14, 15]),
            0xE8 => simd_shuffle8(a, b, [0, 2, 10, 11, 4, 6, 14, 15]),
            0xE9 => simd_shuffle8(a, b, [1, 2, 10, 11, 5, 6, 14, 15]),
            0xEA => simd_shuffle8(a, b, [2, 2, 10, 11, 6, 6, 14, 15]),
            0xEB => simd_shuffle8(a, b, [3, 2, 10, 11, 7, 6, 14, 15]),
            0xEC => simd_shuffle8(a, b, [0, 3, 10, 11, 4, 7, 14, 15]),
            0xED => simd_shuffle8(a, b, [1, 3, 10, 11, 5, 7, 14, 15]),
            0xEE => simd_shuffle8(a, b, [2, 3, 10, 11, 6, 7, 14, 15]),
            0xEF => simd_shuffle8(a, b, [3, 3, 10, 11, 7, 7, 14, 15]),
            0xF0 => simd_shuffle8(a, b, [0, 0, 11, 11, 4, 4, 15, 15]),
            0xF1 => simd_shuffle8(a, b, [1, 0, 11, 11, 5, 4, 15, 15]),
            0xF2 => simd_shuffle8(a, b, [2, 0, 11, 11, 6, 4, 15, 15]),
            0xF3 => simd_shuffle8(a, b, [3, 0, 11, 11, 7, 4, 15, 15]),
            0xF4 => simd_shuffle8(a, b, [0, 1, 11, 11, 4, 5, 15, 15]),
            0xF5 => simd_shuffle8(a, b, [1, 1, 11, 11, 5, 5, 15, 15]),
            0xF6 => simd_shuffle8(a, b, [2, 1, 11, 11, 6, 5, 15, 15]),
            0xF7 => simd_shuffle8(a, b, [3, 1, 11, 11, 7, 5, 15, 15]),
            0xF8 => simd_shuffle8(a, b, [0, 2, 11, 11, 4, 6, 15, 15]),
            0xF9 => simd_shuffle8(a, b, [1, 2, 11, 11, 5, 6, 15, 15]),
            0xFA => simd_shuffle8(a, b, [2, 2, 11, 11, 6, 6, 15, 15]),
            0xFB => simd_shuffle8(a, b, [3, 2, 11, 11, 7, 6, 15, 15]),
            0xFC => simd_shuffle8(a, b, [0, 3, 11, 11, 4, 7, 15, 15]),
            0xFD => simd_shuffle8(a, b, [1, 3, 11, 11, 5, 7, 15, 15]),
            0xFE => simd_shuffle8(a, b, [2, 3, 11, 11, 6, 7, 15, 15]),
            0xFF => simd_shuffle8(a, b, [3, 3, 11, 11, 7, 7, 15, 15]),
            _ => unreachable!()
        }
    }
}

// vsqrtpd
// __m256d _mm256_sqrt_pd (__m256d a)
#[inline]
pub fn mm256_sqrt_pd(a: m256d) -> m256d {
    unsafe { x86_mm256_sqrt_pd(a) }
}

// vsqrtps
// __m256 _mm256_sqrt_ps (__m256 a)
#[inline]
pub fn mm256_sqrt_ps(a: m256) -> m256 {
    unsafe { x86_mm256_sqrt_ps(a) }
}

// TODO(mayah): Implement these.
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
#[inline]
#[allow(unused_variables)]
pub fn mm256_stream_pd(mem_addr: *mut f64, a: m256d) {
    unimplemented!()
}

// vmovntps
// void _mm256_stream_ps (float * mem_addr, __m256 a)
#[inline]
#[allow(unused_variables)]
pub fn mm256_stream_ps(mem_addr: *mut f32, a: m256) {
    unimplemented!()
}

// vmovntdq
// void _mm256_stream_si256 (__m256i * mem_addr, __m256i a)
#[inline]
#[allow(unused_variables)]
pub fn mm256_stream_si256(mem_addr: *mut m256i, a: m256i) {
    unimplemented!()
}

// vsubpd
// __m256d _mm256_sub_pd (__m256d a, __m256d b)
#[inline]
pub fn mm256_sub_pd(a: m256d, b: m256d) -> m256d {
    unsafe { simd_sub(a, b) }
}

// vsubps
// __m256 _mm256_sub_ps (__m256 a, __m256 b)
#[inline]
pub fn mm256_sub_ps(a: m256, b: m256) -> m256 {
    unsafe { simd_sub(a, b) }
}

// vtestpd
// int _mm_testc_pd (__m128d a, __m128d b)
#[inline]
pub fn mm_testc_pd(a: m128d, b: m128d) -> i32 {
    unsafe { x86_mm_testc_pd(a, b) }
}

// vtestpd
// int _mm256_testc_pd (__m256d a, __m256d b)
#[inline]
pub fn mm256_testc_pd(a: m256d, b: m256d) -> i32 {
    unsafe { x86_mm256_testc_pd(a, b) }
}

// vtestps
// int _mm_testc_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_testc_ps(a: m128, b: m128) -> i32 {
    unsafe { x86_mm_testc_ps(a, b) }
}

// vtestps
// int _mm256_testc_ps (__m256 a, __m256 b)
#[inline]
pub fn mm256_testc_ps(a: m256, b: m256) -> i32 {
    unsafe { x86_mm256_testc_ps(a, b) }
}

// vptest
// int _mm256_testc_si256 (__m256i a, __m256i b)
#[inline]
pub fn mm256_testc_si256(a: m256i, b: m256i) -> i32 {
    unsafe { x86_mm256_testc_si256(a.as_u64x4(), b.as_u64x4()) }
}

// vtestpd
// int _mm_testnzc_pd (__m128d a, __m128d b)
#[inline]
pub fn mm_testnzc_pd(a: m128d, b: m128d) -> i32 {
    unsafe { x86_mm_testnzc_pd(a, b) }
}

// vtestpd
// int _mm256_testnzc_pd (__m256d a, __m256d b)
#[inline]
pub fn mm256_testnzc_pd(a: m256d, b: m256d) -> i32 {
    unsafe { x86_mm256_testnzc_pd(a, b) }
}

// vtestps
// int _mm_testnzc_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_testnzc_ps(a: m128, b: m128) -> i32 {
    unsafe { x86_mm_testnzc_ps(a, b) }
}

// vtestps
// int _mm256_testnzc_ps (__m256 a, __m256 b)
#[inline]
pub fn mm256_testnzc_ps(a: m256, b: m256) -> i32 {
    unsafe { x86_mm256_testnzc_ps(a, b) }
}

// vptest
// int _mm256_testnzc_si256 (__m256i a, __m256i b)
#[inline]
pub fn mm256_testnzc_si256(a: m256i, b: m256i) -> i32 {
    unsafe { x86_mm256_testnzc_si256(a.as_u64x4(), b.as_u64x4()) }
}

// vtestpd
// int _mm_testz_pd (__m128d a, __m128d b)
#[inline]
pub fn mm_testz_pd(a: m128d, b: m128d) -> i32 {
    unsafe { x86_mm_testz_pd(a, b) }
}

// vtestpd
// int _mm256_testz_pd (__m256d a, __m256d b)
#[inline]
pub fn mm256_testz_pd(a: m256d, b: m256d) -> i32 {
    unsafe { x86_mm256_testz_pd(a, b) }
}

// vtestps
// int _mm_testz_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_testz_ps(a: m128, b: m128) -> i32 {
    unsafe { x86_mm_testz_ps(a, b) }
}

// vtestps
// int _mm256_testz_ps (__m256 a, __m256 b)
#[inline]
pub fn mm256_testz_ps(a: m256, b: m256) -> i32 {
    unsafe { x86_mm256_testz_ps(a, b) }
}

// vptest
// int _mm256_testz_si256 (__m256i a, __m256i b)
#[inline]
pub fn mm256_testz_si256(a: m256i, b: m256i) -> i32 {
    unsafe { x86_mm256_testz_si256(a.as_u64x4(), b.as_u64x4()) }
}

// __m128d _mm_undefined_pd (void)
#[inline]
pub fn mm_undefined_pd() -> m128d {
    unsafe { std::mem::uninitialized() }
}

// __m256d _mm256_undefined_pd (void)
#[inline]
pub fn mm256_undefined_pd() -> m256d {
    unsafe { std::mem::uninitialized() }
}

// __m128 _mm_undefined_ps (void)
#[inline]
pub fn mm_undefined_ps() -> m128 {
    unsafe { std::mem::uninitialized() }
}

// __m256 _mm256_undefined_ps (void)
#[inline]
pub fn mm256_undefined_ps() -> m256 {
    unsafe { std::mem::uninitialized() }
}

// __m128i _mm_undefined_si128 (void)
#[inline]
pub fn mm_undefined_si128() -> m128i {
    unsafe { std::mem::uninitialized() }
}

// __m256i _mm256_undefined_si256 (void)
#[inline]
pub fn mm256_undefined_si256() -> m256i {
    unsafe { std::mem::uninitialized() }
}

// vunpckhpd
// __m256d _mm256_unpackhi_pd (__m256d a, __m256d b)
#[inline]
pub fn mm256_unpackhi_pd(a: m256d, b: m256d) -> m256d {
    unsafe { simd_shuffle4(a, b, [1, 5, 3, 7]) }
}

// vunpckhps
// __m256 _mm256_unpackhi_ps (__m256 a, __m256 b)
#[inline]
pub fn mm256_unpackhi_ps(a: m256, b: m256) -> m256 {
    unsafe { simd_shuffle8(a, b, [2, 10, 3, 11, 6, 14, 7, 15]) }
}

// vunpcklpd
// __m256d _mm256_unpacklo_pd (__m256d a, __m256d b)
#[inline]
pub fn mm256_unpacklo_pd(a: m256d, b: m256d) -> m256d {
    unsafe { simd_shuffle4(a, b, [0, 4, 2, 6]) }
}

// vunpcklps
// __m256 _mm256_unpacklo_ps (__m256 a, __m256 b)
#[inline]
pub fn mm256_unpacklo_ps(a: m256, b: m256) -> m256 {
    unsafe { simd_shuffle8(a, b, [0, 8, 1, 9, 4, 12, 5, 13]) }
}

// vxorpd
// __m256d _mm256_xor_pd (__m256d a, __m256d b)
#[inline]
pub fn mm256_xor_pd(a: m256d, b: m256d) -> m256d {
    let ai = a.as_m256i();
    let bi = b.as_m256i();
    unsafe { simd_xor(ai, bi).as_m256d() }
}
// vxorps
// __m256 _mm256_xor_ps (__m256 a, __m256 b)
#[inline]
pub fn mm256_xor_ps(a: m256, b: m256) -> m256 {
    let ai = a.as_m256i();
    let bi = b.as_m256i();
    unsafe { simd_xor(ai, bi).as_m256() }
}

// vzeroall
// void _mm256_zeroall (void)
#[inline]
pub fn mm256_zeroall() {
    unsafe { avx_vzeroall() }
}

// vzeroupper
// void _mm256_zeroupper (void)
#[inline]
pub fn mm256_zeroupper() {
    unsafe { avx_vzeroupper() }
}

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

    #[test]
    fn test_broadcast() {
        let xpd = mm_setr_pd(1.0, 2.0);
        let xps = mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let xf32: f32 = 1.0;
        let xf64: f64 = 2.0;

        let r = unsafe { mm256_broadcast_pd(&xpd as *const m128d) };
        assert_eq!(r.as_f64x4().as_array(), [1.0, 2.0, 1.0, 2.0]);

        let r = unsafe { mm256_broadcast_ps(&xps as *const m128) };
        assert_eq!(r.as_f32x8().as_array(), [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);

        let r = unsafe { mm256_broadcast_sd(&xf64 as *const f64) };
        assert_eq!(r.as_f64x4().as_array(), [2.0, 2.0, 2.0, 2.0]);

        let r = unsafe { mm_broadcast_ss(&xf32 as *const f32) };
        assert_eq!(r.as_f32x4().as_array(), [1.0, 1.0, 1.0, 1.0]);

        let r = unsafe { mm256_broadcast_ss(&xf32 as *const f32) };
        assert_eq!(r.as_f32x8().as_array(), [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_mm256_arith() {
        let aps = mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let bps = mm256_set1_ps(2.0);
        let apd = mm256_setr_pd(1.0, 2.0, 3.0, 4.0);
        let bpd = mm256_set1_pd(2.0);

        assert_eq!(mm256_add_ps(aps, bps).as_f32x8().as_array(), [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        assert_eq!(mm256_sub_ps(aps, bps).as_f32x8().as_array(), [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(mm256_mul_ps(aps, bps).as_f32x8().as_array(), [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        assert_eq!(mm256_div_ps(aps, bps).as_f32x8().as_array(), [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]);

        assert_eq!(mm256_addsub_ps(aps, bps).as_f32x8().as_array(), [-1.0, 4.0, 1.0, 6.0, 3.0, 8.0, 5.0, 10.0]);

        assert_eq!(mm256_add_pd(apd, bpd).as_f64x4().as_array(), [3.0, 4.0, 5.0, 6.0]);
        assert_eq!(mm256_sub_pd(apd, bpd).as_f64x4().as_array(), [-1.0, 0.0, 1.0, 2.0]);
        assert_eq!(mm256_mul_pd(apd, bpd).as_f64x4().as_array(), [2.0, 4.0, 6.0, 8.0]);
        assert_eq!(mm256_div_pd(apd, bpd).as_f64x4().as_array(), [0.5, 1.0, 1.5, 2.0]);

        assert_eq!(mm256_addsub_pd(apd, bpd).as_f64x4().as_array(), [-1.0, 4.0, 1.0, 6.0]);

        assert_eq!(mm256_hadd_pd(apd, bpd).as_f64x4().as_array(), [3.0, 4.0, 7.0, 4.0]);
        assert_eq!(mm256_hsub_pd(apd, bpd).as_f64x4().as_array(), [-1.0, 0.0, -1.0, 0.0]);

        assert_eq!(mm256_hadd_ps(aps, bps).as_f32x8().as_array(), [3.0, 7.0, 4.0, 4.0, 11.0, 15.0, 4.0, 4.0]);
        assert_eq!(mm256_hsub_ps(aps, bps).as_f32x8().as_array(), [-1.0, -1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_mm256_logic_pd() {
        let x = i64x4(0x1, 0x2, 0x3, 0x4).as_m256d();
        let y = i64x4(0x3, 0x4, 0x5, 0x6).as_m256d();

        assert_eq!(mm256_and_pd(x, y).as_m256i().as_i64x4().as_array(),
                   [0x1 & 0x3, 0x2 & 0x4, 0x3 & 0x5, 0x4 & 0x6]);
        assert_eq!(mm256_or_pd(x, y).as_m256i().as_i64x4().as_array(),
                   [0x1 | 0x3, 0x2 | 0x4, 0x3 | 0x5, 0x4 | 0x6]);
        assert_eq!(mm256_xor_pd(x, y).as_m256i().as_i64x4().as_array(),
                   [0x1 ^ 0x3, 0x2 ^ 0x4, 0x3 ^ 0x5, 0x4 ^ 0x6]);
        assert_eq!(mm256_andnot_pd(x, y).as_m256i().as_i64x4().as_array(),
                   [!0x1 & 0x3, !0x2 & 0x4, !0x3 & 0x5, !0x4 & 0x6]);
    }

    #[test]
    fn test_mm256_logic_ps() {
        let x = i32x8(0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8).as_m256();
        let y = i32x8(0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA).as_m256();

        assert_eq!(mm256_and_ps(x, y).as_m256i().as_i32x8().as_array(),
                   [0x1 & 0x3, 0x2 & 0x4, 0x3 & 0x5, 0x4 & 0x6, 0x5 & 0x7, 0x6 & 0x8, 0x7 & 0x9, 0x8 & 0xA]);
        assert_eq!(mm256_or_ps(x, y).as_m256i().as_i32x8().as_array(),
                   [0x1 | 0x3, 0x2 | 0x4, 0x3 | 0x5, 0x4 | 0x6, 0x5 | 0x7, 0x6 | 0x8, 0x7 | 0x9, 0x8 | 0xA]);
        assert_eq!(mm256_xor_ps(x, y).as_m256i().as_i32x8().as_array(),
                   [0x1 ^ 0x3, 0x2 ^ 0x4, 0x3 ^ 0x5, 0x4 ^ 0x6, 0x5 ^ 0x7, 0x6 ^ 0x8, 0x7 ^ 0x9, 0x8 ^ 0xA]);
        assert_eq!(mm256_andnot_ps(x, y).as_m256i().as_i32x8().as_array(),
                   [!0x1 & 0x3, !0x2 & 0x4, !0x3 & 0x5, !0x4 & 0x6, !0x5 & 0x7, !0x6 & 0x8, !0x7 & 0x9, !0x8 & 0xA]);
    }

    #[test]
    fn test_mm256_castpd128_pd256() {
        // TODO(mayah): mm256_castpd128_pd256 is not implemented yet.

        // let xpd = mm_setr_pd(1.0, 2.0);
        // let x256 = mm256_castpd128_pd256(xpd).as_f64x4().as_array();
        // assert_eq!(x256[0], 1.0);
        // assert_eq!(x256[1], 2.0);
    }

    #[test]
    fn test_mm256_castpd256_pd128() {
        let x = mm256_setr_pd(1.0, 2.0, 3.0, 4.0);
        assert_eq!(mm256_castpd256_pd128(x).as_f64x2().as_array(), [1.0, 2.0]);
    }

    #[test]
    fn test_mm256_blend_pd() {
        let a = mm256_setr_pd(0.0, 1.0, 2.0, 3.0);
        let b = mm256_setr_pd(4.0, 5.0, 6.0, 7.0);

        assert_eq!(mm256_blend_pd(a, b, 0x0).as_f64x4().as_array(), [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(mm256_blend_pd(a, b, 0x3).as_f64x4().as_array(), [4.0, 5.0, 2.0, 3.0]);
        assert_eq!(mm256_blend_pd(a, b, 0xF).as_f64x4().as_array(), [4.0, 5.0, 6.0, 7.0]);

        assert_eq!(mm256_blendv_pd(a, b, i64x4(0, 0, 0, 0).as_m256i().as_m256d()).as_f64x4().as_array(), [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(mm256_blendv_pd(a, b, i64x4(!0, !0, 0, 0).as_m256i().as_m256d()).as_f64x4().as_array(), [4.0, 5.0, 2.0, 3.0]);
        assert_eq!(mm256_blendv_pd(a, b, i64x4(!0, !0, !0, !0).as_m256i().as_m256d()).as_f64x4().as_array(), [4.0, 5.0, 6.0, 7.0]);

    }

    #[test]
    fn test_mm256_blend_ps() {
        let a = mm256_setr_ps(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
        let b = mm256_setr_ps(8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0);

        assert_eq!(mm256_blend_ps(a, b, 0x00).as_f32x8().as_array(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert_eq!(mm256_blend_ps(a, b, 0x33).as_f32x8().as_array(), [8.0, 9.0, 2.0, 3.0, 12.0, 13.0, 6.0, 7.0]);
        assert_eq!(mm256_blend_ps(a, b, 0xFF).as_f32x8().as_array(), [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);

        assert_eq!(mm256_blendv_ps(a, b, i32x8(0, 0, 0, 0, 0, 0, 0, 0).as_m256i().as_m256()).as_f32x8().as_array(),
                   [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert_eq!(mm256_blendv_ps(a, b, i32x8(!0, !0, 0, 0, !0, !0, 0, 0).as_m256i().as_m256()).as_f32x8().as_array(),
                   [8.0, 9.0, 2.0, 3.0, 12.0, 13.0, 6.0, 7.0]);
        assert_eq!(mm256_blendv_ps(a, b, i32x8(!0, !0, !0, !0, !0, !0, !0, !0).as_m256i().as_m256()).as_f32x8().as_array(),
                   [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
    }

    #[test]
    fn test_cast() {
        let apd256 = mm256_castpd128_pd256(mm_setr_pd(1.0, 2.0));
        assert_eq!(apd256.as_f64x4().extract(0), 1.0);
        assert_eq!(apd256.as_f64x4().extract(1), 2.0);

        let aps256 = mm256_castps128_ps256(mm_setr_ps(1.0, 2.0, 3.0, 4.0));
        assert_eq!(aps256.as_f32x8().extract(0), 1.0);
        assert_eq!(aps256.as_f32x8().extract(1), 2.0);
        assert_eq!(aps256.as_f32x8().extract(2), 3.0);
        assert_eq!(aps256.as_f32x8().extract(3), 4.0);

        let asi256 = mm256_castsi128_si256(i64x2(1, 2).as_m128i());
        assert_eq!(asi256.as_i64x4().extract(0), 1);
        assert_eq!(asi256.as_i64x4().extract(1), 2);
    }

    #[test]
    fn test_extract() {
        let a = mm256_setr_epi8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
                                0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                                0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F);
        assert_eq!(mm256_extract_epi8(a, 3), 0x03);
        assert_eq!(mm256_extract_epi16(a, 3), 0x0706);
        assert_eq!(mm256_extract_epi32(a, 3), 0x0F0E0D0C);
        assert_eq!(mm256_extract_epi64(a, 3), 0x1F1E1D1C1B1A1918);
    }

    #[test]
    fn test_insert() {
        let a8 = mm256_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                  17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);
        let a16 = mm256_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a32 = mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a64 = mm256_setr_epi64x(1, 2, 3, 4);

        assert_eq!(mm256_insert_epi8(a8, 100, 0).as_i8x32().as_array(),
                   [100, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]);
        assert_eq!(mm256_insert_epi16(a16, 100, 0).as_i16x16().as_array(),
                   [100, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        assert_eq!(mm256_insert_epi32(a32, 100, 0).as_i32x8().as_array(), [100, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(mm256_insert_epi64(a64, 100, 0).as_i64x4().as_array(), [100, 2, 3, 4]);
    }

    #[test]
    fn test_cmp_pd() {
        // let apd = mm_setr_pd(1.0, 2.0);
        // let bpd = mm_setr_pd(2.0, 2.0);
        // let apd256 = mm256_setr_pd(1.0, 2.0, 3.0, 4.0);
        // let bpd256 = mm256_setr_pd(3.0, 3.0, 3.0, 3.0);
        //
        // assert_eq!(mm_cmp_pd(apd, bpd, CMP_NLE_US).as_m128i().as_i64x2().as_array(),
        //            [!0, !0]);
        // assert_eq!(mm_cmp_pd(apd, bpd, CMP_NLT_US).as_m128i().as_i64x2().as_array(),
        //            [!0, 0]);
        // assert_eq!(mm_cmp_sd(apd, bpd, CMP_NLE_US).as_m128i().as_i64x2().extract(0), !0);
        // assert_eq!(mm256_cmp_pd(apd256, bpd256, CMP_NLE_US).as_m256i().as_i64x4().as_array(),
        //            [!0, !0, !0, 0]);
        // assert_eq!(mm256_cmp_pd(apd256, bpd256, CMP_NLT_US).as_m256i().as_i64x4().as_array(),
        //            [!0, !0, 0, 0]);
    }

    #[test]
    fn test_cmp_ps() {
        // let aps = mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        // let bps = mm_setr_ps(3.0, 3.0, 3.0, 3.0);
        // let aps256 = mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        // let bps256 = mm256_setr_ps(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0);
        //
        // assert_eq!(mm_cmp_ps(aps, bps, CMP_NLE_US).as_m128i().as_i32x4().as_array(),
        //            [!0, !0, !0, 0]);
        // assert_eq!(mm_cmp_ps(aps, bps, CMP_NLT_US).as_m128i().as_i32x4().as_array(),
        //            [!0, !0, 0, 0]);
        // assert_eq!(mm_cmp_ss(aps, bps, CMP_NLE_US).as_m128i().as_i32x4().extract(0), !0);
        // assert_eq!(mm256_cmp_ps(aps256, bps256, CMP_NLE_US).as_m256i().as_i32x8().as_array(),
        //            [!0, !0, !0, 0, 0, 0, 0, 0]);
        // assert_eq!(mm256_cmp_ps(aps256, bps256, CMP_NLT_US).as_m256i().as_i32x8().as_array(),
        //            [!0, !0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_mm256_dp_ps() {
        let a = mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = mm256_setr_ps(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);

        let t1 = 1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0 + 4.0 * 5.0;
        let t2 = 5.0 * 6.0 + 6.0 * 7.0 + 7.0 * 8.0 + 8.0 * 9.0;

        assert_eq!(mm256_dp_ps(a, b, 0xFF).as_f32x8().as_array(),
                   [t1, t1, t1, t1, t2, t2, t2, t2]);
    }

    #[test]
    fn test_mm256_extractf128_pd() {
        let a = mm256_setr_pd(1.0, 2.0, 3.0, 4.0);
        assert_eq!(mm256_extractf128_pd(a, 0).as_f64x2().as_array(), [1.0, 2.0]);
        assert_eq!(mm256_extractf128_pd(a, 1).as_f64x2().as_array(), [3.0, 4.0]);
    }

    #[test]
    fn test_mm256_extractf128_ps() {
        let a = mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq!(mm256_extractf128_ps(a, 0).as_f32x4().as_array(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(mm256_extractf128_ps(a, 1).as_f32x4().as_array(), [5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_mm256_extractf128_si256() {
        let a = mm256_setr_epi64x(1, 2, 3, 4);
        assert_eq!(mm256_extractf128_si256(a, 0).as_i64x2().as_array(), [1, 2]);
        assert_eq!(mm256_extractf128_si256(a, 1).as_i64x2().as_array(), [3, 4]);
    }

    #[test]
    fn test_mm256_insertf256() {
        let apd256 = mm256_setr_pd(1.0, 2.0, 3.0, 4.0);
        let apd128 = mm_setr_pd(5.0, 6.0);

        let aps256 = mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let aps128 = mm_setr_ps(9.0, 10.0, 11.0, 12.0);

        let asi256 = mm256_setr_epi64x(1, 2, 3, 4);
        let asi128 = mm_set_epi64x(6, 5);

        assert_eq!(mm256_insertf128_pd(apd256, apd128, 1).as_f64x4().as_array(), [1.0, 2.0, 5.0, 6.0]);
        assert_eq!(mm256_insertf128_ps(aps256, aps128, 1).as_f32x8().as_array(),
                   [1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0]);
        assert_eq!(mm256_insertf128_si256(asi256, asi128, 1).as_i64x4().as_array(), [1, 2, 5, 6]);
    }

    #[test]
    fn test_lddqu() {
        let x = mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let p = &x as *const m256i;

        let r = unsafe { mm256_lddqu_si256(p) };
        assert_eq!(r.as_i32x8().as_array(), [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_load() {
        // mm256_load_* must be 32byte aligned.
        // TODO(mayah): Can we make sure alignment in rust? Using m256, m256d, and m256i
        // is OK, since they should be 32byte aligned.
        let xf64 = mm256_setr_pd(1.0, 2.0, 3.0, 4.0);
        let xf32 = mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let xsi = mm256_setr_epi64x(1, 2, 3, 4);

        let p_pd = &xf64 as *const m256d as *const f64;
        let p_ps = &xf32 as *const m256 as *const f32;
        let p_si = &xsi as *const m256i;

        unsafe {
            assert_eq!(mm256_load_pd(p_pd).as_f64x4().as_array(), [1.0, 2.0, 3.0, 4.0]);
            assert_eq!(mm256_loadu_pd(p_pd).as_f64x4().as_array(), [1.0, 2.0, 3.0, 4.0]);

            assert_eq!(mm256_load_ps(p_ps).as_f32x8().as_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            assert_eq!(mm256_loadu_ps(p_ps).as_f32x8().as_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

            assert_eq!(mm256_load_si256(p_si).as_i64x4().as_array(), [1, 2, 3, 4]);
            assert_eq!(mm256_loadu_si256(p_si).as_i64x4().as_array(), [1, 2, 3, 4]);
        }
    }

    #[test]
    fn test_loadu2() {
        let xf64_lo: [f64; 2] = [1.0, 2.0];
        let xf64_hi: [f64; 2] = [3.0, 4.0];
        let xf32_lo: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let xf32_hi: [f32; 4] = [5.0, 6.0, 7.0, 8.0];
        let xsi_lo: [i64; 2] = [1, 2];
        let xsi_hi: [i64; 2] = [3, 4];

        let p_pd_lo = &xf64_lo as *const f64;
        let p_pd_hi = &xf64_hi as *const f64;
        let p_ps_lo = &xf32_lo as *const f32;
        let p_ps_hi = &xf32_hi as *const f32;
        let p_si_lo = &xsi_lo as *const i64 as *const m128i;
        let p_si_hi = &xsi_hi as *const i64 as *const m128i;

        unsafe {
            assert_eq!(mm256_loadu2_m128d(p_pd_hi, p_pd_lo).as_f64x4().as_array(),
                       [1.0, 2.0, 3.0, 4.0]);
            assert_eq!(mm256_loadu2_m128(p_ps_hi, p_ps_lo).as_f32x8().as_array(),
                       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            assert_eq!(mm256_loadu2_m128i(p_si_hi, p_si_lo).as_i64x4().as_array(),
                       [1, 2, 3, 4]);
        }
    }

    #[test]
    fn test_maskload() {
        let a64: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
        let a32: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let p64 = &a64 as *const f64;
        let p32 = &a32 as *const f32;

        let mps128 = mm_setr_epi32(0, !0, 0, !0);
        let mpd128 = i64x2(0, !0).as_m128i();
        let mps256 = mm256_setr_epi32(0, !0, 0, !0, 0, !0, 0, !0);
        let mpd256 = mm256_setr_epi64x(0, !0, 0, !0);

        unsafe {
            assert_eq!(mm_maskload_pd(p64, mpd128).as_f64x2().as_array(), [0.0, 2.0]);
            assert_eq!(mm_maskload_ps(p32, mps128).as_f32x4().as_array(), [0.0, 2.0, 0.0, 4.0]);
            assert_eq!(mm256_maskload_pd(p64, mpd256).as_f64x4().as_array(), [0.0, 2.0, 0.0, 4.0]);
            assert_eq!(mm256_maskload_ps(p32, mps256).as_f32x8().as_array(), [0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0]);
        }
    }

    #[test]
    fn test_maskstore() {
        let apd128 = mm_setr_pd(1.0, 2.0);
        let aps128 = mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let apd256 = mm256_setr_pd(1.0, 2.0, 3.0, 4.0);
        let aps256 = mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);

        let mps128 = mm_setr_epi32(0, !0, 0, !0);
        let mpd128 = i64x2(0, !0).as_m128i();
        let mps256 = mm256_setr_epi32(0, !0, 0, !0, 0, !0, 0, !0);
        let mpd256 = mm256_setr_epi64x(0, !0, 0, !0);

        unsafe {
            let mut x: [f64; 2] = [0.0; 2];
            let p = &mut x as *mut [f64] as *mut f64;

            mm_maskstore_pd(p, mpd128, apd128);
            assert_eq!(x, [0.0, 2.0]);
        }
        unsafe {
            let mut x: [f32; 4] = [0.0; 4];
            let p = &mut x as *mut [f32] as *mut f32;

            mm_maskstore_ps(p, mps128, aps128);
            assert_eq!(x, [0.0, 2.0, 0.0, 4.0]);
        }
        unsafe {
            let mut x: [f64; 4] = [0.0; 4];
            let p = &mut x as *mut [f64] as *mut f64;

            mm256_maskstore_pd(p, mpd256, apd256);
            assert_eq!(x, [0.0, 2.0, 0.0, 4.0]);
        }
        unsafe {
            let mut x: [f32; 8] = [0.0; 8];
            let p = &mut x as *mut [f32] as *mut f32;

            mm256_maskstore_ps(p, mps256, aps256);
            assert_eq!(x, [0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0]);
        }
    }

    #[test]
    fn test_max_min_pd() {
        let a = mm256_setr_pd(1.0, 2.0, 3.0, 4.0);
        let b = mm256_setr_pd(3.0, 2.0, 1.0, 2.0);
        assert_eq!(mm256_max_pd(a, b).as_f64x4().as_array(), [3.0, 2.0, 3.0, 4.0]);
        assert_eq!(mm256_min_pd(a, b).as_f64x4().as_array(), [1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_max_min_ps() {
        let a = mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = mm256_setr_ps(3.0, 2.0, 1.0, 2.0, 3.0, 1.0, 8.0, 9.0);
        assert_eq!(mm256_max_ps(a, b).as_f32x8().as_array(),
                   [3.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0]);
        assert_eq!(mm256_min_ps(a, b).as_f32x8().as_array(),
                   [1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 7.0, 8.0]);
    }

    #[test]
    fn test_movedup_pd() {
        let a = mm256_setr_pd(1.0, 2.0, 3.0, 4.0);
        assert_eq!(mm256_movedup_pd(a).as_f64x4().as_array(),
                   [1.0, 1.0, 3.0, 3.0]);
    }

    #[test]
    fn test_movedup_ps() {
        let a = mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq!(mm256_moveldup_ps(a).as_f32x8().as_array(),
                   [1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 7.0, 7.0]);
        assert_eq!(mm256_movehdup_ps(a).as_f32x8().as_array(),
                   [2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0]);
    }

    #[test]
    fn test_movemask() {
        let apd = mm256_setr_pd(1.0, -2.0, 3.0, -4.0);
        let aps = mm256_setr_ps(1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0);

        assert_eq!(mm256_movemask_pd(apd), 0xA);
        assert_eq!(mm256_movemask_ps(aps), 0xAA);
    }

    #[test]
    fn test_math() {
        let aps = mm256_setr_ps(1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0);
        let apd = mm256_setr_pd(1.0, 4.0, 9.0, 16.0);

        let apd_sqrt = mm256_sqrt_pd(apd).as_f64x4().as_array();
        let aps_sqrt = mm256_sqrt_ps(aps).as_f32x8().as_array();
        let aps_rsqrt = mm256_rsqrt_ps(aps).as_f32x8().as_array();
        let aps_rcp = mm256_rcp_ps(aps).as_f32x8().as_array();

        assert!((apd_sqrt[0] - 1.0).abs() < 0.001);
        assert!((apd_sqrt[1] - 2.0).abs() < 0.001);
        assert!((aps_sqrt[0] - 1.0).abs() < 0.001);
        assert!((aps_sqrt[1] - 2.0).abs() < 0.001);
        assert!((aps_rsqrt[0] - 1.0).abs() < 0.001);
        assert!((aps_rsqrt[1] - 1.0 / 2.0).abs() < 0.001);
        assert!((aps_rcp[0] - 1.0).abs() < 0.001);
        assert!((aps_rcp[1] - 1.0 / 4.0).abs() < 0.001);
    }

    #[test]
    fn test_rounding() {
        let ps = mm256_setr_ps(1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5);
        let pd = mm256_setr_pd(1.5, 2.5, 3.5, 4.5);

        assert_eq!(mm256_ceil_pd(pd).as_f64x4().as_array(), [2.0, 3.0, 4.0, 5.0]);
        assert_eq!(mm256_floor_pd(pd).as_f64x4().as_array(), [1.0, 2.0, 3.0, 4.0]);

        assert_eq!(mm256_ceil_ps(ps).as_f32x8().as_array(), [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(mm256_floor_ps(ps).as_f32x8().as_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_unpack_pd() {
        let a = mm256_setr_pd(1.0, 2.0, 3.0, 4.0);
        let b = mm256_setr_pd(5.0, 6.0, 7.0, 8.0);

        assert_eq!(mm256_unpacklo_pd(a, b).as_f64x4().as_array(), [1.0, 5.0, 3.0, 7.0]);
        assert_eq!(mm256_unpackhi_pd(a, b).as_f64x4().as_array(), [2.0, 6.0, 4.0, 8.0]);
    }

    #[test]
    fn test_unpack_ps() {
        let a = mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = mm256_setr_ps(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);

        assert_eq!(mm256_unpacklo_ps(a, b).as_f32x8().as_array(),
                   [1.0, 9.0, 2.0, 10.0, 5.0, 13.0, 6.0, 14.0]);
        assert_eq!(mm256_unpackhi_ps(a, b).as_f32x8().as_array(),
                   [3.0, 11.0, 4.0, 12.0, 7.0, 15.0, 8.0, 16.0]);
    }

    #[test]
    fn test_testc_si() {
        let x = mm256_setr_epi32(0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7);
        let y = mm256_setr_epi32(0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3);
        let z = mm256_setr_epi32(0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8);

        assert_eq!(mm256_testc_si256(x, y), 1);
        assert_eq!(mm256_testc_si256(x, z), 0);
    }

    #[test]
    fn test_testz_si() {
        let x = mm256_setr_epi32(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0);
        let y = mm256_setr_epi32(0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1);
        let z = mm256_setr_epi32(0x2, 0x2, 0x2, 0x2, 0x2, 0x2, 0x2, 0x2);

        assert_eq!(mm256_testz_si256(x, x), 1);
        assert_eq!(mm256_testz_si256(y, y), 0);
        assert_eq!(mm256_testz_si256(y, z), 1);
    }

    #[test]
    fn test_testnzc_si() {
        let x = mm256_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0);
        let y = mm256_setr_epi32(0, 0, 0, 0, 1, 0, 0, 0);
        let z = mm256_setr_epi32(0, 0, 0, 0, !0, !0, 0, 0);

        assert_eq!(mm256_testnzc_si256(x, z), 0);
        assert_eq!(mm256_testnzc_si256(y, z), 1);
    }

    #[test]
    fn test_testc_p() {
        let ppd = mm_setr_pd(1.0, 1.0);
        let pps = mm_setr_ps(1.0, 1.0, 1.0, 1.0);
        let npd = mm_setr_pd(-1.0, -1.0);
        let nps = mm_setr_ps(-1.0, -1.0, -1.0, -1.0);

        let ppd256 = mm256_setr_pd(1.0, 1.0, 1.0, 1.0);
        let pps256 = mm256_setr_ps(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        let npd256 = mm256_setr_pd(-1.0, -1.0, -1.0, -1.0);
        let nps256 = mm256_setr_ps(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0);

        assert_eq!(mm_testc_ps(pps, pps), 1);
        assert_eq!(mm_testc_ps(nps, nps), 1);
        assert_eq!(mm_testc_ps(pps, nps), 0);
        assert_eq!(mm_testc_pd(ppd, ppd), 1);
        assert_eq!(mm_testc_pd(npd, npd), 1);
        assert_eq!(mm_testc_pd(ppd, npd), 0);
        assert_eq!(mm256_testc_ps(pps256, pps256), 1);
        assert_eq!(mm256_testc_ps(nps256, nps256), 1);
        assert_eq!(mm256_testc_ps(pps256, nps256), 0);
        assert_eq!(mm256_testc_pd(ppd256, ppd256), 1);
        assert_eq!(mm256_testc_pd(npd256, npd256), 1);
        assert_eq!(mm256_testc_pd(ppd256, npd256), 0);
    }

    #[test]
    fn test_testz_p() {
        let ppd = mm_setr_pd(1.0, 1.0);
        let pps = mm_setr_ps(1.0, 1.0, 1.0, 1.0);
        let npd = mm_setr_pd(-1.0, -1.0);
        let nps = mm_setr_ps(-1.0, -1.0, -1.0, -1.0);

        let ppd256 = mm256_setr_pd(1.0, 1.0, 1.0, 1.0);
        let pps256 = mm256_setr_ps(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        let npd256 = mm256_setr_pd(-1.0, -1.0, -1.0, -1.0);
        let nps256 = mm256_setr_ps(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0);

        assert_eq!(mm_testz_ps(pps, pps), 1);
        assert_eq!(mm_testz_ps(nps, nps), 0);
        assert_eq!(mm_testz_pd(ppd, ppd), 1);
        assert_eq!(mm_testz_pd(npd, npd), 0);
        assert_eq!(mm256_testz_ps(pps256, pps256), 1);
        assert_eq!(mm256_testz_ps(nps256, nps256), 0);
        assert_eq!(mm256_testz_pd(ppd256, ppd256), 1);
        assert_eq!(mm256_testz_pd(npd256, npd256), 0);
    }

    #[test]
    fn test_testnzc_p() {
        let ppd = mm_setr_pd(-1.0, 1.0);
        let pps = mm_setr_ps(-1.0, -1.0, 1.0, 1.0);
        let npd = mm_setr_pd(1.0, -1.0);
        let nps = mm_setr_ps(1.0, -1.0, -1.0, -1.0);

        let ppd256 = mm256_setr_pd(-1.0, -1.0, 1.0, 1.0);
        let pps256 = mm256_setr_ps(-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        let npd256 = mm256_setr_pd(1.0, -1.0, -1.0, -1.0);
        let nps256 = mm256_setr_ps(1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0);

        assert_eq!(mm_testnzc_ps(pps, pps), 0);
        assert_eq!(mm_testnzc_ps(nps, nps), 0);
        assert_eq!(mm_testnzc_ps(nps, pps), 1);
        assert_eq!(mm_testnzc_pd(ppd, ppd), 0);
        assert_eq!(mm_testnzc_pd(npd, npd), 0);
        assert_eq!(mm_testnzc_pd(npd, ppd), 0);
        assert_eq!(mm256_testnzc_ps(pps256, pps256), 0);
        assert_eq!(mm256_testnzc_ps(nps256, nps256), 0);
        assert_eq!(mm256_testnzc_ps(nps256, pps256), 1);
        assert_eq!(mm256_testnzc_pd(ppd256, ppd256), 0);
        assert_eq!(mm256_testnzc_pd(npd256, npd256), 0);
        assert_eq!(mm256_testnzc_pd(npd256, ppd256), 1);
    }

    #[test]
    fn test_shufle() {
        let s = mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let d = mm256_setr_pd(1.0, 2.0, 3.0, 4.0);

        assert_eq!(mm256_shuffle_ps(s, s, (3 << 0 | 0 << 2 | 1 << 4 | 2 << 6)).as_f32x8().as_array(),
                   [4.0, 1.0, 2.0, 3.0, 8.0, 5.0, 6.0, 7.0]);
        assert_eq!(mm256_shuffle_pd(d, d, (0 << 0 | 1 << 1 | 1 << 2 | 0 << 3)).as_f64x4().as_array(),
                   [1.0, 2.0, 4.0, 3.0]);
    }
}
