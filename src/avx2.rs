#![allow(improper_ctypes)]  // TODO(mayah): Remove this flag

use std;
use super::*;
use super::{simd_add, simd_sub, simd_mul,
            simd_and, simd_or, simd_xor,
            simd_eq, simd_gt,
            simd_shuffle2, simd_shuffle4, simd_shuffle8, simd_shuffle16, simd_shuffle32,
            simd_cast};

extern "platform-intrinsic" {
    fn x86_mm256_abs_epi8(x: i8x32) -> i8x32;
    fn x86_mm256_abs_epi16(x: i16x16) -> i16x16;
    fn x86_mm256_abs_epi32(x: i32x8) -> i32x8;

    fn x86_mm256_adds_epi8(x: i8x32, y: i8x32) -> i8x32;
    fn x86_mm256_adds_epu8(x: u8x32, y: u8x32) -> u8x32;
    fn x86_mm256_adds_epi16(x: i16x16, y: i16x16) -> i16x16;
    fn x86_mm256_adds_epu16(x: u16x16, y: u16x16) -> u16x16;
    fn x86_mm256_subs_epi8(x: i8x32, y: i8x32) -> i8x32;
    fn x86_mm256_subs_epu8(x: u8x32, y: u8x32) -> u8x32;
    fn x86_mm256_subs_epi16(x: i16x16, y: i16x16) -> i16x16;
    fn x86_mm256_subs_epu16(x: u16x16, y: u16x16) -> u16x16;

    fn x86_mm256_hadd_epi16(x: i16x16, y: i16x16) -> i16x16;
    fn x86_mm256_hadd_epi32(x: i32x8, y: i32x8) -> i32x8;
    fn x86_mm256_hadds_epi16(x: i16x16, y: i16x16) -> i16x16;
    fn x86_mm256_hsub_epi16(x: i16x16, y: i16x16) -> i16x16;
    fn x86_mm256_hsub_epi32(x: i32x8, y: i32x8) -> i32x8;
    fn x86_mm256_hsubs_epi16(x: i16x16, y: i16x16) -> i16x16;

    fn x86_mm256_avg_epu8(x: u8x32, y: u8x32) -> u8x32;
    fn x86_mm256_avg_epu16(x: u16x16, y: u16x16) -> u16x16;

    fn x86_mm_maskload_epi32(mem_addr: *const i32x4, mask: i32x4) -> i32x4;
    fn x86_mm_maskload_epi64(mem_addr: *const i64x2, mask: i64x2) -> i64x2;
    fn x86_mm256_maskload_epi32(mem_addr: *const i32x8, mask: i32x8) -> i32x8;
    fn x86_mm256_maskload_epi64(mem_addr: *const i64x4, mask: i64x4) -> i64x4;

    fn x86_mm_maskstore_epi32(mem_addr: *mut i32, mask: i32x4, a: i32x4);
    fn x86_mm_maskstore_epi64(mem_addr: *mut i64, mask: i64x2, a: i64x2);
    fn x86_mm256_maskstore_epi32(mem_addr: *mut i32, mask: i32x8, a: i32x8);
    fn x86_mm256_maskstore_epi64(mem_addr: *mut i64, mask: i64x4, a: i64x4);

    fn x86_mm256_max_epi16(x: i16x16, y: i16x16) -> i16x16;
    fn x86_mm256_max_epi32(x: i32x8, y: i32x8) -> i32x8;
    fn x86_mm256_max_epi8(x: i8x32, y: i8x32) -> i8x32;
    fn x86_mm256_max_epu16(x: u16x16, y: u16x16) -> u16x16;
    fn x86_mm256_max_epu32(x: u32x8, y: u32x8) -> u32x8;
    fn x86_mm256_max_epu8(x: u8x32, y: u8x32) -> u8x32;
    fn x86_mm256_min_epi16(x: i16x16, y: i16x16) -> i16x16;
    fn x86_mm256_min_epi32(x: i32x8, y: i32x8) -> i32x8;
    fn x86_mm256_min_epi8(x: i8x32, y: i8x32) -> i8x32;
    fn x86_mm256_min_epu16(x: u16x16, y: u16x16) -> u16x16;
    fn x86_mm256_min_epu32(x: u32x8, y: u32x8) -> u32x8;
    fn x86_mm256_min_epu8(x: u8x32, y: u8x32) -> u8x32;

    fn x86_mm256_movemask_epi8(x: i8x32) -> i32;

    fn x86_mm256_mpsadbw_epu8(x: u8x32, y: u8x32, z: i32) -> u16x16;

    // TODO(mayah): rustc exposes x86_mm256_mul_ep{i,u}64, however, these methods name should be
    // x86_mm256_mul_ep{i,u}32 when we obey intel intrinsic names.
    // Also, we use them, rustc says "undefined reference to `llvm.x86.avx2.pmulq.dq'".
    // fn x86_mm256_mul_epi64(x: i32x8, y: i32x8) -> i64x4;
    // fn x86_mm256_mul_epu64(x: u32x8, y: u32x8) -> u64x4;
    // TODO(mayah): rustc complains "undefined reference to `llvm.x86.avx2.pmulhw.w'".
    // fn x86_mm256_mulhi_epi16(x: i16x16, y: i16x16) -> i16x16;
    // fn x86_mm256_mulhi_epu16(x: u16x16, y: u16x16) -> u16x16;
    fn x86_mm256_mulhrs_epi16(x: i16x16, y: i16x16) -> i16x16;

    fn x86_mm256_packs_epi16(x: i16x16, y: i16x16) -> i8x32;
    fn x86_mm256_packus_epi16(x: i16x16, y: i16x16) -> u8x32;
    fn x86_mm256_packs_epi32(x: i32x8, y: i32x8) -> i16x16;
    fn x86_mm256_packus_epi32(x: i32x8, y: i32x8) -> u16x16;

    // TODO(mayah): rust exposes mm256_sad_epu8 as (u8x32, u8x32) -> u8x32.
    // However, the return type should be u64x4 (or i64x4).
    // fn x86_mm256_sad_epu8(x: u8x32, y: u8x32) -> u8x32;

    fn x86_mm256_permutevar8x32_epi32(x: i32x8, y: i32x8) -> i32x8;
    fn x86_mm256_permutevar8x32_ps(x: m256, y: i32x8) -> m256;

    fn x86_mm256_sign_epi8(x: i8x32, y: i8x32) -> i8x32;
    fn x86_mm256_sign_epi16(x: i16x16, y: i16x16) -> i16x16;
    fn x86_mm256_sign_epi32(x: i32x8, y: i32x8) -> i32x8;

    fn x86_mm256_shuffle_epi8(x: i8x32, y: i8x32) -> i8x32;

    // TODO(mayah): Some of these cause llvm assertion failure.
    // fn x86_mm_mask_i32gather_epi32(src: i32x4, base_addr: *const i32, vindex: i32x4, mask: i32x4, scale: i32) -> i32x4;
    // fn x86_mm_mask_i32gather_epi64(src: i64x2, base_addr: *const i64, vindex: i32x4, mask: i64x2, scale: i32) -> i64x2;
    // fn x86_mm_mask_i32gather_ps(src: m128, base_addr: *const f32, vindex: i32x4, mask: i32x4, scale: i32) -> m128;
    // fn x86_mm_mask_i32gather_pd(src: m128d, base_addr: *const f64, vindex: i32x4, mask: i64x2, scale: i32) -> m128d;
    // fn x86_mm256_mask_i32gather_epi32(src: i32x8, base_addr: *const i32, vindex: i32x8, mask: i32x8, scale: i32) -> i32x8;
    // fn x86_mm256_mask_i32gather_epi64(src: i64x4, base_addr: *const i64, vindex: i32x4, mask: i64x4, scale: i32) -> i64x4;
    // fn x86_mm256_mask_i32gather_ps(src: m256, base_addr: *const f32, vindex: i32x8, mask: i32x8, scale: i32) -> m256;
    // fn x86_mm256_mask_i32gather_pd(src: m256d, base_addr: *const f64, vindex: i32x4, mask: i64x4, scale: i32) -> m256d;
    // fn x86_mm_mask_i64gather_epi32(src: i32x4, base_addr: *const i32, vindex: i64x2, mask: i32x4, scale: i32) -> i32x4;
    // fn x86_mm_mask_i64gather_epi64(src: i64x2, base_addr: *const i64, vindex: i64x2, mask: i64x2, scale: i32) -> i64x2;
    // fn x86_mm_mask_i64gather_ps(src: m128, base_addr: *const f32, vindex: i64x2, mask: i32x4, scale: i32) -> m128;
    // fn x86_mm_mask_i64gather_pd(src: m128d, base_addr: *const f64, vindex: i64x2, mask: i64x2, scale: i32) -> m128d;
    // fn x86_mm256_mask_i64gather_epi32(src: i32x4, base_addr: *const i32, vindex: i64x4, mask: i32x4, scale: i32) -> i32x4;
    // fn x86_mm256_mask_i64gather_epi64(src: i64x4, base_addr: *const i64, vindex: i64x4, mask: i64x4, scale: i32) -> i64x4;
    // fn x86_mm256_mask_i64gather_ps(src: m128, base_addr: *const f32, vindex: i64x4, mask: i32x4, scale: i32) -> m128;
    // fn x86_mm256_mask_i64gather_pd(src: m256d, base_addr: *const f64, vindex: i64x4, mask: i64x4, scale: i32) -> m256d;
}

extern {
    #[link_name = "llvm.x86.avx2.pblendvb"]
    fn avx2_pblendvb(a: i8x32, b: i8x32, c: i8x32) -> i8x32;

    #[link_name = "llvm.x86.avx2.pmadd.wd"]
    fn avx2_pmadd_wd(a: i16x16, b: i16x16) -> i32x8;
    #[link_name = "llvm.x86.avx2.pmadd.ub.sw"]
    fn avx2_pmadd_ub_sw(a: i8x32, b: i8x32) -> i16x16;

    #[link_name = "llvm.x86.avx2.pmul.dq"]
    fn avx2_pmul_dq(a: i32x8, b: i32x8) -> i64x4;
    #[link_name = "llvm.x86.avx2.pmulu.dq"]
    fn avx2_pmulu_dq(a: i32x8, b: i32x8) -> i64x4;
    #[link_name = "llvm.x86.avx2.pmulh.w"]
    fn avx2_pmulh_w(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx2.pmulhu.w"]
    fn avx2_pmulhu_w(a: i16x16, b: i16x16) -> i16x16;

    #[link_name = "llvm.x86.avx2.psad.bw"]
    fn avx2_psad_bw(a: u8x32, b: u8x32) -> u64x4;

    #[link_name = "llvm.x86.avx2.psll.w"]
    fn avx2_psll_w(a: i16x16, b: i16x8) -> i16x16;
    #[link_name = "llvm.x86.avx2.psll.d"]
    fn avx2_psll_d(a: i32x8, b: i32x4) -> i32x8;
    #[link_name = "llvm.x86.avx2.psll.q"]
    fn avx2_psll_q(a: i64x4, b: i64x2) -> i64x4;
    #[link_name = "llvm.x86.avx2.psrl.w"]
    fn avx2_psrl_w(a: i16x16, b: i16x8) -> i16x16;
    #[link_name = "llvm.x86.avx2.psrl.d"]
    fn avx2_psrl_d(a: i32x8, b: i32x4) -> i32x8;
    #[link_name = "llvm.x86.avx2.psrl.q"]
    fn avx2_psrl_q(a: i64x4, b: i64x2) -> i64x4;
    #[link_name = "llvm.x86.avx2.psra.w"]
    fn avx2_psra_w(a: i16x16, b: i16x8) -> i16x16;
    #[link_name = "llvm.x86.avx2.psra.d"]
    fn avx2_psra_d(a: i32x8, b: i32x4) -> i32x8;
    #[link_name = "llvm.x86.avx2.pslli.w"]
    fn avx2_pslli_w(a: i16x16, b: i32) -> i16x16;
    #[link_name = "llvm.x86.avx2.pslli.d"]
    fn avx2_pslli_d(a: i32x8, b: i32) -> i32x8;
    #[link_name = "llvm.x86.avx2.pslli.q"]
    fn avx2_pslli_q(a: i64x4, b: i32) -> i64x4;
    #[link_name = "llvm.x86.avx2.psrli.w"]
    fn avx2_psrli_w(a: i16x16, b: i32) -> i16x16;
    #[link_name = "llvm.x86.avx2.psrli.d"]
    fn avx2_psrli_d(a: i32x8, b: i32) -> i32x8;
    #[link_name = "llvm.x86.avx2.psrli.q"]
    fn avx2_psrli_q(a: i64x4, b: i32) -> i64x4;
    #[link_name = "llvm.x86.avx2.psrai.w"]
    fn avx2_psrai_w(a: i16x16, b: i32) -> i16x16;
    #[link_name = "llvm.x86.avx2.psrai.d"]
    fn avx2_psrai_d(a: i32x8, b: i32) -> i32x8;
    #[link_name = "llvm.x86.avx2.psll.dq"]
    fn avx2_psll_dq(a: i64x4, b: i32) -> i64x4;
    #[link_name = "llvm.x86.avx2.psrl.dq"]
    fn avx2_psrl_dq(a: i64x4, b: i32) -> i64x4;
    #[link_name = "llvm.x86.avx2.psllv.d"]
    fn avx2_psllv_d(a: i32x4, b: i32x4) -> i32x4;
    #[link_name = "llvm.x86.avx2.psllv.d.256"]
    fn avx2_psllv_d_256(a: i32x8, b: i32x8) -> i32x8;
    #[link_name = "llvm.x86.avx2.psllv.q"]
    fn avx2_psllv_q(a: i64x2, b: i64x2) -> i64x2;
    #[link_name = "llvm.x86.avx2.psllv.q.256"]
    fn avx2_psllv_q_256(a: i64x4, b: i64x4) -> i64x4;
    #[link_name = "llvm.x86.avx2.psrlv.d"]
    fn avx2_psrlv_d(a: i32x4, b: i32x4) -> i32x4;
    #[link_name = "llvm.x86.avx2.psrlv.d.256"]
    fn avx2_psrlv_d_256(a: i32x8, b: i32x8) -> i32x8;
    #[link_name = "llvm.x86.avx2.psrlv.q"]
    fn avx2_psrlv_q(a: i64x2, b: i64x2) -> i64x2;
    #[link_name = "llvm.x86.avx2.psrlv.q.256"]
    fn avx2_psrlv_q_256(a: i64x4, b: i64x4) -> i64x4;
    #[link_name = "llvm.x86.avx2.psrav.d"]
    fn avx2_psrav_d(a: i32x4, b: i32x4) -> i32x4;
    #[link_name = "llvm.x86.avx2.psrav.d.256"]
    fn avx2_psrav_d_256(a: i32x8, b: i32x8) -> i32x8;

    #[link_name = "llvm.x86.avx2.vperm2i128"]
    fn avx2_vperm2i128(a: i64x4, b: i64x4, c: u8) -> i64x4;

    #[link_name = "llvm.x86.avx2.gather.d.pd"]
    fn avx2_gather_d_pd(a: m128d, b: *const i8, c: i32x4, d: m128d, e: i8) -> m128d;
    #[link_name = "llvm.x86.avx2.gather.d.pd.256"]
    fn avx2_gather_d_pd_256(a: m256d, b: *const i8, c: i32x4, d: m256d, e: i8) -> m256d;
    #[link_name = "llvm.x86.avx2.gather.q.pd"]
    fn avx2_gather_q_pd(a: m128d, b: *const i8, c: i64x2, d: m128d, e: i8) -> m128d;
    #[link_name = "llvm.x86.avx2.gather.q.pd.256"]
    fn avx2_gather_q_pd_256(a: m256d, b: *const i8, c: i64x4, d: m256d, e: i8) -> m256d;
    #[link_name = "llvm.x86.avx2.gather.d.ps"]
    fn avx2_gather_d_ps(a: m128, b: *const i8, c: i32x4, d: m128, e: i8) -> m128;
    #[link_name = "llvm.x86.avx2.gather.d.ps.256"]
    fn avx2_gather_d_ps_256(a: m256, b: *const i8, c: i32x8, d: m256, e: i8) -> m256;
    #[link_name = "llvm.x86.avx2.gather.q.ps"]
    fn avx2_gather_q_ps(a: m128, b: *const i8, c: i64x2, d: m128, e: i8) -> m128;
    #[link_name = "llvm.x86.avx2.gather.q.ps.256"]
    fn avx2_gather_q_ps_256(a: m128, b: *const i8, c: i64x4, d: m128, e: i8) -> m128;
    #[link_name = "llvm.x86.avx2.gather.d.q"]
    fn avx2_gather_d_q(a: i64x2, b: *const i8, c: i32x4, d: i64x2, e: i8) -> i64x2;
    #[link_name = "llvm.x86.avx2.gather.d.q.256"]
    fn avx2_gather_d_q_256(a: i64x4, b: *const i8, c: i32x4, d: i64x4, e: i8) -> i64x4;
    #[link_name = "llvm.x86.avx2.gather.q.q"]
    fn avx2_gather_q_q(a: i64x2, b: *const i8, c: i64x2, d: i64x2, e: i8) -> i64x2;
    #[link_name = "llvm.x86.avx2.gather.q.q.256"]
    fn avx2_gather_q_q_256(a: i64x4, b: *const i8, c: i64x4, d: i64x4, e: i8) -> i64x4;
    #[link_name = "llvm.x86.avx2.gather.d.d"]
    fn avx2_gather_d_d(a: i32x4, b: *const i8, c: i32x4, d: i32x4, e: i8) -> i32x4;
    #[link_name = "llvm.x86.avx2.gather.d.d.256"]
    fn avx2_gather_d_d_256(a: i32x8, b: *const i8, c: i32x8, d: i32x8, e: i8) -> i32x8;
    #[link_name = "llvm.x86.avx2.gather.q.d"]
    fn avx2_gather_q_d(a: i32x4, b: *const i8, c: i64x2, d: i32x4, e: i8) -> i32x4;
    #[link_name = "llvm.x86.avx2.gather.q.d.256"]
    fn avx2_gather_q_d_256(a: i32x4, b: *const i8, c: i64x4, d: i32x4, e: i8) -> i32x4;
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(simd)]
struct i8x4(i8, i8, i8, i8);

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(simd)]
struct u8x4(u8, u8, u8, u8);

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(simd)]
struct i8x8(i8, i8, i8, i8, i8, i8, i8, i8);

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(simd)]
struct u8x8(u8, u8, u8, u8, u8, u8, u8, u8);

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(simd)]
struct i16x4(i16, i16, i16, i16);

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(simd)]
struct u16x4(u16, u16, u16, u16);

// Add &, |, ^ operators for m256i.

macro_rules! m256i_operators {
    ($name: ident, $method: ident, $func: ident) => {
        impl std::ops::$name for m256i {
            type Output = Self;

            #[inline]
            fn $method(self, x: Self) -> Self {
                unsafe { $func(self, x) }
            }
        }
    }
}
m256i_operators! { BitAnd, bitand, simd_and }
m256i_operators! { BitOr,  bitor,  simd_or  }
m256i_operators! { BitXor, bitxor, simd_xor }

#[inline]
unsafe fn trunc_cast<T, U>(x: T) -> U {
    debug_assert!(std::mem::size_of::<T>() >= std::mem::size_of::<U>());
    std::mem::transmute_copy(&x)
}

// vpabsw
// __m256i _mm256_abs_epi16 (__m256i a)
#[inline]
pub fn mm256_abs_epi16(a: m256i) -> m256i {
    unsafe { x86_mm256_abs_epi16(a.as_i16x16()).as_m256i() }
}

// vpabsd
// __m256i _mm256_abs_epi32 (__m256i a)
#[inline]
pub fn mm256_abs_epi32(a: m256i) -> m256i {
    unsafe { x86_mm256_abs_epi32(a.as_i32x8()).as_m256i() }
}

// vpabsb
// __m256i _mm256_abs_epi8 (__m256i a)
#[inline]
pub fn mm256_abs_epi8(a: m256i) -> m256i {
    unsafe { x86_mm256_abs_epi8(a.as_i8x32()).as_m256i() }
}

// vpaddw
// __m256i _mm256_add_epi16 (__m256i a, __m256i b)
#[inline]
pub fn mm256_add_epi16(a: m256i, b: m256i) -> m256i {
    unsafe { simd_add(a.as_i16x16(), b.as_i16x16()).as_m256i() }
}

// vpaddd
// __m256i _mm256_add_epi32 (__m256i a, __m256i b)
#[inline]
pub fn mm256_add_epi32(a: m256i, b: m256i) -> m256i {
    unsafe { simd_add(a.as_i32x8(), b.as_i32x8()).as_m256i() }
}

// vpaddq
// __m256i _mm256_add_epi64 (__m256i a, __m256i b)
#[inline]
pub fn mm256_add_epi64(a: m256i, b: m256i) -> m256i {
    unsafe { simd_add(a.as_i64x4(), b.as_i64x4()).as_m256i() }
}

// vpaddb
// __m256i _mm256_add_epi8 (__m256i a, __m256i b)
#[inline]
pub fn mm256_add_epi8(a: m256i, b: m256i) -> m256i {
    unsafe { simd_add(a.as_i8x32(), b.as_i8x32()).as_m256i() }
}

// vpaddsw
// __m256i _mm256_adds_epi16 (__m256i a, __m256i b)
#[inline]
pub fn mm256_adds_epi16(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_adds_epi16(a.as_i16x16(), b.as_i16x16()).as_m256i() }
}

// vpaddsb
// __m256i _mm256_adds_epi8 (__m256i a, __m256i b)
#[inline]
pub fn mm256_adds_epi8(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_adds_epi8(a.as_i8x32(), b.as_i8x32()).as_m256i() }
}

// vpaddusw
// __m256i _mm256_adds_epu16 (__m256i a, __m256i b)
#[inline]
pub fn mm256_adds_epu16(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_adds_epu16(a.as_u16x16(), b.as_u16x16()).as_m256i() }
}

// vpaddusb
// __m256i _mm256_adds_epu8 (__m256i a, __m256i b)
#[inline]
pub fn mm256_adds_epu8(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_adds_epu8(a.as_u8x32(), b.as_u8x32()).as_m256i() }
}

// vpalignr
// __m256i _mm256_alignr_epi8 (__m256i a, __m256i b, const int count)
#[inline]
pub fn mm256_alignr_epi8(a: m256i, b: m256i, count: i32) -> m256i {
    let ai = a.as_i8x32();
    let bi = b.as_i8x32();
    let zi = mm256_setzero_si256().as_i8x32();

    unsafe {
        let c: i8x32 = match count {
            0 => simd_shuffle32(ai, bi, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, ]),
            1 => simd_shuffle32(ai, bi, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, ]),
            2 => simd_shuffle32(ai, bi, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, ]),
            3 => simd_shuffle32(ai, bi, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, ]),
            4 => simd_shuffle32(ai, bi, [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, ]),
            5 => simd_shuffle32(ai, bi, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, ]),
            6 => simd_shuffle32(ai, bi, [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, ]),
            7 => simd_shuffle32(ai, bi, [7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, ]),
            8 => simd_shuffle32(ai, bi, [8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, ]),
            9 => simd_shuffle32(ai, bi, [9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, ]),
            10 => simd_shuffle32(ai, bi, [10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, ]),
            11 => simd_shuffle32(ai, bi, [11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, ]),
            12 => simd_shuffle32(ai, bi, [12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, ]),
            13 => simd_shuffle32(ai, bi, [13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, ]),
            14 => simd_shuffle32(ai, bi, [14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, ]),
            15 => simd_shuffle32(ai, bi, [15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, ]),
            16 => simd_shuffle32(bi, zi, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, ]),
            17 => simd_shuffle32(bi, zi, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, ]),
            18 => simd_shuffle32(bi, zi, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, ]),
            19 => simd_shuffle32(bi, zi, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, ]),
            20 => simd_shuffle32(bi, zi, [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, ]),
            21 => simd_shuffle32(bi, zi, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, ]),
            22 => simd_shuffle32(bi, zi, [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, ]),
            23 => simd_shuffle32(bi, zi, [7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, ]),
            24 => simd_shuffle32(bi, zi, [8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, ]),
            25 => simd_shuffle32(bi, zi, [9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, ]),
            26 => simd_shuffle32(bi, zi, [10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, ]),
            27 => simd_shuffle32(bi, zi, [11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, ]),
            28 => simd_shuffle32(bi, zi, [12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, ]),
            29 => simd_shuffle32(bi, zi, [13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, ]),
            30 => simd_shuffle32(bi, zi, [14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, ]),
            31 => simd_shuffle32(bi, zi, [15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, ]),
            _ => zi,
        };
        c.as_m256i()
    }
}

// vpand
// __m256i _mm256_and_si256 (__m256i a, __m256i b)
pub fn mm256_and_si256(a: m256i, b: m256i) -> m256i {
    unsafe { simd_and(a, b) }
}

// vpandn
// __m256i _mm256_andnot_si256 (__m256i a, __m256i b)
pub fn mm256_andnot_si256(a: m256i, b: m256i) -> m256i {
    let ones = i64x4(!0, !0, !0, !0).as_m256i();
    mm256_and_si256(mm256_xor_si256(a, ones), b)
}

// vpavgw
// __m256i _mm256_avg_epu16 (__m256i a, __m256i b)
#[inline]
pub fn mm256_avg_epu16(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_avg_epu16(a.as_u16x16(), b.as_u16x16()).as_m256i() }
}

// vpavgb
// __m256i _mm256_avg_epu8 (__m256i a, __m256i b)
#[inline]
pub fn mm256_avg_epu8(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_avg_epu8(a.as_u8x32(), b.as_u8x32()).as_m256i() }
}

// vpblendw
// __m256i _mm256_blend_epi16 (__m256i a, __m256i b, const int imm8)
#[inline]
pub fn mm256_blend_epi16(a: m256i, b: m256i, imm8: i32) -> m256i {
    let x: i16x16 = blend_shuffle16!(a.as_i16x16(), b.as_i16x16(), imm8);
    x.as_m256i()
}

// vpblendd
// __m128i _mm_blend_epi32 (__m128i a, __m128i b, const int imm8)
#[inline]
pub fn mm_blend_epi32(a: m128i, b: m128i, imm8: i32) -> m128i {
    let x: i32x4 = blend_shuffle4!(a.as_i32x4(), b.as_i32x4(), imm8);
    x.as_m128i()
}

// vpblendd
// __m256i _mm256_blend_epi32 (__m256i a, __m256i b, const int imm8)
#[inline]
pub fn mm256_blend_epi32(a: m256i, b: m256i, imm8: i32) -> m256i {
    let x: i32x8 = blend_shuffle8!(a.as_i32x8(), b.as_i32x8(), imm8);
    x.as_m256i()
}

// vpblendvb
// __m256i _mm256_blendv_epi8 (__m256i a, __m256i b, __m256i mask)
#[inline]
pub fn mm256_blendv_epi8(a: m256i, b: m256i, mask: m256i) -> m256i {
    unsafe { avx2_pblendvb(a.as_i8x32(), b.as_i8x32(), mask.as_i8x32()).as_m256i() }
}

// vpbroadcastb
// __m128i _mm_broadcastb_epi8 (__m128i a)
#[inline]
pub fn mm_broadcastb_epi8(a: m128i) -> m128i {
    let x: i8x16 = unsafe {
        simd_shuffle16(a.as_i8x16(), a.as_i8x16(), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    };
    x.as_m128i()
}

// vpbroadcastb
// __m256i _mm256_broadcastb_epi8 (__m128i a)
#[inline]
pub fn mm256_broadcastb_epi8(a: m128i) -> m256i {
    let x: i8x32 = unsafe {
        simd_shuffle32(a.as_i8x16(), a.as_i8x16(),
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    };
    x.as_m256i()
}

// vpbroadcastd
// __m128i _mm_broadcastd_epi32 (__m128i a)
#[inline]
pub fn mm_broadcastd_epi32(a: m128i) -> m128i {
    let x: i32x4 = unsafe {
        simd_shuffle4(a.as_i32x4(), a.as_i32x4(), [0, 0, 0, 0])
    };
    x.as_m128i()
}

// vpbroadcastd
// __m256i _mm256_broadcastd_epi32 (__m128i a)
#[inline]
pub fn mm256_broadcastd_epi32(a: m128i) -> m256i {
    let x: i32x8 = unsafe {
        simd_shuffle8(a.as_i32x4(), a.as_i32x4(), [0, 0, 0, 0, 0, 0, 0, 0])
    };
    x.as_m256i()
}

// vpbroadcastq
// __m128i _mm_broadcastq_epi64 (__m128i a)
#[inline]
pub fn mm_broadcastq_epi64(a: m128i) -> m128i {
    let x: i64x2 = unsafe {
        simd_shuffle2(a.as_i64x2(), a.as_i64x2(), [0, 0])
    };
    x.as_m128i()
}

// vpbroadcastq
// __m256i _mm256_broadcastq_epi64 (__m128i a)
#[inline]
pub fn mm256_broadcastq_epi64(a: m128i) -> m256i {
    let x: i64x4 = unsafe {
        simd_shuffle4(a.as_i64x2(), a.as_i64x2(), [0, 0, 0, 0])
    };
    x.as_m256i()
}

// movddup
// __m128d _mm_broadcastsd_pd (__m128d a)
#[inline]
pub fn mm_broadcastsd_pd(a: m128d) -> m128d {
    unsafe { simd_shuffle2(a, a, [0, 0]) }
}

// vbroadcastsd
// __m256d _mm256_broadcastsd_pd (__m128d a)
#[inline]
pub fn mm256_broadcastsd_pd(a: m128d) -> m256d {
    unsafe { simd_shuffle4(a, a, [0, 0, 0, 0]) }
}

// vbroadcasti128
// __m256i _mm256_broadcastsi128_si256 (__m128i a)
#[inline]
pub fn mm256_broadcastsi128_si256(a: m128i) -> m256i {
    let x: i64x4 = unsafe {
        simd_shuffle4(a.as_i64x2(), a.as_i64x2(), [0, 1, 0, 1])
    };
    x.as_m256i()
}

// vbroadcastss
// __m128 _mm_broadcastss_ps (__m128 a)
#[inline]
pub fn mm_broadcastss_ps(a: m128) -> m128 {
    unsafe { simd_shuffle4(a, a, [0, 0, 0, 0]) }
}

// vbroadcastss
// __m256 _mm256_broadcastss_ps (__m128 a)
#[inline]
pub fn mm256_broadcastss_ps(a: m128) -> m256 {
    unsafe { simd_shuffle8(a, a, [0, 0, 0, 0, 0, 0, 0, 0]) }
}

// vpbroadcastw
// __m128i _mm_broadcastw_epi16 (__m128i a)
#[inline]
pub fn mm_broadcastw_epi16(a: m128i) -> m128i {
    let x: i16x8 = unsafe {
        simd_shuffle8(a.as_i16x8(), a.as_i16x8(), [0, 0, 0, 0, 0, 0, 0, 0])
    };
    x.as_m128i()
}

// vpbroadcastw
// __m256i _mm256_broadcastw_epi16 (__m128i a)
#[inline]
pub fn mm256_broadcastw_epi16(a: m128i) -> m256i {
    let x: i16x16 = unsafe {
        simd_shuffle16(a.as_i16x8(), a.as_i16x8(), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    };
    x.as_m256i()
}

// vpslldq
// __m256i _mm256_bslli_epi128 (__m256i a, const int imm8)
#[inline]
pub fn mm256_bslli_epi128(a: m256i, imm8: i32) -> m256i {
    mm256_slli_si256(a, imm8)
}

// vpsrldq
// __m256i _mm256_bsrli_epi128 (__m256i a, const int imm8)
#[inline]
pub fn mm256_bsrli_epi128(a: m256i, imm8: i32) -> m256i {
    mm256_srli_si256(a, imm8)
}

// vpcmpeqw
// __m256i _mm256_cmpeq_epi16 (__m256i a, __m256i b)
#[inline]
pub fn mm256_cmpeq_epi16(a: m256i, b: m256i) -> m256i {
    let x: i16x16 = unsafe { simd_eq(a.as_i16x16(), b.as_i16x16()) };
    x.as_m256i()
}

// vpcmpeqd
// __m256i _mm256_cmpeq_epi32 (__m256i a, __m256i b)
#[inline]
pub fn mm256_cmpeq_epi32(a: m256i, b: m256i) -> m256i {
    let x: i32x8 = unsafe { simd_eq(a.as_i32x8(), b.as_i32x8()) };
    x.as_m256i()
}

// vpcmpeqq
// __m256i _mm256_cmpeq_epi64 (__m256i a, __m256i b)
#[inline]
pub fn mm256_cmpeq_epi64(a: m256i, b: m256i) -> m256i {
    let x: i64x4 = unsafe { simd_eq(a.as_i64x4(), b.as_i64x4()) };
    x.as_m256i()
}

// vpcmpeqb
// __m256i _mm256_cmpeq_epi8 (__m256i a, __m256i b)
#[inline]
pub fn mm256_cmpeq_epi8(a: m256i, b: m256i) -> m256i {
    let x: i8x32 = unsafe { simd_eq(a.as_i8x32(), b.as_i8x32()) };
    x.as_m256i()
}

// vpcmpgtw
// __m256i _mm256_cmpgt_epi16 (__m256i a, __m256i b)
#[inline]
pub fn mm256_cmpgt_epi16(a: m256i, b: m256i) -> m256i {
    let x: i16x16 = unsafe { simd_gt(a.as_i16x16(), b.as_i16x16()) };
    x.as_m256i()
}

// vpcmpgtd
// __m256i _mm256_cmpgt_epi32 (__m256i a, __m256i b)
#[inline]
pub fn mm256_cmpgt_epi32(a: m256i, b: m256i) -> m256i {
    let x: i32x8 = unsafe { simd_gt(a.as_i32x8(), b.as_i32x8()) };
    x.as_m256i()
}

// vpcmpgtq
// __m256i _mm256_cmpgt_epi64 (__m256i a, __m256i b)
#[inline]
pub fn mm256_cmpgt_epi64(a: m256i, b: m256i) -> m256i {
    let x: i64x4 = unsafe { simd_gt(a.as_i64x4(), b.as_i64x4()) };
    x.as_m256i()
}

// vpcmpgtb
// __m256i _mm256_cmpgt_epi8 (__m256i a, __m256i b)
#[inline]
pub fn mm256_cmpgt_epi8(a: m256i, b: m256i) -> m256i {
    let x: i8x32 = unsafe { simd_gt(a.as_i8x32(), b.as_i8x32()) };
    x.as_m256i()
}

// vpmovsxwd
// __m256i _mm256_cvtepi16_epi32 (__m128i a)
#[inline]
pub fn mm256_cvtepi16_epi32(a: m128i) -> m256i {
    let x: i32x8 = unsafe { simd_cast(a.as_i16x8()) };
    x.as_m256i()
}

// vpmovsxwq
// __m256i _mm256_cvtepi16_epi64 (__m128i a)
#[inline]
pub fn mm256_cvtepi16_epi64(a: m128i) -> m256i {
    let b: i16x4 = unsafe { trunc_cast(a) };
    let x: i64x4 = unsafe { simd_cast(b) };
    x.as_m256i()
}

// vpmovsxdq
// __m256i _mm256_cvtepi32_epi64 (__m128i a)
#[inline]
pub fn mm256_cvtepi32_epi64(a: m128i) -> m256i {
    let x: i64x4 = unsafe { simd_cast(a.as_i32x4()) };
    x.as_m256i()
}

// vpmovsxbw
// __m256i _mm256_cvtepi8_epi16 (__m128i a)
#[inline]
pub fn mm256_cvtepi8_epi16(a: m128i) -> m256i {
    let x: i16x16 = unsafe { simd_cast(a.as_i8x16()) };
    x.as_m256i()
}

// vpmovsxbd
// __m256i _mm256_cvtepi8_epi32 (__m128i a)
#[inline]
pub fn mm256_cvtepi8_epi32(a: m128i) -> m256i {
    let b: i8x8 = unsafe { trunc_cast(a) };
    let x: i32x8 = unsafe { simd_cast(b) };
    x.as_m256i()
}

// vpmovsxbq
// __m256i _mm256_cvtepi8_epi64 (__m128i a)
#[inline]
pub fn mm256_cvtepi8_epi64(a: m128i) -> m256i {
    let b: i8x4 = unsafe { trunc_cast(a) };
    let x: i64x4 = unsafe { simd_cast(b) };
    x.as_m256i()
}

// vpmovzxwd
// __m256i _mm256_cvtepu16_epi32 (__m128i a)
#[inline]
pub fn mm256_cvtepu16_epi32(a: m128i) -> m256i {
    let x: i32x8 = unsafe { simd_cast(a.as_u16x8()) };
    x.as_m256i()
}

// vpmovzxwq
// __m256i _mm256_cvtepu16_epi64 (__m128i a)
#[inline]
pub fn mm256_cvtepu16_epi64(a: m128i) -> m256i {
    let b: u16x4 = unsafe { trunc_cast(a) };
    let x: i64x4 = unsafe { simd_cast(b) };
    x.as_m256i()
}

// vpmovzxdq
// __m256i _mm256_cvtepu32_epi64 (__m128i a)
#[inline]
pub fn mm256_cvtepu32_epi64(a: m128i) -> m256i {
    let x: i64x4 = unsafe { simd_cast(a.as_u32x4()) };
    x.as_m256i()
}

// vpmovzxbw
// __m256i _mm256_cvtepu8_epi16 (__m128i a)
#[inline]
pub fn mm256_cvtepu8_epi16(a: m128i) -> m256i {
    let x: i16x16 = unsafe { simd_cast(a.as_u8x16()) };
    x.as_m256i()
}

// vpmovzxbd
// __m256i _mm256_cvtepu8_epi32 (__m128i a)
#[inline]
pub fn mm256_cvtepu8_epi32(a: m128i) -> m256i {
    let b: u8x8 = unsafe { trunc_cast(a) };
    let x: i32x8 = unsafe { simd_cast(b) };
    x.as_m256i()
}

// vpmovzxbq
// __m256i _mm256_cvtepu8_epi64 (__m128i a)
#[inline]
pub fn mm256_cvtepu8_epi64(a: m128i) -> m256i {
    let b: u8x4 = unsafe { trunc_cast(a) };
    let x: i64x4 = unsafe { simd_cast(b) };
    x.as_m256i()
}

// vextracti128
// __m128i _mm256_extracti128_si256 (__m256i a, const int imm8)
#[inline]
pub fn mm256_extracti128_si256(a: m256i, imm8: i32) -> m128i {
    unsafe {
        match imm8 & 0x1 {
            0 => simd_shuffle2(a.as_i64x4(), a.as_i64x4(), [0, 1]),
            1 => simd_shuffle2(a.as_i64x4(), a.as_i64x4(), [2, 3]),
            _ => unreachable!()
        }
    }
}

// vphaddw
// __m256i _mm256_hadd_epi16 (__m256i a, __m256i b)
#[inline]
pub fn mm256_hadd_epi16(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_hadd_epi16(a.as_i16x16(), b.as_i16x16()).as_m256i() }
}

// vphaddd
// __m256i _mm256_hadd_epi32 (__m256i a, __m256i b)
#[inline]
pub fn mm256_hadd_epi32(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_hadd_epi32(a.as_i32x8(), b.as_i32x8()).as_m256i() }
}

// vphaddsw
// __m256i _mm256_hadds_epi16 (__m256i a, __m256i b)
#[inline]
pub fn mm256_hadds_epi16(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_hadds_epi16(a.as_i16x16(), b.as_i16x16()).as_m256i() }
}

// vphsubw
// __m256i _mm256_hsub_epi16 (__m256i a, __m256i b)
#[inline]
pub fn mm256_hsub_epi16(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_hsub_epi16(a.as_i16x16(), b.as_i16x16()).as_m256i() }
}

// vphsubd
// __m256i _mm256_hsub_epi32 (__m256i a, __m256i b)
#[inline]
pub fn mm256_hsub_epi32(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_hsub_epi32(a.as_i32x8(), b.as_i32x8()).as_m256i() }
}

// vphsubsw
// __m256i _mm256_hsubs_epi16 (__m256i a, __m256i b)
#[inline]
pub fn mm256_hsubs_epi16(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_hsubs_epi16(a.as_i16x16(), b.as_i16x16()).as_m256i() }
}

// Since scale must be immediate, and its value must be 1, 2, 4 or 8.
macro_rules! gather_impl {
    ($fname: expr, $src: expr, $base_addr: expr, $vindex: expr, $mask: expr, $scale: expr) => {
        match $scale {
            1 => $fname($src, $base_addr, $vindex, $mask, 1),
            2 => $fname($src, $base_addr, $vindex, $mask, 2),
            4 => $fname($src, $base_addr, $vindex, $mask, 4),
            8 => $fname($src, $base_addr, $vindex, $mask, 8),
            _ => {
                assert!(false, "scale should be 1, 2, 4 or 8");
                unreachable!()
            }
        }
    }
}

// vpgatherdd
// __m128i _mm_i32gather_epi32 (int const* base_addr, __m128i vindex, const int scale)
#[inline]
pub unsafe fn mm_i32gather_epi32(base_addr: *const i32, vindex: m128i, scale: i32) -> m128i {
    mm_mask_i32gather_epi32(mm_setzero_si128(), base_addr, vindex, mm_set1_epi32(-1), scale)
}

// vpgatherdd
// __m128i _mm_mask_i32gather_epi32 (__m128i src, int const* base_addr, __m128i vindex, __m128i mask, const int scale)
#[inline]
pub unsafe fn mm_mask_i32gather_epi32(src: m128i, base_addr: *const i32, vindex: m128i, mask: m128i, scale: i32) -> m128i {
    // x86_mm_mask_i32gather_epi32(src.as_i32x4(), base_addr, vindex.as_i32x4(), mask.as_i32x4(), scale).as_m128i()
    gather_impl!(avx2_gather_d_d, src.as_i32x4(), base_addr as *const i8, vindex.as_i32x4(), mask.as_i32x4(), scale).as_m128i()
}

// vpgatherdd
// __m256i _mm256_i32gather_epi32 (int const* base_addr, __m256i vindex, const int scale)
#[inline]
pub unsafe fn mm256_i32gather_epi32(base_addr: *const i32, vindex: m256i, scale: i32) -> m256i {
    mm256_mask_i32gather_epi32(mm256_setzero_si256(), base_addr, vindex, mm256_set1_epi32(-1), scale)
}

// vpgatherdd
// __m256i _mm256_mask_i32gather_epi32 (__m256i src, int const* base_addr, __m256i vindex, __m256i mask, const int scale)
#[inline]
pub unsafe fn mm256_mask_i32gather_epi32(src: m256i, base_addr: *const i32, vindex: m256i, mask: m256i, scale: i32) -> m256i {
    // x86_mm256_mask_i32gather_epi32(src.as_i32x8(), base_addr, vindex.as_i32x8(), mask.as_i32x8(), scale).as_m256i()
    gather_impl!(avx2_gather_d_d_256, src.as_i32x8(), base_addr as *const i8, vindex.as_i32x8(), mask.as_i32x8(), scale).as_m256i()
}

// vpgatherdq
// __m128i _mm_i32gather_epi64 (__int64 const* base_addr, __m128i vindex, const int scale)
#[inline]
pub unsafe fn mm_i32gather_epi64(base_addr: *const i64, vindex: m128i, scale: i32) -> m128i {
    mm_mask_i32gather_epi64(mm_setzero_si128(), base_addr, vindex, mm_set1_epi32(-1), scale)
}

// vpgatherdq
// __m128i _mm_mask_i32gather_epi64 (__m128i src, __int64 const* base_addr, __m128i vindex, __m128i mask, const int scale)
#[inline]
pub unsafe fn mm_mask_i32gather_epi64(src: m128i, base_addr: *const i64, vindex: m128i, mask: m128i, scale: i32) -> m128i {
    // x86_mm_mask_i32gather_epi64(src.as_i64x2(), base_addr, vindex.as_i32x4(), mask.as_i64x2(), scale).as_m128i()
    gather_impl!(avx2_gather_d_q, src.as_i64x2(), base_addr as *const i8, vindex.as_i32x4(), mask.as_i64x2(), scale).as_m128i()
}

// vpgatherdq
// __m256i _mm256_i32gather_epi64 (__int64 const* base_addr, __m128i vindex, const int scale)
#[inline]
pub unsafe fn mm256_i32gather_epi64(base_addr: *const i64, vindex: m128i, scale: i32) -> m256i {
    mm256_mask_i32gather_epi64(mm256_setzero_si256(), base_addr, vindex, mm256_set1_epi64x(-1), scale)
}

// vpgatherdq
// __m256i _mm256_mask_i32gather_epi64 (__m256i src, __int64 const* base_addr, __m128i vindex, __m256i mask, const int scale)
#[inline]
pub unsafe fn mm256_mask_i32gather_epi64(src: m256i, base_addr: *const i64, vindex: m128i, mask: m256i, scale: i32) -> m256i {
    // x86_mm256_mask_i32gather_epi64(src.as_i64x4(), base_addr, vindex.as_i32x4(), mask.as_i64x4(), scale).as_m256i()
    gather_impl!(avx2_gather_d_q_256, src.as_i64x4(), base_addr as *const i8, vindex.as_i32x4(), mask.as_i64x4(), scale).as_m256i()
}

// vgatherdpd
// __m128d _mm_i32gather_pd (double const* base_addr, __m128i vindex, const int scale)
#[inline]
pub unsafe fn mm_i32gather_pd(base_addr: *const f64, vindex: m128i, scale: i32) -> m128d {
    mm_mask_i32gather_pd(mm_setzero_pd(), base_addr, vindex, mm_set1_pd(-1.0), scale)
}

// vgatherdpd
// __m128d _mm_mask_i32gather_pd (__m128d src, double const* base_addr, __m128i vindex, __m128d mask, const int scale)
#[inline]
pub unsafe fn mm_mask_i32gather_pd(src: m128d, base_addr: *const f64, vindex: m128i, mask: m128d, scale: i32) -> m128d {
    // x86_mm_mask_i32gather_pd(src, base_addr, vindex.as_i32x4(), mask.as_m128i().as_i64x2(), scale)
    gather_impl!(avx2_gather_d_pd, src, base_addr as *const i8, vindex.as_i32x4(), mask, scale)
}

// vgatherdpd
// __m256d _mm256_i32gather_pd (double const* base_addr, __m128i vindex, const int scale)
#[inline]
pub unsafe fn mm256_i32gather_pd(base_addr: *const f64, vindex: m128i, scale: i32) -> m256d {
    mm256_mask_i32gather_pd(mm256_setzero_pd(), base_addr, vindex, mm256_set1_pd(-1.0), scale)
}

// vgatherdpd
// __m256d _mm256_mask_i32gather_pd (__m256d src, double const* base_addr, __m128i vindex, __m256d mask, const int scale)
#[inline]
pub unsafe fn mm256_mask_i32gather_pd(src: m256d, base_addr: *const f64, vindex: m128i, mask: m256d, scale: i32) -> m256d {
    // x86_mm256_mask_i32gather_pd(src, base_addr, vindex.as_i32x4(), mask.as_m256i().as_i64x4(), scale)
    gather_impl!(avx2_gather_d_pd_256, src, base_addr as *const i8, vindex.as_i32x4(), mask, scale)
}

// vgatherdps
// __m128 _mm_i32gather_ps (float const* base_addr, __m128i vindex, const int scale)
#[inline]
pub unsafe fn mm_i32gather_ps(base_addr: *const f32, vindex: m128i, scale: i32) -> m128 {
    mm_mask_i32gather_ps(mm_setzero_ps(), base_addr, vindex, mm_set1_ps(-1.0), scale)
}

// vgatherdps
// __m128 _mm_mask_i32gather_ps (__m128 src, float const* base_addr, __m128i vindex, __m128 mask, const int scale)
#[inline]
pub unsafe fn mm_mask_i32gather_ps(src: m128, base_addr: *const f32, vindex: m128i, mask: m128, scale: i32) -> m128 {
    // x86_mm_mask_i32gather_ps(src, base_addr, vindex.as_i32x4(), mask.as_m128i().as_i32x4(), scale)
    gather_impl!(avx2_gather_d_ps, src, base_addr as *const i8, vindex.as_i32x4(), mask, scale)
}

// vgatherdps
// __m256 _mm256_i32gather_ps (float const* base_addr, __m256i vindex, const int scale)
#[inline]
pub unsafe fn mm256_i32gather_ps(base_addr: *const f32, vindex: m256i, scale: i32) -> m256 {
    mm256_mask_i32gather_ps(mm256_setzero_ps(), base_addr, vindex, mm256_set1_ps(-1.0), scale)
}

// vgatherdps
// __m256 _mm256_mask_i32gather_ps (__m256 src, float const* base_addr, __m256i vindex, __m256 mask, const int scale)
#[inline]
pub unsafe fn mm256_mask_i32gather_ps(src: m256, base_addr: *const f32, vindex: m256i, mask: m256, scale: i32) -> m256 {
    // x86_mm256_mask_i32gather_ps(src, base_addr, vindex.as_i32x8(), mask.as_m256i().as_i32x8(), scale)
    gather_impl!(avx2_gather_d_ps_256, src, base_addr as *const i8, vindex.as_i32x8(), mask, scale)
}

// vpgatherqd
// __m128i _mm_i64gather_epi32 (int const* base_addr, __m128i vindex, const int scale)
#[inline]
pub unsafe fn mm_i64gather_epi32(base_addr: *const i32, vindex: m128i, scale: i32) -> m128i {
    mm_mask_i64gather_epi32(mm_setzero_si128(), base_addr, vindex, mm_set1_epi32(-1), scale)
}

// vpgatherqd
// __m128i _mm_mask_i64gather_epi32 (__m128i src, int const* base_addr, __m128i vindex, __m128i mask, const int scale)
#[inline]
pub unsafe fn mm_mask_i64gather_epi32(src: m128i, base_addr: *const i32, vindex: m128i, mask: m128i, scale: i32) -> m128i {
    // x86_mm_mask_i64gather_epi32(src.as_i32x4(), base_addr, vindex.as_i64x2(), mask.as_i32x4(), scale).as_m128i()
    gather_impl!(avx2_gather_q_d, src.as_i32x4(), base_addr as *const i8, vindex.as_i64x2(), mask.as_i32x4(), scale).as_m128i()
}

// vpgatherqd
// __m128i _mm256_i64gather_epi32 (int const* base_addr, __m256i vindex, const int scale)
#[inline]
pub unsafe fn mm256_i64gather_epi32(base_addr: *const i32, vindex: m256i, scale: i32) -> m128i {
    mm256_mask_i64gather_epi32(mm_setzero_si128(), base_addr, vindex, mm_set1_epi32(-1), scale)
}

// vpgatherqd
// __m128i _mm256_mask_i64gather_epi32 (__m128i src, int const* base_addr, __m256i vindex, __m128i mask, const int scale)
#[inline]
pub unsafe fn mm256_mask_i64gather_epi32(src: m128i, base_addr: *const i32, vindex: m256i, mask: m128i, scale: i32) -> m128i {
    // x86_mm256_mask_i64gather_epi32(src.as_i32x4(), base_addr, vindex.as_i64x4(), mask.as_i32x4(), scale).as_m128i()
    gather_impl!(avx2_gather_q_d_256, src.as_i32x4(), base_addr as *const i8, vindex.as_i64x4(), mask.as_i32x4(), scale).as_m128i()
}

// vpgatherqq
// __m128i _mm_i64gather_epi64 (__int64 const* base_addr, __m128i vindex, const int scale)
#[inline]
pub unsafe fn mm_i64gather_epi64(base_addr: *const i64, vindex: m128i, scale: i32) -> m128i {
    mm_mask_i64gather_epi64(mm_setzero_si128(), base_addr, vindex, mm_set1_epi64x(-1), scale)
}

// vpgatherqq
// __m128i _mm_mask_i64gather_epi64 (__m128i src, __int64 const* base_addr, __m128i vindex, __m128i mask, const int scale)
#[inline]
pub unsafe fn mm_mask_i64gather_epi64(src: m128i, base_addr: *const i64, vindex: m128i, mask: m128i, scale: i32) -> m128i {
    // x86_mm_mask_i64gather_epi64(src.as_i64x2(), base_addr, vindex.as_i64x2(), mask.as_i64x2(), scale).as_m128i()
    gather_impl!(avx2_gather_q_q, src.as_i64x2(), base_addr as *const i8, vindex.as_i64x2(), mask.as_i64x2(), scale).as_m128i()
}

// vpgatherqq
// __m256i _mm256_i64gather_epi64 (__int64 const* base_addr, __m256i vindex, const int scale)
#[inline]
pub unsafe fn mm256_i64gather_epi64(base_addr: *const i64, vindex: m256i, scale: i32) -> m256i {
    mm256_mask_i64gather_epi64(mm256_setzero_si256(), base_addr, vindex, mm256_set1_epi64x(-1), scale)
}

// vpgatherqq
// __m256i _mm256_mask_i64gather_epi64 (__m256i src, __int64 const* base_addr, __m256i vindex, __m256i mask, const int scale)
#[inline]
pub unsafe fn mm256_mask_i64gather_epi64(src: m256i, base_addr: *const i64, vindex: m256i, mask: m256i, scale: i32) -> m256i {
    // x86_mm256_mask_i64gather_epi64(src.as_i64x4(), base_addr, vindex.as_i64x4(), mask.as_i64x4(), scale).as_m256i()
    gather_impl!(avx2_gather_q_q_256, src.as_i64x4(), base_addr as *const i8, vindex.as_i64x4(), mask.as_i64x4(), scale).as_m256i()
}

// vgatherqpd
// __m128d _mm_i64gather_pd (double const* base_addr, __m128i vindex, const int scale)
#[inline]
pub unsafe fn mm_i64gather_pd(base_addr: *const f64, vindex: m128i, scale: i32) -> m128d {
    mm_mask_i64gather_pd(mm_setzero_pd(), base_addr, vindex, mm_set1_pd(-1.0), scale)
}

// vgatherqpd
// __m128d _mm_mask_i64gather_pd (__m128d src, double const* base_addr, __m128i vindex, __m128d mask, const int scale)
#[inline]
pub unsafe fn mm_mask_i64gather_pd(src: m128d, base_addr: *const f64, vindex: m128i, mask: m128d, scale: i32) -> m128d {
    // x86_mm_mask_i64gather_pd(src, base_addr, vindex.as_i64x2(), mask.as_m128i().as_i64x2(), scale)
    gather_impl!(avx2_gather_q_pd, src, base_addr as *const i8, vindex.as_i64x2(), mask, scale)
}

// vgatherqpd
// __m256d _mm256_i64gather_pd (double const* base_addr, __m256i vindex, const int scale)
#[inline]
pub unsafe fn mm256_i64gather_pd(base_addr: *const f64, vindex: m256i, scale: i32) -> m256d {
    mm256_mask_i64gather_pd(mm256_setzero_pd(), base_addr, vindex, mm256_set1_pd(-1.0), scale)
}

// vgatherqpd
// __m256d _mm256_mask_i64gather_pd (__m256d src, double const* base_addr, __m256i vindex, __m256d mask, const int scale)
#[inline]
pub unsafe fn mm256_mask_i64gather_pd(src: m256d, base_addr: *const f64, vindex: m256i, mask: m256d, scale: i32) -> m256d {
    // x86_mm256_mask_i64gather_pd(src, base_addr, vindex.as_i64x4(), mask.as_m256i().as_i64x4(), scale)
    gather_impl!(avx2_gather_q_pd_256, src, base_addr as *const i8, vindex.as_i64x4(), mask, scale)
}

// vgatherqps
// __m128 _mm_i64gather_ps (float const* base_addr, __m128i vindex, const int scale)
#[inline]
pub unsafe fn mm_i64gather_ps(base_addr: *const f32, vindex: m128i, scale: i32) -> m128 {
    mm_mask_i64gather_ps(mm_setzero_ps(), base_addr, vindex, mm_set1_ps(-1.0), scale)
}

// vgatherqps
// __m128 _mm_mask_i64gather_ps (__m128 src, float const* base_addr, __m128i vindex, __m128 mask, const int scale)
#[inline]
pub unsafe fn mm_mask_i64gather_ps(src: m128, base_addr: *const f32, vindex: m128i, mask: m128, scale: i32) -> m128 {
    // x86_mm_mask_i64gather_ps(src, base_addr, vindex.as_i64x2(), mask.as_m128i().as_i32x4(), scale)
    gather_impl!(avx2_gather_q_ps, src, base_addr as *const i8, vindex.as_i64x2(), mask, scale)
}

// vgatherqps
// __m128 _mm256_i64gather_ps (float const* base_addr, __m256i vindex, const int scale)
#[inline]
pub unsafe fn mm256_i64gather_ps(base_addr: *const f32, vindex: m256i, scale: i32) -> m128 {
    mm256_mask_i64gather_ps(mm_setzero_ps(), base_addr, vindex, mm_set1_ps(-1.0), scale)
}

// vgatherqps
// __m128 _mm256_mask_i64gather_ps (__m128 src, float const* base_addr, __m256i vindex, __m128 mask, const int scale)
#[inline]
pub unsafe fn mm256_mask_i64gather_ps(src: m128, base_addr: *const f32, vindex: m256i, mask: m128, scale: i32) -> m128 {
    // x86_mm256_mask_i64gather_ps(src, base_addr, vindex.as_i64x4(), mask.as_m128i().as_i32x4(), scale)
    gather_impl!(avx2_gather_q_ps_256, src, base_addr as *const i8, vindex.as_i64x4(), mask, scale)
}

// vinserti128
// __m256i _mm256_inserti128_si256 (__m256i a, __m128i b, const int imm8)
#[inline]
pub fn mm256_inserti128_si256(a: m256i, b: m128i, imm8: i32) -> m256i {
    unsafe {
        let b256 = mm256_castsi128_si256(b);
        let x: i64x4 = match imm8 & 0x1 {
            0 => simd_shuffle4(a.as_i64x4(), b256.as_i64x4(), [4, 5, 2, 3]),
            1 => simd_shuffle4(a.as_i64x4(), b256.as_i64x4(), [0, 1, 4, 5]),
            _ => unreachable!()
        };
        x.as_m256i()
    }
}

// vpmaddwd
// __m256i _mm256_madd_epi16 (__m256i a, __m256i b)
#[inline]
pub fn mm256_madd_epi16(a: m256i, b: m256i) -> m256i {
    unsafe { avx2_pmadd_wd(a.as_i16x16(), b.as_i16x16()).as_m256i() }
}

// vpmaddubsw
// __m256i _mm256_maddubs_epi16 (__m256i a, __m256i b)
#[inline]
pub fn mm256_maddubs_epi16(a: m256i, b: m256i) -> m256i {
    unsafe { avx2_pmadd_ub_sw(a.as_i8x32(), b.as_i8x32()).as_m256i() }
}

// vpmaskmovd
// __m128i _mm_maskload_epi32 (int const* mem_addr, __m128i mask)
#[inline]
pub unsafe fn mm_maskload_epi32(mem_addr: *const i32, mask: m128i) -> m128i {
    x86_mm_maskload_epi32(mem_addr as *const i32x4, mask.as_i32x4()).as_m128i()
}

// vpmaskmovd
// __m256i _mm256_maskload_epi32 (int const* mem_addr, __m256i mask)
#[inline]
pub unsafe fn mm256_maskload_epi32(mem_addr: *const i32, mask: m256i) -> m256i {
    x86_mm256_maskload_epi32(mem_addr as *const i32x8, mask.as_i32x8()).as_m256i()
}

// vpmaskmovq
// __m128i _mm_maskload_epi64 (__int64 const* mem_addr, __m128i mask)
#[inline]
pub unsafe fn mm_maskload_epi64(mem_addr: *const i64, mask: m128i) -> m128i {
    x86_mm_maskload_epi64(mem_addr as *const i64x2, mask.as_i64x2()).as_m128i()
}

// vpmaskmovq
// __m256i _mm256_maskload_epi64 (__int64 const* mem_addr, __m256i mask)
#[inline]
pub unsafe fn mm256_maskload_epi64(mem_addr: *const i64, mask: m256i) -> m256i {
    x86_mm256_maskload_epi64(mem_addr as *const i64x4, mask.as_i64x4()).as_m256i()
}

// vpmaskmovd
// void _mm_maskstore_epi32 (int* mem_addr, __m128i mask, __m128i a)
#[inline]
pub unsafe fn mm_maskstore_epi32(mem_addr: *mut i32, mask: m128i, a: m128i) {
    x86_mm_maskstore_epi32(mem_addr, mask.as_i32x4(), a.as_i32x4())
}

// vpmaskmovd
// void _mm256_maskstore_epi32 (int* mem_addr, __m256i mask, __m256i a)
#[inline]
pub unsafe fn mm256_maskstore_epi32(mem_addr: *mut i32, mask: m256i, a: m256i) {
    x86_mm256_maskstore_epi32(mem_addr, mask.as_i32x8(), a.as_i32x8())
}

// vpmaskmovq
// void _mm_maskstore_epi64 (__int64* mem_addr, __m128i mask, __m128i a)
#[inline]
pub unsafe fn mm_maskstore_epi64(mem_addr: *mut i64, mask: m128i, a: m128i) {
    x86_mm_maskstore_epi64(mem_addr, mask.as_i64x2(), a.as_i64x2())
}

// vpmaskmovq
// void _mm256_maskstore_epi64 (__int64* mem_addr, __m256i mask, __m256i a)
#[inline]
pub unsafe fn mm256_maskstore_epi64(mem_addr: *mut i64, mask: m256i, a: m256i) {
    x86_mm256_maskstore_epi64(mem_addr, mask.as_i64x4(), a.as_i64x4())
}

// vpmaxsw
// __m256i _mm256_max_epi16 (__m256i a, __m256i b)
#[inline]
pub fn mm256_max_epi16(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_max_epi16(a.as_i16x16(), b.as_i16x16()).as_m256i() }
}

// vpmaxsd
// __m256i _mm256_max_epi32 (__m256i a, __m256i b)
#[inline]
pub fn mm256_max_epi32(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_max_epi32(a.as_i32x8(), b.as_i32x8()).as_m256i() }
}

// vpmaxsb
// __m256i _mm256_max_epi8 (__m256i a, __m256i b)
#[inline]
pub fn mm256_max_epi8(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_max_epi8(a.as_i8x32(), b.as_i8x32()).as_m256i() }
}

// vpmaxuw
// __m256i _mm256_max_epu16 (__m256i a, __m256i b)
#[inline]
pub fn mm256_max_epu16(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_max_epu16(a.as_u16x16(), b.as_u16x16()).as_m256i() }
}

// vpmaxud
// __m256i _mm256_max_epu32 (__m256i a, __m256i b)
#[inline]
pub fn mm256_max_epu32(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_max_epu32(a.as_u32x8(), b.as_u32x8()).as_m256i() }
}

// vpmaxub
// __m256i _mm256_max_epu8 (__m256i a, __m256i b)
#[inline]
pub fn mm256_max_epu8(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_max_epu8(a.as_u8x32(), b.as_u8x32()).as_m256i() }
}

// vpminsw
// __m256i _mm256_min_epi16 (__m256i a, __m256i b)
#[inline]
pub fn mm256_min_epi16(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_min_epi16(a.as_i16x16(), b.as_i16x16()).as_m256i() }
}

// vpminsd
// __m256i _mm256_min_epi32 (__m256i a, __m256i b)
#[inline]
pub fn mm256_min_epi32(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_min_epi32(a.as_i32x8(), b.as_i32x8()).as_m256i() }
}

// vpminsb
// __m256i _mm256_min_epi8 (__m256i a, __m256i b)
#[inline]
pub fn mm256_min_epi8(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_min_epi8(a.as_i8x32(), b.as_i8x32()).as_m256i() }
}

// vpminuw
// __m256i _mm256_min_epu16 (__m256i a, __m256i b)
#[inline]
pub fn mm256_min_epu16(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_min_epu16(a.as_u16x16(), b.as_u16x16()).as_m256i() }
}

// vpminud
// __m256i _mm256_min_epu32 (__m256i a, __m256i b)
#[inline]
pub fn mm256_min_epu32(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_min_epu32(a.as_u32x8(), b.as_u32x8()).as_m256i() }
}

// vpminub
// __m256i _mm256_min_epu8 (__m256i a, __m256i b)
#[inline]
pub fn mm256_min_epu8(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_min_epu8(a.as_u8x32(), b.as_u8x32()).as_m256i() }
}

// vpmovmskb
// int _mm256_movemask_epi8 (__m256i a)
#[inline]
pub fn mm256_movemask_epi8(a: m256i) -> i32 {
    unsafe { x86_mm256_movemask_epi8(a.as_i8x32()) }
}

// vmpsadbw
// __m256i _mm256_mpsadbw_epu8 (__m256i a, __m256i b, const int imm8)
#[inline]
pub fn mm256_mpsadbw_epu8(a: m256i, b: m256i, imm8: i32) -> m256i {
    fn_imm8_arg2!(x86_mm256_mpsadbw_epu8, a.as_u8x32(), b.as_u8x32(), imm8).as_m256i()
}

// vpmuldq
// __m256i _mm256_mul_epi32 (__m256i a, __m256i b)
#[inline]
pub fn mm256_mul_epi32(a: m256i, b: m256i) -> m256i {
    unsafe { avx2_pmul_dq(a.as_i32x8(), b.as_i32x8()).as_m256i() }
}

// vpmuludq
// __m256i _mm256_mul_epu32 (__m256i a, __m256i b)
#[inline]
pub fn mm256_mul_epu32(a: m256i, b: m256i) -> m256i {
    unsafe { avx2_pmulu_dq(a.as_i32x8(), b.as_i32x8()).as_m256i() }
}

// vpmulhw
// __m256i _mm256_mulhi_epi16 (__m256i a, __m256i b)
#[inline]
pub fn mm256_mulhi_epi16(a: m256i, b: m256i) -> m256i {
    unsafe { avx2_pmulh_w(a.as_i16x16(), b.as_i16x16()).as_m256i() }
}

// vpmulhuw
// __m256i _mm256_mulhi_epu16 (__m256i a, __m256i b)
#[inline]
pub fn mm256_mulhi_epu16(a: m256i, b: m256i) -> m256i {
    unsafe { avx2_pmulhu_w(a.as_i16x16(), b.as_i16x16()).as_m256i() }
}

// vpmulhrsw
// __m256i _mm256_mulhrs_epi16 (__m256i a, __m256i b)
#[inline]
pub fn mm256_mulhrs_epi16(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_mulhrs_epi16(a.as_i16x16(), b.as_i16x16()).as_m256i() }
}

// vpmullw
// __m256i _mm256_mullo_epi16 (__m256i a, __m256i b)
#[inline]
pub fn mm256_mullo_epi16(a: m256i, b: m256i) -> m256i {
    unsafe { simd_mul(a.as_i16x16(), b.as_i16x16()).as_m256i() }
}

// vpmulld
// __m256i _mm256_mullo_epi32 (__m256i a, __m256i b)
#[inline]
pub fn mm256_mullo_epi32(a: m256i, b: m256i) -> m256i {
    unsafe { simd_mul(a.as_i32x8(), b.as_i32x8()).as_m256i() }
}

// vpor
// __m256i _mm256_or_si256 (__m256i a, __m256i b)
pub fn mm256_or_si256(a: m256i, b: m256i) -> m256i {
    unsafe { simd_or(a, b) }
}

// vpacksswb
// __m256i _mm256_packs_epi16 (__m256i a, __m256i b)
pub fn mm256_packs_epi16(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_packs_epi16(a.as_i16x16(), b.as_i16x16()).as_m256i() }
}

// vpackssdw
// __m256i _mm256_packs_epi32 (__m256i a, __m256i b)
pub fn mm256_packs_epi32(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_packs_epi32(a.as_i32x8(), b.as_i32x8()).as_m256i() }
}

// vpackuswb
// __m256i _mm256_packus_epi16 (__m256i a, __m256i b)
pub fn mm256_packus_epi16(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_packus_epi16(a.as_i16x16(), b.as_i16x16()).as_m256i() }
}

// vpackusdw
// __m256i _mm256_packus_epi32 (__m256i a, __m256i b)
pub fn mm256_packus_epi32(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_packus_epi32(a.as_i32x8(), b.as_i32x8()).as_m256i() }
}

// vperm2i128
// __m256i _mm256_permute2x128_si256 (__m256i a, __m256i b, const int imm8)
#[inline]
pub fn mm256_permute2x128_si256(a: m256i, b: m256i, imm8: i32) -> m256i {
    fn_imm8_arg2!(avx2_vperm2i128, a.as_i64x4(), b.as_i64x4(), imm8).as_m256i()
}

// vpermq
// __m256i _mm256_permute4x64_epi64 (__m256i a, const int imm8)
#[inline]
pub fn mm256_permute4x64_epi64(a: m256i, imm8: i32) -> m256i {
    let x: i64x4 = permute_shuffle4!(a.as_i64x4(), mm256_setzero_si256().as_i64x4(), imm8);
    x.as_m256i()
}

// vpermpd
// __m256d _mm256_permute4x64_pd (__m256d a, const int imm8)
#[inline]
pub fn mm256_permute4x64_pd(a: m256d, imm8: i32) -> m256d {
    permute_shuffle4!(a, mm256_setzero_pd(), imm8)
}

// vpermd
// __m256i _mm256_permutevar8x32_epi32 (__m256i a, __m256i idx)
#[inline]
pub fn mm256_permutevar8x32_epi32(a: m256i, idx: m256i) -> m256i {
    unsafe { x86_mm256_permutevar8x32_epi32(a.as_i32x8(), idx.as_i32x8()).as_m256i() }
}

// vpermps
// __m256 _mm256_permutevar8x32_ps (__m256 a, __m256i idx)
#[inline]
pub fn mm256_permutevar8x32_ps(a: m256, idx: m256i) -> m256 {
    unsafe { x86_mm256_permutevar8x32_ps(a, idx.as_i32x8()) }
}

// vpsadbw
// __m256i _mm256_sad_epu8 (__m256i a, __m256i b)
#[inline]
pub fn mm256_sad_epu8(a: m256i, b: m256i) -> m256i {
    unsafe { avx2_psad_bw(a.as_u8x32(), b.as_u8x32()).as_m256i() }
}

// vpshufd
// __m256i _mm256_shuffle_epi32 (__m256i a, const int imm8)
#[inline]
pub fn mm256_shuffle_epi32(a: m256i, imm8: i32) -> m256i {
    macro_rules! shuffle4 {
        ($v0: expr, $v4: expr, $v1: expr, $v5: expr, $v2: expr, $v6: expr, $v3: expr, $v7: expr) => {
            unsafe {
                let x: i32x8 = simd_shuffle8(a.as_i32x8(), a.as_i32x8(), [$v0, $v1, $v2, $v3, $v4, $v5, $v6, $v7]);
                x.as_m256i()
            }
        }
    }
    macro_rules! shuffle3 {
        ($v0: expr, $v4: expr, $v1: expr, $v5: expr, $v2: expr, $v6: expr) => {
            match (imm8 >> 6) & 3 {
                0 => shuffle4!($v0, $v4, $v1, $v5, $v2, $v6, 0, 4),
                1 => shuffle4!($v0, $v4, $v1, $v5, $v2, $v6, 1, 5),
                2 => shuffle4!($v0, $v4, $v1, $v5, $v2, $v6, 2, 6),
                3 => shuffle4!($v0, $v4, $v1, $v5, $v2, $v6, 3, 7),
                _ => unreachable!()
            }
        }
    }
    macro_rules! shuffle2 {
        ($v0: expr, $v4: expr, $v1:expr, $v5: expr) => {
            match (imm8 >> 4) & 3 {
                0 => shuffle3!($v0, $v4, $v1, $v5, 0, 4),
                1 => shuffle3!($v0, $v4, $v1, $v5, 1, 5),
                2 => shuffle3!($v0, $v4, $v1, $v5, 2, 6),
                3 => shuffle3!($v0, $v4, $v1, $v5, 3, 7),
                _ => unreachable!()
            }
        }
    }
    macro_rules! shuffle1 {
        ($v0: expr, $v4: expr) => {
            match (imm8 >> 2) & 0x3 {
                0 => shuffle2!($v0, $v4, 0, 4),
                1 => shuffle2!($v0, $v4, 1, 5),
                2 => shuffle2!($v0, $v4, 2, 6),
                3 => shuffle2!($v0, $v4, 3, 7),
                _ => unreachable!()
            }
        }
    }
    macro_rules! shuffle0 {
        () => {
            match (imm8 >> 0) & 0x3 {
                0 => shuffle1!(0, 4),
                1 => shuffle1!(1, 5),
                2 => shuffle1!(2, 6),
                3 => shuffle1!(3, 7),
                _ => unreachable!()
            }
        }
    }

    shuffle0!()
}

// vpshufb
// __m256i _mm256_shuffle_epi8 (__m256i a, __m256i b)
#[inline]
pub fn mm256_shuffle_epi8(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_shuffle_epi8(a.as_i8x32(), b.as_i8x32()).as_m256i() }
}

// vpshufhw
// __m256i _mm256_shufflehi_epi16 (__m256i a, const int imm8)
#[inline]
pub fn mm256_shufflehi_epi16(a: m256i, imm8: i32) -> m256i {
    macro_rules! shuffle4 {
        ($v0: expr, $v4: expr, $v1: expr, $v5: expr, $v2: expr, $v6: expr, $v3: expr, $v7: expr) => {
            unsafe {
                let x: i16x16 = simd_shuffle16(a.as_i16x16(), a.as_i16x16(),
                                               [0, 1, 2, 3, $v0, $v1, $v2, $v3, 8, 9, 10, 11, $v4, $v5, $v6, $v7]);

                x.as_m256i()
            }
        }
    }
    macro_rules! shuffle3 {
        ($v0: expr, $v4: expr, $v1: expr, $v5: expr, $v2: expr, $v6: expr) => {
            match (imm8 >> 6) & 3 {
                0 => shuffle4!($v0, $v4, $v1, $v5, $v2, $v6, 4, 12),
                1 => shuffle4!($v0, $v4, $v1, $v5, $v2, $v6, 5, 13),
                2 => shuffle4!($v0, $v4, $v1, $v5, $v2, $v6, 6, 14),
                3 => shuffle4!($v0, $v4, $v1, $v5, $v2, $v6, 7, 15),
                _ => unreachable!()
            }
        }
    }
    macro_rules! shuffle2 {
        ($v0: expr, $v4: expr, $v1:expr, $v5: expr) => {
            match (imm8 >> 4) & 3 {
                0 => shuffle3!($v0, $v4, $v1, $v5, 4, 12),
                1 => shuffle3!($v0, $v4, $v1, $v5, 5, 13),
                2 => shuffle3!($v0, $v4, $v1, $v5, 6, 14),
                3 => shuffle3!($v0, $v4, $v1, $v5, 7, 15),
                _ => unreachable!()
            }
        }
    }
    macro_rules! shuffle1 {
        ($v0: expr, $v4: expr) => {
            match (imm8 >> 2) & 0x3 {
                0 => shuffle2!($v0, $v4, 4, 12),
                1 => shuffle2!($v0, $v4, 5, 13),
                2 => shuffle2!($v0, $v4, 6, 14),
                3 => shuffle2!($v0, $v4, 7, 15),
                _ => unreachable!()
            }
        }
    }
    macro_rules! shuffle0 {
        () => {
            match (imm8 >> 0) & 0x3 {
                0 => shuffle1!(4, 12),
                1 => shuffle1!(5, 13),
                2 => shuffle1!(6, 14),
                3 => shuffle1!(7, 15),
                _ => unreachable!()
            }
        }
    }

    shuffle0!()
}

// vpshuflw
// __m256i _mm256_shufflelo_epi16 (__m256i a, const int imm8)
#[inline]
pub fn mm256_shufflelo_epi16(a: m256i, imm8: i32) -> m256i {
    macro_rules! shuffle4 {
        ($v0: expr, $v4: expr, $v1: expr, $v5: expr, $v2: expr, $v6: expr, $v3: expr, $v7: expr) => {
            unsafe {
                let x: i16x16 = simd_shuffle16(a.as_i16x16(), a.as_i16x16(),
                                              [$v0, $v1, $v2, $v3, 4, 5, 6, 7, $v4, $v5, $v6, $v7, 12, 13, 14, 15]);

                x.as_m256i()
            }
        }
    }
    macro_rules! shuffle3 {
        ($v0: expr, $v4: expr, $v1: expr, $v5: expr, $v2: expr, $v6: expr) => {
            match (imm8 >> 6) & 3 {
                0 => shuffle4!($v0, $v4, $v1, $v5, $v2, $v6, 0, 8),
                1 => shuffle4!($v0, $v4, $v1, $v5, $v2, $v6, 1, 9),
                2 => shuffle4!($v0, $v4, $v1, $v5, $v2, $v6, 2, 10),
                3 => shuffle4!($v0, $v4, $v1, $v5, $v2, $v6, 3, 11),
                _ => unreachable!()
            }
        }
    }
    macro_rules! shuffle2 {
        ($v0: expr, $v4: expr, $v1:expr, $v5: expr) => {
            match (imm8 >> 4) & 3 {
                0 => shuffle3!($v0, $v4, $v1, $v5, 0, 8),
                1 => shuffle3!($v0, $v4, $v1, $v5, 1, 9),
                2 => shuffle3!($v0, $v4, $v1, $v5, 2, 10),
                3 => shuffle3!($v0, $v4, $v1, $v5, 3, 11),
                _ => unreachable!()
            }
        }
    }
    macro_rules! shuffle1 {
        ($v0: expr, $v4: expr) => {
            match (imm8 >> 2) & 0x3 {
                0 => shuffle2!($v0, $v4, 0, 8),
                1 => shuffle2!($v0, $v4, 1, 9),
                2 => shuffle2!($v0, $v4, 2, 10),
                3 => shuffle2!($v0, $v4, 3, 11),
                _ => unreachable!()
            }
        }
    }
    macro_rules! shuffle0 {
        () => {
            match (imm8 >> 0) & 0x3 {
                0 => shuffle1!(0, 8),
                1 => shuffle1!(1, 9),
                2 => shuffle1!(2, 10),
                3 => shuffle1!(3, 11),
                _ => unreachable!()
            }
        }
    }

    shuffle0!()
}

// vpsignw
// __m256i _mm256_sign_epi16 (__m256i a, __m256i b)
#[inline]
pub fn mm256_sign_epi16(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_sign_epi16(a.as_i16x16(), b.as_i16x16()).as_m256i() }
}

// vpsignd
// __m256i _mm256_sign_epi32 (__m256i a, __m256i b)
#[inline]
pub fn mm256_sign_epi32(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_sign_epi32(a.as_i32x8(), b.as_i32x8()).as_m256i() }
}

// vpsignb
// __m256i _mm256_sign_epi8 (__m256i a, __m256i b)
#[inline]
pub fn mm256_sign_epi8(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_sign_epi8(a.as_i8x32(), b.as_i8x32()).as_m256i() }
}

// vpsllw
// __m256i _mm256_sll_epi16 (__m256i a, __m128i count)
#[inline]
pub fn mm256_sll_epi16(a: m256i, count: m128i) -> m256i {
    unsafe { avx2_psll_w(a.as_i16x16(), count.as_i16x8()).as_m256i() }
}

// vpslld
// __m256i _mm256_sll_epi32 (__m256i a, __m128i count)
#[inline]
pub fn mm256_sll_epi32(a: m256i, count: m128i) -> m256i {
    unsafe { avx2_psll_d(a.as_i32x8(), count.as_i32x4()).as_m256i() }
}

// vpsllq
// __m256i _mm256_sll_epi64 (__m256i a, __m128i count)
#[inline]
pub fn mm256_sll_epi64(a: m256i, count: m128i) -> m256i {
    unsafe { avx2_psll_q(a.as_i64x4(), count.as_i64x2()).as_m256i() }
}

// vpsllw
// __m256i _mm256_slli_epi16 (__m256i a, int imm8)
#[inline]
pub fn mm256_slli_epi16(a: m256i, imm8: i32) -> m256i {
    fn_imm8_arg1!(avx2_pslli_w, a.as_i16x16(), imm8).as_m256i()
}

// vpslld
// __m256i _mm256_slli_epi32 (__m256i a, int imm8)
#[inline]
pub fn mm256_slli_epi32(a: m256i, imm8: i32) -> m256i {
    fn_imm8_arg1!(avx2_pslli_d, a.as_i32x8(), imm8).as_m256i()
}

// vpsllq
// __m256i _mm256_slli_epi64 (__m256i a, int imm8)
#[inline]
pub fn mm256_slli_epi64(a: m256i, imm8: i32) -> m256i {
    fn_imm8_arg1!(avx2_pslli_q, a.as_i64x4(), imm8).as_m256i()
}

// vpslldq
// __m256i _mm256_slli_si256 (__m256i a, const int imm8)
#[inline]
pub fn mm256_slli_si256(a: m256i, imm8: i32) -> m256i {
    fn_imm8_arg1!(avx2_psll_dq, a.as_i64x4(), imm8 * 8).as_m256i()
}

// vpsllvd
// __m128i _mm_sllv_epi32 (__m128i a, __m128i count)
#[inline]
pub fn mm_sllv_epi32(a: m128i, count: m128i) -> m128i {
    unsafe { avx2_psllv_d(a.as_i32x4(), count.as_i32x4()).as_m128i() }
}

// vpsllvd
// __m256i _mm256_sllv_epi32 (__m256i a, __m256i count)
#[inline]
pub fn mm256_sllv_epi32(a: m256i, count: m256i) -> m256i {
    unsafe { avx2_psllv_d_256(a.as_i32x8(), count.as_i32x8()).as_m256i() }
}

// vpsllvq
// __m128i _mm_sllv_epi64 (__m128i a, __m128i count)
#[inline]
pub fn mm_sllv_epi64(a: m128i, count: m128i) -> m128i {
    unsafe { avx2_psllv_q(a.as_i64x2(), count.as_i64x2()).as_m128i() }
}

// vpsllvq
// __m256i _mm256_sllv_epi64 (__m256i a, __m256i count)
#[inline]
pub fn mm256_sllv_epi64(a: m256i, count: m256i) -> m256i {
    unsafe { avx2_psllv_q_256(a.as_i64x4(), count.as_i64x4()).as_m256i() }
}

// vpsraw
// __m256i _mm256_sra_epi16 (__m256i a, __m128i count)
#[inline]
pub fn mm256_sra_epi16(a: m256i, count: m128i) -> m256i {
    unsafe { avx2_psra_w(a.as_i16x16(), count.as_i16x8()).as_m256i() }
}

// vpsrad
// __m256i _mm256_sra_epi32 (__m256i a, __m128i count)
#[inline]
pub fn mm256_sra_epi32(a: m256i, count: m128i) -> m256i {
    unsafe { avx2_psra_d(a.as_i32x8(), count.as_i32x4()).as_m256i() }
}

// vpsraw
// __m256i _mm256_srai_epi16 (__m256i a, int imm8)
#[inline]
pub fn mm256_srai_epi16(a: m256i, imm8: i32) -> m256i {
    let x: i16x16 = fn_imm8_arg1!(avx2_psrai_w, a.as_i16x16(), imm8);
    x.as_m256i()
}

// vpsrad
// __m256i _mm256_srai_epi32 (__m256i a, int imm8)
#[inline]
pub fn mm256_srai_epi32(a: m256i, imm8: i32) -> m256i {
    let x: i32x8 = fn_imm8_arg1!(avx2_psrai_d, a.as_i32x8(), imm8);
    x.as_m256i()
}

// vpsravd
// __m128i _mm_srav_epi32 (__m128i a, __m128i count)
#[inline]
pub fn mm_srav_epi32(a: m128i, count: m128i) -> m128i {
    unsafe { avx2_psrav_d(a.as_i32x4(), count.as_i32x4()).as_m128i() }
}

// vpsravd
// __m256i _mm256_srav_epi32 (__m256i a, __m256i count)
#[inline]
pub fn mm256_srav_epi32(a: m256i, count: m256i) -> m256i {
    unsafe { avx2_psrav_d_256(a.as_i32x8(), count.as_i32x8()).as_m256i() }
}

// vpsrlw
// __m256i _mm256_srl_epi16 (__m256i a, __m128i count)
#[inline]
pub fn mm256_srl_epi16(a: m256i, count: m128i) -> m256i {
    unsafe { avx2_psrl_w(a.as_i16x16(), count.as_i16x8()).as_m256i() }
}

// vpsrld
// __m256i _mm256_srl_epi32 (__m256i a, __m128i count)
#[inline]
pub fn mm256_srl_epi32(a: m256i, count: m128i) -> m256i {
    unsafe { avx2_psrl_d(a.as_i32x8(), count.as_i32x4()).as_m256i() }
}

// vpsrlq
// __m256i _mm256_srl_epi64 (__m256i a, __m128i count)
#[inline]
pub fn mm256_srl_epi64(a: m256i, count: m128i) -> m256i {
    unsafe { avx2_psrl_q(a.as_i64x4(), count.as_i64x2()).as_m256i() }
}

// vpsrlw
// __m256i _mm256_srli_epi16 (__m256i a, int imm8)
#[inline]
pub fn mm256_srli_epi16(a: m256i, imm8: i32) -> m256i {
    let x: i16x16 = fn_imm8_arg1!(avx2_psrli_w, a.as_i16x16(), imm8);
    x.as_m256i()
}

// vpsrld
// __m256i _mm256_srli_epi32 (__m256i a, int imm8)
#[inline]
pub fn mm256_srli_epi32(a: m256i, imm8: i32) -> m256i {
    let x: i32x8 = fn_imm8_arg1!(avx2_psrli_d, a.as_i32x8(), imm8);
    x.as_m256i()
}

// vpsrlq
// __m256i _mm256_srli_epi64 (__m256i a, int imm8)
#[inline]
pub fn mm256_srli_epi64(a: m256i, imm8: i32) -> m256i {
    let x: i64x4 = fn_imm8_arg1!(avx2_psrli_q, a.as_i64x4(), imm8);
    x.as_m256i()
}

// vpsrldq
// __m256i _mm256_srli_si256 (__m256i a, const int imm8)
#[inline]
pub fn mm256_srli_si256(a: m256i, imm8: i32) -> m256i {
    fn_imm8_arg1!(avx2_psrl_dq, a.as_i64x4(), imm8 * 8).as_m256i()
}

// vpsrlvd
// __m128i _mm_srlv_epi32 (__m128i a, __m128i count)
#[inline]
pub fn mm_srlv_epi32(a: m128i, count: m128i) -> m128i {
    unsafe { avx2_psrlv_d(a.as_i32x4(), count.as_i32x4()).as_m128i() }
}

// vpsrlvd
// __m256i _mm256_srlv_epi32 (__m256i a, __m256i count)
#[inline]
pub fn mm256_srlv_epi32(a: m256i, count: m256i) -> m256i {
    unsafe { avx2_psrlv_d_256(a.as_i32x8(), count.as_i32x8()).as_m256i() }
}

// vpsrlvq
// __m128i _mm_srlv_epi64 (__m128i a, __m128i count)
#[inline]
pub fn mm_srlv_epi64(a: m128i, count: m128i) -> m128i {
    unsafe { avx2_psrlv_q(a.as_i64x2(), count.as_i64x2()).as_m128i() }
}

// vpsrlvq
// __m256i _mm256_srlv_epi64 (__m256i a, __m256i count)
#[inline]
pub fn mm256_srlv_epi64(a: m256i, count: m256i) -> m256i {
    unsafe { avx2_psrlv_q_256(a.as_i64x4(), count.as_i64x4()).as_m256i() }
}

// vmovntdqa
// __m256i _mm256_stream_load_si256 (__m256i const* mem_addr)
#[inline]
#[allow(unused_variables)]
pub fn mm256_stream_load_si256(mem_addr: *const m256i) -> m256i {
    unimplemented!()
}

// vpsubw
// __m256i _mm256_sub_epi16 (__m256i a, __m256i b)
#[inline]
pub fn mm256_sub_epi16(a: m256i, b: m256i) -> m256i {
    unsafe { simd_sub(a.as_i16x16(), b.as_i16x16()).as_m256i() }
}

// vpsubd
// __m256i _mm256_sub_epi32 (__m256i a, __m256i b)
#[inline]
pub fn mm256_sub_epi32(a: m256i, b: m256i) -> m256i {
    unsafe { simd_sub(a.as_i32x8(), b.as_i32x8()).as_m256i() }
}

// vpsubq
// __m256i _mm256_sub_epi64 (__m256i a, __m256i b)
#[inline]
pub fn mm256_sub_epi64(a: m256i, b: m256i) -> m256i {
    unsafe { simd_sub(a.as_i64x4(), b.as_i64x4()).as_m256i() }
}

// vpsubb
// __m256i _mm256_sub_epi8 (__m256i a, __m256i b)
#[inline]
pub fn mm256_sub_epi8(a: m256i, b: m256i) -> m256i {
    unsafe { simd_sub(a.as_i8x32(), b.as_i8x32()).as_m256i() }
}

// vpsubsw
// __m256i _mm256_subs_epi16 (__m256i a, __m256i b)
#[inline]
pub fn mm256_subs_epi16(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_subs_epi16(a.as_i16x16(), b.as_i16x16()).as_m256i() }
}

// vpsubsb
// __m256i _mm256_subs_epi8 (__m256i a, __m256i b)
#[inline]
pub fn mm256_subs_epi8(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_subs_epi8(a.as_i8x32(), b.as_i8x32()).as_m256i() }
}

// vpsubusw
// __m256i _mm256_subs_epu16 (__m256i a, __m256i b)
#[inline]
pub fn mm256_subs_epu16(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_subs_epu16(a.as_u16x16(), b.as_u16x16()).as_m256i() }
}

// vpsubusb
// __m256i _mm256_subs_epu8 (__m256i a, __m256i b)
#[inline]
pub fn mm256_subs_epu8(a: m256i, b: m256i) -> m256i {
    unsafe { x86_mm256_subs_epu8(a.as_u8x32(), b.as_u8x32()).as_m256i() }
}

// vpunpckhwd
// __m256i _mm256_unpackhi_epi16 (__m256i a, __m256i b)
#[inline]
pub fn mm256_unpackhi_epi16(a: m256i, b: m256i) -> m256i {
    let x: i16x16 = unsafe {
        simd_shuffle16(a.as_i16x16(), b.as_i16x16(),
                       [4, 20, 5, 21, 6, 22, 7, 23, 12, 28, 13, 29, 14, 30, 15, 31])
    };
    x.as_m256i()
}

// vpunpckhdq
// __m256i _mm256_unpackhi_epi32 (__m256i a, __m256i b)
#[inline]
pub fn mm256_unpackhi_epi32(a: m256i, b: m256i) -> m256i {
    let x: i32x8 = unsafe {
        simd_shuffle8(a.as_i32x8(), b.as_i32x8(), [2, 10, 3, 11, 6, 14, 7, 15])
    };
    x.as_m256i()
}

// vpunpckhqdq
// __m256i _mm256_unpackhi_epi64 (__m256i a, __m256i b)
#[inline]
pub fn mm256_unpackhi_epi64(a: m256i, b: m256i) -> m256i {
    let x: i64x4 = unsafe {
        simd_shuffle4(a.as_i64x4(), b.as_i64x4(), [1, 5, 3, 7])
    };
    x.as_m256i()
}

// vpunpckhbw
// __m256i _mm256_unpackhi_epi8 (__m256i a, __m256i b)
#[inline]
pub fn mm256_unpackhi_epi8(a: m256i, b: m256i) -> m256i {
    let x: i8x32 = unsafe {
        simd_shuffle32(a.as_i8x32(), b.as_i8x32(),
                       [8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47,
                        24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63])
    };
    x.as_m256i()
}

// vpunpcklwd
// __m256i _mm256_unpacklo_epi16 (__m256i a, __m256i b)
#[inline]
pub fn mm256_unpacklo_epi16(a: m256i, b: m256i) -> m256i {
    let x: i16x16 = unsafe {
        simd_shuffle16(a.as_i16x16(), b.as_i16x16(),
                       [0, 16, 1, 17, 2, 18, 3, 19, 8, 24, 9, 25, 10, 26, 11, 27])
    };
    x.as_m256i()
}

// vpunpckldq
// __m256i _mm256_unpacklo_epi32 (__m256i a, __m256i b)
#[inline]
pub fn mm256_unpacklo_epi32(a: m256i, b: m256i) -> m256i {
    let x: i32x8 = unsafe {
        simd_shuffle8(a.as_i32x8(), b.as_i32x8(), [0, 8, 1, 9, 4, 12, 5, 13])
    };
    x.as_m256i()
}

// vpunpcklqdq
// __m256i _mm256_unpacklo_epi64 (__m256i a, __m256i b)
#[inline]
pub fn mm256_unpacklo_epi64(a: m256i, b: m256i) -> m256i {
    let x: i64x4 = unsafe {
        simd_shuffle4(a.as_i64x4(), b.as_i64x4(), [0, 4, 2, 6])
    };
    x.as_m256i()
}

// vpunpcklbw
// __m256i _mm256_unpacklo_epi8 (__m256i a, __m256i b)
#[inline]
pub fn mm256_unpacklo_epi8(a: m256i, b: m256i) -> m256i {
    let x: i8x32 = unsafe {
        simd_shuffle32(a.as_i8x32(), b.as_i8x32(),
                       [0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39,
                        16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55])
    };
    x.as_m256i()
}

// vpxor
// __m256i _mm256_xor_si256 (__m256i a, __m256i b)
pub fn mm256_xor_si256(a: m256i, b: m256i) -> m256i {
    unsafe { simd_xor(a, b) }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    fn seq8() -> m256i {
        mm256_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
    }
    fn seq8_128() -> m128i {
        mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    }
    fn mseq8() -> m256i {
        mm256_setr_epi8(-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16,
                        -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31, -32)
    }
    fn mseq8_128() -> m128i {
        mm_setr_epi8(-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16)
    }

    fn seq16() -> m256i {
        mm256_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    }
    fn seq16_128() -> m128i {
        mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8)
    }
    fn mseq16() -> m256i {
        mm256_setr_epi16(-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16)
    }
    fn mseq16_128() -> m128i {
        mm_setr_epi16(-1, -2, -3, -4, -5, -6, -7, -8)
    }

    fn seq32() -> m256i {
        mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8)
    }
    fn seq32_128() -> m128i {
        mm_setr_epi32(1, 2, 3, 4)
    }
    fn mseq32() -> m256i {
        mm256_setr_epi32(-1, -2, -3, -4, -5, -6, -7, -8)
    }
    fn mseq32_128() -> m128i {
        mm_setr_epi32(-1, -2, -3, -4)
    }

    fn seq64() -> m256i {
        mm256_setr_epi64x(1, 2, 3, 4)
    }
    fn seq64_128() -> m128i {
        mm_set_epi64x(2, 1)
    }
    fn mseq64() -> m256i {
        mm256_setr_epi64x(-1, -2, -3, -4)
    }
    fn mseq64_128() -> m128i {
        mm_set_epi64x(-2, -1)
    }

    fn seqps() -> m256 { mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0) }
    fn seqps_128() -> m128 { mm_setr_ps(1.0, 2.0, 3.0, 4.0) }
    fn seqpd() -> m256d { mm256_setr_pd(1.0, 2.0, 3.0, 4.0) }
    fn seqpd_128() -> m128d { mm_setr_pd(1.0, 2.0) }

    #[test]
    fn test_mm256_abs() {
        let a8 = mm256_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8,
                                 1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8);
        let a16 = mm256_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8);
        let a32 = mm256_setr_epi32(1, 2, 3, 4, -1, -2, -3, -4);

        assert_eq!(mm256_abs_epi8(a8).as_i8x32().as_array(),
                   [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
                    1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(mm256_abs_epi16(a16).as_i16x16().as_array(),
                   [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(mm256_abs_epi32(a32).as_i32x8().as_array(),
                   [1, 2, 3, 4, 1, 2, 3, 4]);
    }

    #[test]
    fn test_mm256_arith() {
        let a8 = mm256_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8,
                                 1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8);
        let a16 = mm256_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8);
        let a32 = mm256_setr_epi32(1, 2, 3, 4, -1, -2, -3, -4);
        let a64 = mm256_setr_epi64x(1, 2, -1, -2);

        assert_eq!(mm256_add_epi8(a8, a8).as_i8x32().as_array(),
                   [2, 4, 6, 8, 10, 12, 14, 16, -2, -4, -6, -8, -10, -12, -14, -16,
                    2, 4, 6, 8, 10, 12, 14, 16, -2, -4, -6, -8, -10, -12, -14, -16]);
        assert_eq!(mm256_add_epi16(a16, a16).as_i16x16().as_array(),
                   [2, 4, 6, 8, 10, 12, 14, 16, -2, -4, -6, -8, -10, -12, -14, -16]);
        assert_eq!(mm256_add_epi32(a32, a32).as_i32x8().as_array(),
                   [2, 4, 6, 8, -2, -4, -6, -8]);
        assert_eq!(mm256_add_epi64(a64, a64).as_i64x4().as_array(),
                   [2, 4, -2, -4]);

        assert_eq!(mm256_sub_epi8(seq8(), mseq8()).as_i8x32().as_array(),
                   [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
                    34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64]);
        assert_eq!(mm256_sub_epi16(seq16(), mseq16()).as_i16x16().as_array(),
                   [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]);
        assert_eq!(mm256_sub_epi32(seq32(), mseq32()).as_i32x8().as_array(),
                   [2, 4, 6, 8, 10, 12, 14, 16]);
        assert_eq!(mm256_sub_epi64(seq64(), mseq64()).as_i64x4().as_array(),
                   [2, 4, 6, 8]);
    }

    #[test]
    fn test_mm256_adds() {
        let a8 = mm256_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -0x80, 0x7F, -0x80, 0x7F,
                                 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -0x80, 0x7F, -0x80, 0x7F);
        let b8 = mm256_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0x7F, -0x80, -0x80, 0x7F,
                                 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0x7F, -0x80, -0x80, 0x7F);
        let a16 = mm256_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -0x8000, 0x7FFF, -0x8000, 0x7FFF);
        let b16 = mm256_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0x7FFF, -0x8000, -0x8000, 0x7FFF);

        assert_eq!(mm256_adds_epi8(a8, b8).as_i8x32().as_array(),
                   [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, -1, -1, -0x80, 0x7F,
                    2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, -1, -1, -0x80, 0x7F]);
        assert_eq!(mm256_adds_epu8(a8, b8).as_u8x32().as_array(),
                   [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 0xFF, 0xFF, 0xFF, 0xFE,
                    2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 0xFF, 0xFF, 0xFF, 0xFE]);

        assert_eq!(mm256_adds_epi16(a16, b16).as_i16x16().as_array(),
                   [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, -1, -1, -0x8000, 0x7FFF]);

        assert_eq!(mm256_adds_epu16(a16, b16).as_u16x16().as_array(),
                   [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFE]);
    }

    #[test]
    fn test_mm256_subs() {
        let a8 = mm256_setr_epi8(1, -3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b8 = mm256_setr_epi8(-0x80, 0x7F, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

        let a16 = mm256_setr_epi16(1, -3, 3, 4, 5, 6, 7, 8,
                                   1, 2, 3, 4, 5, 6, 7, 8);
        let b16 = mm256_setr_epi16(-0x8000, 0x7FFF, 1, 1, 1, 1, 1, 1,
                                   3, 1, 1, 1, 1, 1, 1, 1);


        assert_eq!(mm256_subs_epi8(a8, b8).as_i8x32().as_array(),
                   [127, -128, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                    -2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        assert_eq!(mm256_subs_epu8(a8, b8).as_u8x32().as_array(),
                   [0, 126, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);

        assert_eq!(mm256_subs_epi16(a16, b16).as_i16x16().as_array(),
                   [0x7FFF, -0x8000, 2, 3, 4, 5, 6, 7,
                    -2, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(mm256_subs_epu16(a16, b16).as_u16x16().as_array(),
                   [0, 0x7FFE, 2, 3, 4, 5, 6, 7,
                    0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_mm256_avg() {
        let a8 = mm256_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b8 = mm256_setr_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let a16 = mm256_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b16 = mm256_setr_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

        assert_eq!(mm256_avg_epu8(a8, b8).as_i8x32().as_array(),
                   [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9,
                    1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9]);
        assert_eq!(mm256_avg_epu16(a16, b16).as_i16x16().as_array(),
                   [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9]);
    }

    #[test]
    fn test_mm256_logic() {
        let a = mm256_setr_epi32(0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8);
        let b = mm256_setr_epi32(0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9);

        assert_eq!(mm256_and_si256(a, b).as_i32x8().as_array(),
                   [1 & 2, 2 & 3, 3 & 4, 4 & 5, 5 & 6, 6 & 7, 7 & 8, 8 & 9]);
        assert_eq!(mm256_andnot_si256(a, b).as_i32x8().as_array(),
                   [!1 & 2, !2 & 3, !3 & 4, !4 & 5, !5 & 6, !6 & 7, !7 & 8, !8 & 9]);
        assert_eq!(mm256_or_si256(a, b).as_i32x8().as_array(),
                   [1 | 2, 2 | 3, 3 | 4, 4 | 5, 5 | 6, 6 | 7, 7 | 8, 8 | 9]);
        assert_eq!(mm256_xor_si256(a, b).as_i32x8().as_array(),
                   [1 ^ 2, 2 ^ 3, 3 ^ 4, 4 ^ 5, 5 ^ 6, 6 ^ 7, 7 ^ 8, 8 ^ 9]);
    }

    #[test]
    fn test_blend() {
        {
            let mask = mm256_setr_epi8(0, !0, 0, !0, 0, !0, 0, !0, 0, !0, 0, !0, 0, !0, 0, !0,
                                       0, !0, 0, !0, 0, !0, 0, !0, 0, !0, 0, !0, 0, !0, 0, !0);

            assert_eq!(mm256_blendv_epi8(seq8(), mseq8(), mask).as_i8x32().as_array(),
                       [1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
                        17, -18, 19, -20, 21, -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32]);
        }
        {
            let a = mm256_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
            let b = mm256_setr_epi16(17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);

            assert_eq!(mm256_blend_epi16(a, b, 0).as_i16x16().as_array(),
                       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
            assert_eq!(mm256_blend_epi16(a, b, 0xFF).as_i16x16().as_array(),
                       [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]);
            assert_eq!(mm256_blend_epi16(a, b, 0x11).as_i16x16().as_array(),
                       [17, 2, 3, 4, 21, 6, 7, 8, 25, 10, 11, 12, 29, 14, 15, 16]);
        }
        {
            let a = mm_setr_epi32(1, 2, 3, 4);
            let b = mm_setr_epi32(11, 12, 13, 14);

            assert_eq!(mm_blend_epi32(a, b, 0).as_i32x4().as_array(), [1, 2, 3, 4]);
            assert_eq!(mm_blend_epi32(a, b, 0xFF).as_i32x4().as_array(), [11, 12, 13, 14]);
            assert_eq!(mm_blend_epi32(a, b, 0x03).as_i32x4().as_array(), [11, 12, 3, 4]);
        }
        {
            let a = mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
            let b = mm256_setr_epi32(11, 12, 13, 14, 15, 16, 17, 18);

            assert_eq!(mm256_blend_epi32(a, b, 0).as_i32x8().as_array(), [1, 2, 3, 4, 5, 6, 7, 8]);
            assert_eq!(mm256_blend_epi32(a, b, 0xFF).as_i32x8().as_array(), [11, 12, 13, 14, 15, 16, 17, 18]);
            assert_eq!(mm256_blend_epi32(a, b, 0x11).as_i32x8().as_array(), [11, 2, 3, 4, 15, 6, 7, 8]);
        }
    }

    #[test]
    fn test_broadcast() {
        assert_eq!(mm_broadcastb_epi8(seq8_128()).as_i8x16().as_array(),
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(mm256_broadcastb_epi8(seq8_128()).as_i8x32().as_array(),
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(mm_broadcastd_epi32(seq32_128()).as_i32x4().as_array(),
                   [1, 1, 1, 1]);
        assert_eq!(mm256_broadcastd_epi32(seq32_128()).as_i32x8().as_array(),
                   [1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(mm_broadcastq_epi64(seq64_128()).as_i64x2().as_array(),
                   [1, 1]);
        assert_eq!(mm256_broadcastq_epi64(seq64_128()).as_i64x4().as_array(),
                   [1, 1, 1, 1]);
        assert_eq!(mm_broadcastw_epi16(seq16_128()).as_i16x8().as_array(),
                   [1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(mm256_broadcastw_epi16(seq16_128()).as_i16x16().as_array(),
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);

        assert_eq!(mm_broadcastsd_pd(seqpd_128()).as_f64x2().as_array(),
                   [1.0, 1.0]);
        assert_eq!(mm256_broadcastsd_pd(seqpd_128()).as_f64x4().as_array(),
                   [1.0, 1.0, 1.0, 1.0]);

        assert_eq!(mm256_broadcastsi128_si256(seq64_128()).as_i64x4().as_array(),
                   [1, 2, 1, 2]);

        assert_eq!(mm_broadcastss_ps(seqps_128()).as_f32x4().as_array(),
                   [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(mm256_broadcastss_ps(seqps_128()).as_f32x8().as_array(),
                   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_byte_shift() {
        assert_eq!(mm256_bslli_epi128(seq8(), 3).as_i8x32().as_array(),
                   [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                    0, 0, 0, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]);
        assert_eq!(mm256_bsrli_epi128(seq8(), 3).as_i8x32().as_array(),
                   [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0,
                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 0, 0, 0]);
    }

    #[test]
    fn test_shift() {
        let one = mm_setr_epi32(1, 0, 0, 0);
        assert_eq!(mm256_sll_epi16(seq16(), one).as_i16x16().as_array(),
                   [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]);
        assert_eq!(mm256_sll_epi32(seq32(), one).as_i32x8().as_array(),
                   [2, 4, 6, 8, 10, 12, 14, 16]);
        assert_eq!(mm256_sll_epi64(seq64(), one).as_i64x4().as_array(),
                   [2, 4, 6, 8]);

        assert_eq!(mm256_slli_epi16(seq16(), 1).as_i16x16().as_array(),
                   [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]);
        assert_eq!(mm256_slli_epi32(seq32(), 1).as_i32x8().as_array(),
                   [2, 4, 6, 8, 10, 12, 14, 16]);
        assert_eq!(mm256_slli_epi64(seq64(), 1).as_i64x4().as_array(),
                   [2, 4, 6, 8]);

        assert_eq!(mm256_slli_si256(seq8(), 3).as_i8x32().as_array(),
                   [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                    0, 0, 0, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]);

        assert_eq!(mm_sllv_epi32(seq32_128(), seq32_128()).as_i32x4().as_array(),
                   [1 << 1, 2 << 2, 3 << 3, 4 << 4]);
        assert_eq!(mm256_sllv_epi32(seq32(), seq32()).as_i32x8().as_array(),
                   [1 << 1, 2 << 2, 3 << 3, 4 << 4, 5 << 5, 6 << 6, 7 << 7, 8 << 8]);
        assert_eq!(mm_sllv_epi64(seq64_128(), seq64_128()).as_i64x2().as_array(),
                   [1 << 1, 2 << 2]);
        assert_eq!(mm256_sllv_epi64(seq64(), seq64()).as_i64x4().as_array(),
                   [1 << 1, 2 << 2, 3 << 3, 4 << 4]);

        assert_eq!(mm256_sra_epi16(mseq16(), one).as_i16x16().as_array(),
                   [-1, -1, -2, -2, -3, -3, -4, -4, -5, -5, -6, -6, -7, -7, -8, -8]);
        assert_eq!(mm256_sra_epi32(mseq32(), one).as_i32x8().as_array(),
                   [-1, -1, -2, -2, -3, -3, -4, -4]);

        assert_eq!(mm256_srai_epi16(mseq16(), 1).as_i16x16().as_array(),
                   [-1, -1, -2, -2, -3, -3, -4, -4, -5, -5, -6, -6, -7, -7, -8, -8]);
        assert_eq!(mm256_srai_epi32(mseq32(), 1).as_i32x8().as_array(),
                   [-1, -1, -2, -2, -3, -3, -4, -4]);

        assert_eq!(mm_srav_epi32(mseq32_128(), mm_setr_epi32(1, 0, 1, 2)).as_i32x4().as_array(),
                   [-1, -2, -2, -1]);
        assert_eq!(mm256_srav_epi32(mseq32(), mm256_setr_epi32(1, 1, 1, 1, 0, 0, 0, 0)).as_i32x8().as_array(),
                   [-1, -1, -2, -2, -5, -6, -7, -8]);

        assert_eq!(mm256_srl_epi16(mseq16(), one).as_u16x16().as_array(),
                   [0x7FFF, 0x7FFF, 0x7FFE, 0x7FFE, 0x7FFD, 0x7FFD, 0x7FFC, 0x7FFC,
                    0x7FFB, 0x7FFB, 0x7FFA, 0x7FFA, 0x7FF9, 0x7FF9, 0x7FF8, 0x7FF8]);
        assert_eq!(mm256_srl_epi32(mseq32(), one).as_u32x8().as_array(),
                   [0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFE, 0x7FFFFFFE,
                    0x7FFFFFFD, 0x7FFFFFFD, 0x7FFFFFFC, 0x7FFFFFFC]);
        assert_eq!(mm256_srl_epi64(mseq64(), one).as_u64x4().as_array(),
                   [0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFE, 0x7FFFFFFFFFFFFFFE]);

        assert_eq!(mm256_srli_epi16(mseq16(), 1).as_u16x16().as_array(),
                   [0x7FFF, 0x7FFF, 0x7FFE, 0x7FFE, 0x7FFD, 0x7FFD, 0x7FFC, 0x7FFC,
                    0x7FFB, 0x7FFB, 0x7FFA, 0x7FFA, 0x7FF9, 0x7FF9, 0x7FF8, 0x7FF8]);
        assert_eq!(mm256_srli_epi32(mseq32(), 1).as_u32x8().as_array(),
                   [0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFE, 0x7FFFFFFE, 0x7FFFFFFD, 0x7FFFFFFD, 0x7FFFFFFC, 0x7FFFFFFC]);
        assert_eq!(mm256_srli_epi64(mseq64(), 1).as_u64x4().as_array(),
                   [0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFE, 0x7FFFFFFFFFFFFFFE]);

        assert_eq!(mm256_srli_si256(seq8(), 3).as_u8x32().as_array(),
                   [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0,
                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 0, 0, 0]);

        assert_eq!(mm_srlv_epi32(mseq32_128(), mm_setr_epi32(1, 0, 1, 2)).as_u32x4().as_array(),
                   [0x7FFFFFFF, 0xFFFFFFFE, 0x7FFFFFFE, 0x3FFFFFFF]);
        assert_eq!(mm256_srlv_epi32(mseq32(), mm256_setr_epi32(1, 1, 1, 1, 0, 0, 0, 0)).as_u32x8().as_array(),
                   [0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFE, 0x7FFFFFFE,
                    0xFFFFFFFB, 0xFFFFFFFA, 0xFFFFFFF9, 0xFFFFFFF8]);
        assert_eq!(mm256_srlv_epi64(mseq64(), mm256_setr_epi64x(1, 1, 0, 0)).as_u64x4().as_array(),
                   [0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFD, 0xFFFFFFFFFFFFFFFC]);
        }

    #[test]
    fn test_cmp() {
        let x8 = mm256_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
                                 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50);
        let x16 = mm256_setr_epi16(1, 2, 3, 4, 1, 2, 3, 4, 50, 50, 50, 50, 50, 50, 50, 50);
        let x32 = mm256_setr_epi32(1, 2, 1, 2, 50, 50, 50, 50);
        let x64 = mm256_setr_epi64x(1, 1, 50, 50);

        assert_eq!(mm256_cmpeq_epi8(seq8(), x8).as_i8x32().as_array(),
                   [!0, !0, !0, !0, !0, !0, !0, !0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(mm256_cmpeq_epi16(seq16(), x16).as_i16x16().as_array(),
                   [!0, !0, !0, !0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(mm256_cmpeq_epi32(seq32(), x32).as_i32x8().as_array(),
                   [!0, !0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(mm256_cmpeq_epi64(seq64(), x64).as_i64x4().as_array(),
                   [!0, 0, 0, 0]);

        assert_eq!(mm256_cmpgt_epi8(seq8(), x8).as_i8x32().as_array(),
                   [0, 0, 0, 0, 0, 0, 0, 0, !0, !0, !0, !0, !0, !0, !0, !0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(mm256_cmpgt_epi16(seq16(), x16).as_i16x16().as_array(),
                   [0, 0, 0, 0, !0, !0, !0, !0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(mm256_cmpgt_epi32(seq32(), x32).as_i32x8().as_array(),
                   [0, 0, !0, !0, 0, 0, 0, 0]);
        assert_eq!(mm256_cmpgt_epi64(seq64(), x64).as_i64x4().as_array(),
                   [0, !0, 0, 0]);
    }

    #[test]
    fn test_extract() {
        assert_eq!(mm256_extracti128_si256(seq64(), 0).as_i64x2().as_array(), [1, 2]);
        assert_eq!(mm256_extracti128_si256(seq64(), 1).as_i64x2().as_array(), [3, 4]);
    }

    #[test]
    fn test_insert() {
        assert_eq!(mm256_inserti128_si256(seq64(), mseq64_128(), 0).as_i64x4().as_array(), [-1, -2, 3, 4]);
        assert_eq!(mm256_inserti128_si256(seq64(), mseq64_128(), 1).as_i64x4().as_array(), [1, 2, -1, -2]);
    }

    #[test]
    fn test_hadd() {
        let x16 = mm256_setr_epi16(1, 2, 0x7000, 0x7000, -1, -2, -0x7000, -0x7000,
                                   1, 2, 0x7000, 0x7000, -1, -2, -0x7000, -0x7000);
        let x32 = mm256_setr_epi32(1, 2, -1, -2, 1, 2, -1, -2);

        // 0x7000 + 0x7000 = 0xE000
        let e = 0xE000u16 as i16;
        assert_eq!(mm256_hadd_epi16(x16, x16).as_i16x16().as_array(),
                   [3, e, -3, -e, 3, e, -3, -e, 3, e, -3, -e, 3, e, -3, -e]);
        assert_eq!(mm256_hadd_epi32(x32, x32).as_i32x8().as_array(),
                   [3, -3, 3, -3, 3, -3, 3, -3]);
        assert_eq!(mm256_hadds_epi16(x16, x16).as_i16x16().as_array(),
                   [3, 0x7FFF, -3, -0x8000, 3, 0x7FFF, -3, -0x8000,
                    3, 0x7FFF, -3, -0x8000, 3, 0x7FFF, -3, -0x8000]);
    }

    #[test]
    fn test_hsub() {
        let x16 = mm256_setr_epi16(1, 2, 0x7000, -0x7000, -1, -2, -0x7000, 0x7000,
                                   1, 2, 0x7000, -0x7000, -1, -2, -0x7000, 0x7000);
        let x32 = mm256_setr_epi32(1, 2, -1, -2, 1, 2, -1, -2);

        // 0x7000 + 0x7000 = 0xE000
        let e = 0xE000u16 as i16;
        assert_eq!(mm256_hsub_epi16(x16, x16).as_i16x16().as_array(),
                   [-1, e, 1, -e, -1, e, 1, -e, -1, e, 1, -e, -1, e, 1, -e]);
        assert_eq!(mm256_hsub_epi32(x32, x32).as_i32x8().as_array(),
                   [-1, 1, -1, 1, -1, 1, -1, 1]);
        assert_eq!(mm256_hsubs_epi16(x16, x16).as_i16x16().as_array(),
                   [-1, 0x7FFF, 1, -0x8000, -1, 0x7FFF, 1, -0x8000,
                    -1, 0x7FFF, 1, -0x8000, -1, 0x7FFF, 1, -0x8000]);
    }

    #[test]
    fn test_madd() {
        assert_eq!(mm256_madd_epi16(seq16(), seq16()).as_i32x8().as_array(),
                   [1 * 1 + 2 * 2, 3 * 3 + 4 * 4, 5 * 5 + 6 * 6, 7 * 7 + 8 * 8,
                    9 * 9 + 10 * 10, 11 * 11 + 12 * 12, 13 * 13 + 14 * 14, 15 * 15 + 16 * 16]);
        assert_eq!(mm256_maddubs_epi16(seq8(), seq8()).as_i16x16().as_array(),
                   [1 * 1 + 2 * 2, 3 * 3 + 4 * 4, 5 * 5 + 6 * 6, 7 * 7 + 8 * 8,
                    9 * 9 + 10 * 10, 11 * 11 + 12 * 12, 13 * 13 + 14 * 14, 15 * 15 + 16 * 16,
                    17 * 17 + 18 * 18, 19 * 19 + 20 * 20, 21 * 21 + 22 * 22, 23 * 23 + 24 * 24,
                    25 * 25 + 26 * 26, 27 * 27 + 28 * 28, 29 * 29 + 30 * 30, 31 * 31 + 32 * 32]);
    }

    #[test]
    fn test_maskload() {
        let a32_128 = seq32_128();
        let a64_128 = seq64_128();
        let a32_256 = seq32();
        let a64_256 = seq64();

        let p32_128 = &a32_128 as *const m128i as *const i32;
        let p64_128 = &a64_128 as *const m128i as *const i64;
        let p32_256 = &a32_256 as *const m256i as *const i32;
        let p64_256 = &a64_256 as *const m256i as *const i64;

        let m32_128 = mm_setr_epi32(0, !0, 0, !0);
        let m64_128 = mm_set_epi64x(!0, 0);
        let m32_256 = mm256_setr_epi32(0, !0, 0, !0, 0, !0, 0, !0);
        let m64_256 = mm256_setr_epi64x(0, !0, 0, !0);

        unsafe {
            assert_eq!(mm_maskload_epi32(p32_128, m32_128).as_i32x4().as_array(), [0, 2, 0, 4]);
            assert_eq!(mm_maskload_epi64(p64_128, m64_128).as_i64x2().as_array(), [0, 2]);
            assert_eq!(mm256_maskload_epi32(p32_256, m32_256).as_i32x8().as_array(), [0, 2, 0, 4, 0, 6, 0, 8]);
            assert_eq!(mm256_maskload_epi64(p64_256, m64_256).as_i64x4().as_array(), [0, 2, 0, 4]);
        }
    }

    #[test]
    fn test_maskstore() {
        let mut a32_128 = mm_setzero_si128();
        let mut a64_128 = mm_setzero_si128();
        let mut a32_256 = mm256_setzero_si256();
        let mut a64_256 = mm256_setzero_si256();

        let m32_128 = mm_setr_epi32(0, !0, 0, !0);
        let m64_128 = mm_set_epi64x(!0, 0);
        let m32_256 = mm256_setr_epi32(0, !0, 0, !0, 0, !0, 0, !0);
        let m64_256 = mm256_setr_epi64x(0, !0, 0, !0);

        unsafe {
            let p32_128 = &mut a32_128 as *mut m128i as *mut i32;
            let p64_128 = &mut a64_128 as *mut m128i as *mut i64;
            let p32_256 = &mut a32_256 as *mut m256i as *mut i32;
            let p64_256 = &mut a64_256 as *mut m256i as *mut i64;

            mm_maskstore_epi32(p32_128, m32_128, seq32_128());
            mm_maskstore_epi64(p64_128, m64_128, seq64_128());
            mm256_maskstore_epi32(p32_256, m32_256, seq32());
            mm256_maskstore_epi64(p64_256, m64_256, seq64());
        };

        assert_eq!(a32_128.as_i32x4().as_array(), [0, 2, 0, 4]);
        assert_eq!(a64_128.as_i64x2().as_array(), [0, 2]);
        assert_eq!(a32_256.as_i32x8().as_array(), [0, 2, 0, 4, 0, 6, 0, 8]);
        assert_eq!(a64_256.as_i64x4().as_array(), [0, 2, 0, 4]);
    }

    #[test]
    fn test_minmax() {
        assert_eq!(mm256_max_epi8(seq8(), mseq8()).as_i8x32().as_array(), seq8().as_i8x32().as_array());
        assert_eq!(mm256_max_epi16(seq16(), mseq16()).as_i16x16().as_array(), seq16().as_i16x16().as_array());
        assert_eq!(mm256_max_epi32(seq32(), mseq32()).as_i32x8().as_array(), seq32().as_i32x8().as_array());

        assert_eq!(mm256_max_epu8(seq8(), mseq8()).as_i8x32().as_array(), mseq8().as_i8x32().as_array());
        assert_eq!(mm256_max_epu16(seq16(), mseq16()).as_i16x16().as_array(), mseq16().as_i16x16().as_array());
        assert_eq!(mm256_max_epu32(seq32(), mseq32()).as_i32x8().as_array(), mseq32().as_i32x8().as_array());

        assert_eq!(mm256_min_epi8(seq8(), mseq8()).as_i8x32().as_array(), mseq8().as_i8x32().as_array());
        assert_eq!(mm256_min_epi16(seq16(), mseq16()).as_i16x16().as_array(), mseq16().as_i16x16().as_array());
        assert_eq!(mm256_min_epi32(seq32(), mseq32()).as_i32x8().as_array(), mseq32().as_i32x8().as_array());

        assert_eq!(mm256_min_epu8(seq8(), mseq8()).as_i8x32().as_array(), seq8().as_i8x32().as_array());
        assert_eq!(mm256_min_epu16(seq16(), mseq16()).as_i16x16().as_array(), seq16().as_i16x16().as_array());
        assert_eq!(mm256_min_epu32(seq32(), mseq32()).as_i32x8().as_array(), seq32().as_i32x8().as_array());
    }

    #[test]
    fn test_movemask() {
        let a = mm256_setr_epi8(1, 2, 3, 4, -1, -2, -3, -4, 1, 2, 3, 4, -1, -2, -3, -4,
                                1, 2, 3, 4, -1, -2, -3, -4, 1, 2, 3, 4, -1, -2, -3, -4);
        assert_eq!(mm256_movemask_epi8(a), 0xF0F0F0F0u32 as i32);
    }

    #[test]
    fn test_mpsadbw() {
        let a = u8x32(15, 60, 55, 31, 0, 1, 2, 4, 8, 16, 32, 64, 128, 255, 1, 17,
                      15, 60, 55, 31, 0, 1, 2, 4, 8, 16, 32, 64, 128, 255, 1, 17).as_m256i();
        let b = u8x32(2, 4, 8, 64, 255, 0, 1, 16, 32, 64, 128, 255, 75, 31, 42, 11,
                      2, 4, 8, 64, 255, 0, 1, 16, 32, 64, 128, 255, 75, 31, 42, 11).as_m256i();
        assert_eq!(mm256_mpsadbw_epu8(a, b, 1 | 4 | 32 | 8).as_u16x16().as_array(),
                   [269, 267, 264, 290, 342, 446, 653, 588,
                    269, 267, 264, 290, 342, 446, 653, 588]);
    }

    #[test]
    fn test_mul() {
        assert_eq!(mm256_mul_epi32(seq32(), mseq32()).as_i64x4().as_array(),
                   [-1, -9, -25, -49]);
        assert_eq!(mm256_mul_epu32(seq32(), mseq32()).as_u64x4().as_array(),
                   [4294967295, 12884901879, 21474836455, 30064771023]);

        let x = mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7);
        let y = mm256_setr_epi16(0, 1, 2, 3, -4, -5, -6, -7, 0, 1, 2, 3, -4, -5, -6, -7);

        assert_eq!(mm256_mulhi_epi16(x, y).as_i16x16().as_array(),
                   [0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1]);
        assert_eq!(mm256_mulhi_epu16(x, y).as_u16x16().as_array(),
                   [0, 0, 0, 0, 3, 4, 5, 6, 0, 0, 0, 0, 3, 4, 5, 6]);

        assert_eq!(mm256_mullo_epi16(x, y).as_i16x16().as_array(),
                   [0, 1, 4, 9, -16, -25, -36, -49, 0, 1, 4, 9, -16, -25, -36, -49]);
        assert_eq!(mm256_mullo_epi32(seq32(), mseq32()).as_i32x8().as_array(),
                   [-1, -4, -9, -16, -25, -36, -49, -64]);

        let x16 = mm256_setr_epi16(-0x5CEE, 0x0105, 0x3DA9, -0x7FFF, 0x7FFF, 0x1111, -0x219D, -0x1DBC,
                                   -0x5CEE, 0x0105, 0x3DA9, -0x7FFF, 0x7FFF, 0x1111, -0x219D, -0x1DBC);
        let y16 = mm256_setr_epi16(0x4000, -0x510A, 0x209D, -0x7FFF, 0x0000, 0x2222, 0x1027, 0x7AEF,
                                   0x4000, -0x510A, 0x209D, -0x7FFF, 0x0000, 0x2222, 0x1027, 0x7AEF);

        assert_eq!(mm256_mulhrs_epi16(x16, y16).as_i16x16().as_array(),
                   [-11895, -165, 4022, 32766, 0, 1165, -1086, -7311,
                    -11895, -165, 4022, 32766, 0, 1165, -1086, -7311]);
    }

    #[test]
    fn test_pack() {
        let a = mm256_setr_epi16(1, -1, 0x7FFF, -0x8000, 1, -1, 0x7FFF, -0x8000, 1, -1, 0x7FFF, -0x8000, 1, -1, 0x7FFF, -0x8000);
        let b = mm256_setr_epi32(1, -1, 0x7FFFFFF, -0x8000000, 1, -1, 0x7FFFFFFF, -0x80000000);

        assert_eq!(mm256_packs_epi16(a, a).as_i8x32().as_array(),
                   [1, -1, 127, -128, 1, -1, 127, -128, 1, -1, 127, -128, 1, -1, 127, -128,
                    1, -1, 127, -128, 1, -1, 127, -128, 1, -1, 127, -128, 1, -1, 127, -128]);
        assert_eq!(mm256_packus_epi16(a, a).as_u8x32().as_array(),
                   [1, 0, 255, 0, 1, 0, 255, 0, 1, 0, 255, 0, 1, 0, 255, 0,
                    1, 0, 255, 0, 1, 0, 255, 0, 1, 0, 255, 0, 1, 0, 255, 0]);

        assert_eq!(mm256_packs_epi32(b, b).as_i16x16().as_array(),
                   [1, -1, 0x7FFF, -0x8000, 1, -1, 0x7FFF, -0x8000, 1, -1, 0x7FFF, -0x8000, 1, -1, 0x7FFF, -0x8000]);
        assert_eq!(mm256_packus_epi32(b, b).as_u16x16().as_array(),
                   [1, 0, 0xFFFF, 0, 1, 0, 0xFFFF, 0, 1, 0, 0xFFFF, 0, 1, 0, 0xFFFF, 0]);
    }

    #[test]
    fn test_permute() {
        assert_eq!(mm256_permute2x128_si256(seq64(), mseq64(), (1 << 7) | (1 << 3)).as_i64x4().as_array(),
                   [0, 0, 0, 0]);
        assert_eq!(mm256_permute2x128_si256(seq64(), mseq64(), (3 << 0) | (1 << 4)).as_i64x4().as_array(),
                   [-3, -4, 3, 4]);

        assert_eq!(mm256_permute4x64_epi64(seq64(), (1 << 0) | (0 << 2) | (3 << 4) | (2 << 6)).as_i64x4().as_array(),
                   [2, 1, 4, 3]);
        assert_eq!(mm256_permute4x64_pd(seqpd(), (1 << 0) | (0 << 2) | (3 << 4) | (2 << 6)).as_f64x4().as_array(),
                   [2.0, 1.0, 4.0, 3.0]);

        let idx = mm256_setr_epi32(3, 4, 5, 6, 1, 2, 3, 0);
        assert_eq!(mm256_permutevar8x32_epi32(seq32(), idx).as_i32x8().as_array(),
                   [4, 5, 6, 7, 2, 3, 4, 1]);
        assert_eq!(mm256_permutevar8x32_ps(seqps(), idx).as_f32x8().as_array(),
                   [4.0, 5.0, 6.0, 7.0, 2.0, 3.0, 4.0, 1.0]);
    }

    #[test]
    fn test_sad() {
        let x8 = mm256_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let y8 = mm256_setr_epi8(5, 5, 5, 5, 5, 5, 5, 5, 5,  5,  5,  5,  5,  5,  5,  5,
                                 5, 5, 5, 5, 5, 5, 5, 5, 5,  5,  5,  5,  5,  5,  5,  5);

        assert_eq!(mm256_sad_epu8(x8, y8).as_i64x4().as_array(), [16, 60, 16, 60]);
    }

    #[test]
    fn test_sign() {
        let idx8 = mm256_setr_epi8(0, 1, 2, 3, -1, -2, -3, -4, 0, 1, 2, 3, -1, -2, -3, -4,
                                   0, 1, 2, 3, -1, -2, -3, -4, 0, 1, 2, 3, -1, -2, -3, -4);
        let idx16 = mm256_setr_epi16(0, 1, -1, 0, 1, -1, 0, 1, 0, 1, -1, 0, 1, -1, 0, 1);
        let idx32 = mm256_setr_epi32(0, 1, -1, 0, 0, 1, -1, 0);

        assert_eq!(mm256_sign_epi8(seq8(), idx8).as_i8x32().as_array(),
                   [0, 2, 3, 4, -5, -6, -7, -8, 0, 10, 11, 12, -13, -14, -15, -16,
                    0, 18, 19, 20, -21, -22, -23, -24, 0, 26, 27, 28, -29, -30, -31, -32]);
        assert_eq!(mm256_sign_epi16(seq16(), idx16).as_i16x16().as_array(),
                   [0, 2, -3, 0, 5, -6, 0, 8, 0, 10, -11, 0, 13, -14, 0, 16]);
        assert_eq!(mm256_sign_epi32(seq32(), idx32).as_i32x8().as_array(),
                   [0, 2, -3, 0, 0, 6, -7, 0]);
    }

    #[test]
    fn test_shuffle_epi32() {
        let s32 = mm256_shuffle_epi32(seq32(), (2 << 0) | (0 << 2) | (3 << 4) | (1 << 6));
        assert_eq!(s32.as_i32x8().as_array(),
                   [3, 1, 4, 2, 7, 5, 8, 6]);

        let h16 = mm256_shufflehi_epi16(seq16(), (2 << 0) | (0 << 2) | (3 << 4) | (1 << 6));
        assert_eq!(h16.as_i16x16().as_array(),
                   [1, 2, 3, 4, 7, 5, 8, 6, 9, 10, 11, 12, 15, 13, 16, 14]);
        let l16 = mm256_shufflelo_epi16(seq16(), (2 << 0) | (0 << 2) | (3 << 4) | (1 << 6));
        assert_eq!(l16.as_i16x16().as_array(),
                   [3, 1, 4, 2, 5, 6, 7, 8, 11, 9, 12, 10, 13, 14, 15, 16]);
    }

    #[test]
    fn test_shuffle_epi8() {
        let x8 = mm256_setr_epi8(51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                                 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66);
        let idx = mm256_setr_epi8(4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11,
                                  4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11);

        assert_eq!(mm256_shuffle_epi8(x8, idx).as_i8x32().as_array(),
                   [55, 56, 57, 58, 51, 52, 53, 54, 63, 64, 65, 66, 59, 60, 61, 62,
                    55, 56, 57, 58, 51, 52, 53, 54, 63, 64, 65, 66, 59, 60, 61, 62]);
    }

    #[test]
    fn test_unpack() {
        assert_eq!(mm256_unpacklo_epi8(seq8(), mseq8()).as_i8x32().as_array(),
                   [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8,
                    17, -17, 18, -18, 19, -19, 20, -20, 21, -21, 22, -22, 23, -23, 24, -24]);
        assert_eq!(mm256_unpackhi_epi8(seq8(), mseq8()).as_i8x32().as_array(),
                   [9, -9, 10, -10, 11, -11, 12, -12, 13, -13, 14, -14, 15, -15, 16, -16,
                    25, -25, 26, -26, 27, -27, 28, -28, 29, -29, 30, -30, 31, -31, 32, -32]);

        assert_eq!(mm256_unpacklo_epi16(seq16(), mseq16()).as_i16x16().as_array(),
                   [1, -1, 2, -2, 3, -3, 4, -4, 9, -9, 10, -10, 11, -11, 12, -12]);
        assert_eq!(mm256_unpackhi_epi16(seq16(), mseq16()).as_i16x16().as_array(),
                   [5, -5, 6, -6, 7, -7, 8, -8, 13, -13, 14, -14, 15, -15, 16, -16]);

        assert_eq!(mm256_unpacklo_epi32(seq32(), mseq32()).as_i32x8().as_array(),
                   [1, -1, 2, -2, 5, -5, 6, -6]);
        assert_eq!(mm256_unpackhi_epi32(seq32(), mseq32()).as_i32x8().as_array(),
                   [3, -3, 4, -4, 7, -7, 8, -8]);

        assert_eq!(mm256_unpacklo_epi64(seq64(), mseq64()).as_i64x4().as_array(),
                   [1, -1, 3, -3]);
        assert_eq!(mm256_unpackhi_epi64(seq64(), mseq64()).as_i64x4().as_array(),
                   [2, -2, 4, -4]);
    }

    #[test]
    fn test_gather_epi32() {
        let mut a = [0i32; 128];
        for i in 0..128 {
            a[i] = i as i32
        }

        let p = &a as *const [i32; 128] as *const i32;

        unsafe {
            let index = mm_setr_epi32(6, 10, 4, 8);
            let x = mm_i32gather_epi32(p, index, 4);
            assert_eq!(x.as_i32x4().as_array(), [6, 10, 4, 8])
        };
        unsafe {
            let index = mm256_setr_epi32(6, 10, 4, 8, 1, 4, 2, 3);
            let x = mm256_i32gather_epi32(p, index, 4);
            assert_eq!(x.as_i32x8().as_array(), [6, 10, 4, 8, 1, 4, 2, 3])
        };
        unsafe {
            let index = mm_setr_epi64x(6, 10);
            let x = mm_i64gather_epi32(p, index, 4);
            assert_eq!(x.as_i32x4().as_array(), [6, 10, 0, 0])
        };
        unsafe {
            let index = mm256_setr_epi64x(6, 10, 4, 8);
            let x = mm256_i64gather_epi32(p, index, 4);
            assert_eq!(x.as_i32x4().as_array(), [6, 10, 4, 8])
        }
    }

    #[test]
    fn test_gather_epi64() {
        let mut a = [0i64; 128];
        for i in 0..128 {
            a[i] = i as i64
        }

        let p = &a as *const [i64; 128] as *const i64;

        unsafe {
            let index = mm_setr_epi32(6, 10, 4, 8);
            let x = mm_i32gather_epi64(p, index, 8);
            assert_eq!(x.as_i64x2().as_array(), [6, 10])
        };
        unsafe {
            let index = mm_setr_epi32(6, 10, 4, 8);
            let x = mm256_i32gather_epi64(p, index, 8);
            assert_eq!(x.as_i64x4().as_array(), [6, 10, 4, 8])
        };
        unsafe {
            let index = mm_setr_epi64x(6, 10);
            let x = mm_i64gather_epi64(p, index, 8);
            assert_eq!(x.as_i64x2().as_array(), [6, 10])
        };
        unsafe {
            let index = mm256_setr_epi64x(6, 10, 4, 8);
            let x = mm256_i64gather_epi64(p, index, 8);
            assert_eq!(x.as_i64x4().as_array(), [6, 10, 4, 8])
        }
    }

    #[test]
    fn test_gather_f32() {
        let mut a = [0.0f32; 128];
        for i in 0..128 {
            a[i] = i as f32
        }

        let p = &a as *const [f32; 128] as *const f32;

        unsafe {
            let index = mm_setr_epi32(6, 10, 4, 8);
            let x = mm_i32gather_ps(p, index, 4);
            assert_eq!(x.as_f32x4().as_array(), [6.0, 10.0, 4.0, 8.0])
        };
        unsafe {
            let index = mm256_setr_epi32(6, 10, 4, 8, 1, 2, 3, 4);
            let x = mm256_i32gather_ps(p, index, 4);
            assert_eq!(x.as_f32x8().as_array(), [6.0, 10.0, 4.0, 8.0, 1.0, 2.0, 3.0, 4.0])
        };
        unsafe {
            let index = mm_setr_epi64x(6, 10);
            let x = mm_i64gather_ps(p, index, 4);
            assert_eq!(x.as_f32x4().as_array(), [6.0, 10.0, 0.0, 0.0])
        };
        unsafe {
            let index = mm256_setr_epi64x(6, 10, 4, 8);
            let x = mm256_i64gather_ps(p, index, 4);
            assert_eq!(x.as_f32x4().as_array(), [6.0, 10.0, 4.0, 8.0])
        }
    }

    #[test]
    fn test_gather_f64() {
        let mut a = [0.0f64; 128];
        for i in 0..128 {
            a[i] = i as f64
        }

        let p = &a as *const [f64; 128] as *const f64;

        unsafe {
            let index = mm_setr_epi32(6, 10, 4, 8);
            let x = mm_i32gather_pd(p, index, 8);
            assert_eq!(x.as_f64x2().as_array(), [6.0, 10.0])
        };
        unsafe {
            let index = mm_setr_epi32(6, 10, 4, 8);
            let x = mm256_i32gather_pd(p, index, 8);
            assert_eq!(x.as_f64x4().as_array(), [6.0, 10.0, 4.0, 8.0])
        };
        unsafe {
            let index = mm_setr_epi64x(6, 10);
            let x = mm_i64gather_pd(p, index, 8);
            assert_eq!(x.as_f64x2().as_array(), [6.0, 10.0])
        };
        unsafe {
            let index = mm256_setr_epi64x(6, 10, 4, 8);
            let x = mm256_i64gather_pd(p, index, 8);
            assert_eq!(x.as_f64x4().as_array(), [6.0, 10.0, 4.0, 8.0])
        }
    }

    #[test]
    fn test_convert() {
        assert_eq!(mm256_cvtepi16_epi32(seq16_128()).as_i32x8().as_array(),
                   [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(mm256_cvtepi16_epi64(seq16_128()).as_i64x4().as_array(),
                   [1, 2, 3, 4]);
        assert_eq!(mm256_cvtepi32_epi64(seq32_128()).as_i64x4().as_array(),
                   [1, 2, 3, 4]);
        assert_eq!(mm256_cvtepi8_epi16(seq8_128()).as_i16x16().as_array(),
                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        assert_eq!(mm256_cvtepi8_epi32(seq8_128()).as_i32x8().as_array(),
                   [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(mm256_cvtepi8_epi64(seq8_128()).as_i64x4().as_array(),
                   [1, 2, 3, 4]);

        assert_eq!(mm256_cvtepu16_epi32(mseq16_128()).as_i32x8().as_array(),
                   [0xFFFF, 0xFFFE, 0xFFFD, 0xFFFC, 0xFFFB, 0xFFFA, 0xFFF9, 0xFFF8]);
        assert_eq!(mm256_cvtepu16_epi64(mseq16_128()).as_i64x4().as_array(),
                   [0xFFFF, 0xFFFE, 0xFFFD, 0xFFFC]);
        assert_eq!(mm256_cvtepu8_epi16(mseq8_128()).as_i16x16().as_array(),
                   [0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9, 0xF8, 0xF7, 0xF6, 0xF5, 0xF4, 0xF3, 0xF2, 0xF1, 0xF0]);
        assert_eq!(mm256_cvtepu8_epi32(mseq8_128()).as_i32x8().as_array(),
                   [0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9, 0xF8]);
        assert_eq!(mm256_cvtepu8_epi64(mseq8_128()).as_i64x4().as_array(),
                   [0xFF, 0xFE, 0xFD, 0xFC]);
    }

    #[test]
    fn test_palignr() {
        let a = mm256_set1_epi8(1);
        let b = mm256_set1_epi8(2);

        assert_eq!(mm256_alignr_epi8(a, b, 1).as_i8x32().as_array(),
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]);
        assert_eq!(mm256_alignr_epi8(a, b, 31).as_i8x32().as_array(),
                   [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

        for i in 0..33 {
            let c = mm256_alignr_epi8(a, b, i).as_i8x32().as_array();
            for j in 0..16 {
                if i + j >= 32 {
                    assert_eq!(c[j as usize], 0);
                    assert_eq!(c[(j + 16) as usize], 0);
                } else if i + j >= 16 {
                    assert_eq!(c[j as usize], 2);
                    assert_eq!(c[(j + 16) as usize], 2);
                } else {
                    assert_eq!(c[j as usize], 1);
                    assert_eq!(c[(j + 16) as usize], 1);
                }
            }
        }
    }

}
