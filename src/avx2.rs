use super::*;
use super::{simd_add, simd_mul,
            simd_and, simd_or, simd_xor,
            simd_eq, simd_gt,
            simd_shuffle2, simd_shuffle4, simd_shuffle8, simd_shuffle16, simd_shuffle32};

extern "platform-intrinsic" {
    fn x86_mm256_abs_epi8(x: i8x32) -> i8x32;
    fn x86_mm256_abs_epi16(x: i16x16) -> i16x16;
    fn x86_mm256_abs_epi32(x: i32x8) -> i32x8;

    fn x86_mm256_adds_epi8(x: i8x32, y: i8x32) -> i8x32;
    fn x86_mm256_adds_epu8(x: u8x32, y: u8x32) -> u8x32;
    fn x86_mm256_adds_epi16(x: i16x16, y: i16x16) -> i16x16;
    fn x86_mm256_adds_epu16(x: u16x16, y: u16x16) -> u16x16;

    fn x86_mm256_hadd_epi16(x: i16x16, y: i16x16) -> i16x16;
    fn x86_mm256_hadd_epi32(x: i32x8, y: i32x8) -> i32x8;
    fn x86_mm256_hadds_epi16(x: i16x16, y: i16x16) -> i16x16;
    fn x86_mm256_hsub_epi16(x: i16x16, y: i16x16) -> i16x16;
    fn x86_mm256_hsub_epi32(x: i32x8, y: i32x8) -> i32x8;
    fn x86_mm256_hsubs_epi16(x: i16x16, y: i16x16) -> i16x16;

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

    fn x86_mm256_avg_epu8(x: u8x32, y: u8x32) -> u8x32;
    fn x86_mm256_avg_epu16(x: u16x16, y: u16x16) -> u16x16;
}

extern {
    #[link_name = "llvm.x86.avx2.pblendvb"]
    fn avx2_pblendvb(a: i8x32, b: i8x32, c: i8x32) -> i8x32;

    #[link_name = "llvm.x86.avx2.pmadd.wd"]
    fn avx2_pmadd_wd(a: i16x16, b: i16x16) -> i32x8;
    #[link_name = "llvm.x86.avx2.pmadd.ub.sw"]
    fn avx2_pmadd_ub_sw(a: i8x32, b: i8x32) -> i16x16;

    #[link_name = "llvm.x86.avx2.psll.dq"]
    fn avx2_psll_dq(a: i64x4, b: i32) -> i64x4;
    #[link_name = "llvm.x86.avx2.psrl.dq"]
    fn avx2_psrl_dq(a: i64x4, b: i32) -> i64x4;
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
#[allow(unused_variables)]
pub fn mm256_alignr_epi8(a: m256i, b: m256i, count: i32) -> m256i {
    unimplemented!()
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

// TODO(mayah): rust does not have avx2 cvt functions yet.
// vpmovsxwd
// __m256i _mm256_cvtepi16_epi32 (__m128i a)
// vpmovsxwq
// __m256i _mm256_cvtepi16_epi64 (__m128i a)
// vpmovsxdq
// __m256i _mm256_cvtepi32_epi64 (__m128i a)
// vpmovsxbw
// __m256i _mm256_cvtepi8_epi16 (__m128i a)
// vpmovsxbd
// __m256i _mm256_cvtepi8_epi32 (__m128i a)
// vpmovsxbq
// __m256i _mm256_cvtepi8_epi64 (__m128i a)
// vpmovzxwd
// __m256i _mm256_cvtepu16_epi32 (__m128i a)
// vpmovzxwq
// __m256i _mm256_cvtepu16_epi64 (__m128i a)
// vpmovzxdq
// __m256i _mm256_cvtepu32_epi64 (__m128i a)
// vpmovzxbw
// __m256i _mm256_cvtepu8_epi16 (__m128i a)
// vpmovzxbd
// __m256i _mm256_cvtepu8_epi32 (__m128i a)
// vpmovzxbq
// __m256i _mm256_cvtepu8_epi64 (__m128i a)

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

// vpgatherdd
// __m128i _mm_i32gather_epi32 (int const* base_addr, __m128i vindex, const int scale)
// vpgatherdd
// __m128i _mm_mask_i32gather_epi32 (__m128i src, int const* base_addr, __m128i vindex, __m128i mask, const int scale)
// vpgatherdd
// __m256i _mm256_i32gather_epi32 (int const* base_addr, __m256i vindex, const int scale)
// vpgatherdd
// __m256i _mm256_mask_i32gather_epi32 (__m256i src, int const* base_addr, __m256i vindex, __m256i mask, const int scale)
// vpgatherdq
// __m128i _mm_i32gather_epi64 (__int64 const* base_addr, __m128i vindex, const int scale)
// vpgatherdq
// __m128i _mm_mask_i32gather_epi64 (__m128i src, __int64 const* base_addr, __m128i vindex, __m128i mask, const int scale)
// vpgatherdq
// __m256i _mm256_i32gather_epi64 (__int64 const* base_addr, __m128i vindex, const int scale)
// vpgatherdq
// __m256i _mm256_mask_i32gather_epi64 (__m256i src, __int64 const* base_addr, __m128i vindex, __m256i mask, const int scale)
// vgatherdpd
// __m128d _mm_i32gather_pd (double const* base_addr, __m128i vindex, const int scale)
// vgatherdpd
// __m128d _mm_mask_i32gather_pd (__m128d src, double const* base_addr, __m128i vindex, __m128d mask, const int scale)
// vgatherdpd
// __m256d _mm256_i32gather_pd (double const* base_addr, __m128i vindex, const int scale)
// vgatherdpd
// __m256d _mm256_mask_i32gather_pd (__m256d src, double const* base_addr, __m128i vindex, __m256d mask, const int scale)
// vgatherdps
// __m128 _mm_i32gather_ps (float const* base_addr, __m128i vindex, const int scale)
// vgatherdps
// __m128 _mm_mask_i32gather_ps (__m128 src, float const* base_addr, __m128i vindex, __m128 mask, const int scale)
// vgatherdps
// __m256 _mm256_i32gather_ps (float const* base_addr, __m256i vindex, const int scale)
// vgatherdps
// __m256 _mm256_mask_i32gather_ps (__m256 src, float const* base_addr, __m256i vindex, __m256 mask, const int scale)
// vpgatherqd
// __m128i _mm_i64gather_epi32 (int const* base_addr, __m128i vindex, const int scale)
// vpgatherqd
// __m128i _mm_mask_i64gather_epi32 (__m128i src, int const* base_addr, __m128i vindex, __m128i mask, const int scale)
// vpgatherqd
// __m128i _mm256_i64gather_epi32 (int const* base_addr, __m256i vindex, const int scale)
// vpgatherqd
// __m128i _mm256_mask_i64gather_epi32 (__m128i src, int const* base_addr, __m256i vindex, __m128i mask, const int scale)
// vpgatherqq
// __m128i _mm_i64gather_epi64 (__int64 const* base_addr, __m128i vindex, const int scale)
// vpgatherqq
// __m128i _mm_mask_i64gather_epi64 (__m128i src, __int64 const* base_addr, __m128i vindex, __m128i mask, const int scale)
// vpgatherqq
// __m256i _mm256_i64gather_epi64 (__int64 const* base_addr, __m256i vindex, const int scale)
// vpgatherqq
// __m256i _mm256_mask_i64gather_epi64 (__m256i src, __int64 const* base_addr, __m256i vindex, __m256i mask, const int scale)
// vgatherqpd
// __m128d _mm_i64gather_pd (double const* base_addr, __m128i vindex, const int scale)
// vgatherqpd
// __m128d _mm_mask_i64gather_pd (__m128d src, double const* base_addr, __m128i vindex, __m128d mask, const int scale)
// vgatherqpd
// __m256d _mm256_i64gather_pd (double const* base_addr, __m256i vindex, const int scale)
// vgatherqpd
// __m256d _mm256_mask_i64gather_pd (__m256d src, double const* base_addr, __m256i vindex, __m256d mask, const int scale)
// vgatherqps
// __m128 _mm_i64gather_ps (float const* base_addr, __m128i vindex, const int scale)
// vgatherqps
// __m128 _mm_mask_i64gather_ps (__m128 src, float const* base_addr, __m128i vindex, __m128 mask, const int scale)
// vgatherqps
// __m128 _mm256_i64gather_ps (float const* base_addr, __m256i vindex, const int scale)
// vgatherqps
// __m128 _mm256_mask_i64gather_ps (__m128 src, float const* base_addr, __m256i vindex, __m128 mask, const int scale)

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
// vpmaskmovd
// __m256i _mm256_maskload_epi32 (int const* mem_addr, __m256i mask)
// vpmaskmovq
// __m128i _mm_maskload_epi64 (__int64 const* mem_addr, __m128i mask)
// vpmaskmovq
// __m256i _mm256_maskload_epi64 (__int64 const* mem_addr, __m256i mask)
// vpmaskmovd
// void _mm_maskstore_epi32 (int* mem_addr, __m128i mask, __m128i a)
// vpmaskmovd
// void _mm256_maskstore_epi32 (int* mem_addr, __m256i mask, __m256i a)
// vpmaskmovq
// void _mm_maskstore_epi64 (__int64* mem_addr, __m128i mask, __m128i a)
// vpmaskmovq
// void _mm256_maskstore_epi64 (__int64* mem_addr, __m256i mask, __m256i a)

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
#[allow(unused_variables)]
pub fn mm256_mul_epi32(a: m256i, b: m256i) -> m256i {
    // TODO(mayah): rustc uses `mm256_mul_epi64`, which is the same as mm256_mul_epi32 (in intel).
    // unsafe { x86_mm256_mul_epi64(a.as_i32x8(), b.as_i32x8()).as_m256i() }

    // Also, when we use mm256_mul_epi64, rust says undefined reference to `llvm.x86.avx2.pmulq.dq'
    unimplemented!()
}

// vpmuludq
// __m256i _mm256_mul_epu32 (__m256i a, __m256i b)
#[inline]
#[allow(unused_variables)]
pub fn mm256_mul_epu32(a: m256i, b: m256i) -> m256i {
    // TODO(mayah): rustc uses `mm256_mul_epu64`, which is the same as mm256_mul_epu32 (in intel).
    // unsafe { x86_mm256_mul_epu64(a.as_u32x8(), b.as_u32x8()).as_m256i() }

    // Also, when we use mm256_mul_epu64, rust says undefined reference to `llvm.x86.avx2.pmulq.dq'
    unimplemented!()
}

// vpmulhw
// __m256i _mm256_mulhi_epi16 (__m256i a, __m256i b)
#[inline]
#[allow(unused_variables)]
pub fn mm256_mulhi_epi16(a: m256i, b: m256i) -> m256i {
    // rustc complains "undefined reference to `llvm.x86.avx2.pmulhw.w'".
    // unsafe { x86_mm256_mulhi_epi16(a.as_i16x16(), b.as_i16x16()).as_m256i() }
    unimplemented!()
}

// vpmulhuw
// __m256i _mm256_mulhi_epu16 (__m256i a, __m256i b)
#[inline]
#[allow(unused_variables)]
pub fn mm256_mulhi_epu16(a: m256i, b: m256i) -> m256i {
    // rustc complains "undefined reference to `llvm.x86.avx2.pmulhw.w'".
    // unsafe { x86_mm256_mulhi_epu16(a.as_u16x16(), b.as_u16x16()).as_m256i() }
    unimplemented!()
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
// vpermq
// __m256i _mm256_permute4x64_epi64 (__m256i a, const int imm8)
// vpermpd
// __m256d _mm256_permute4x64_pd (__m256d a, const int imm8)
// vpermd
// __m256i _mm256_permutevar8x32_epi32 (__m256i a, __m256i idx)
// vpermps
// __m256 _mm256_permutevar8x32_ps (__m256 a, __m256i idx)
// vpsadbw
// __m256i _mm256_sad_epu8 (__m256i a, __m256i b)
// vpshufd
// __m256i _mm256_shuffle_epi32 (__m256i a, const int imm8)
// vpshufb
// __m256i _mm256_shuffle_epi8 (__m256i a, __m256i b)
// vpshufhw
// __m256i _mm256_shufflehi_epi16 (__m256i a, const int imm8)
// vpshuflw
// __m256i _mm256_shufflelo_epi16 (__m256i a, const int imm8)
// vpsignw
// __m256i _mm256_sign_epi16 (__m256i a, __m256i b)
// vpsignd
// __m256i _mm256_sign_epi32 (__m256i a, __m256i b)
// vpsignb
// __m256i _mm256_sign_epi8 (__m256i a, __m256i b)
// vpsllw
// __m256i _mm256_sll_epi16 (__m256i a, __m128i count)
// vpslld
// __m256i _mm256_sll_epi32 (__m256i a, __m128i count)
// vpsllq
// __m256i _mm256_sll_epi64 (__m256i a, __m128i count)
// vpsllw
// __m256i _mm256_slli_epi16 (__m256i a, int imm8)
// vpslld
// __m256i _mm256_slli_epi32 (__m256i a, int imm8)
// vpsllq
// __m256i _mm256_slli_epi64 (__m256i a, int imm8)

// vpslldq
// __m256i _mm256_slli_si256 (__m256i a, const int imm8)
#[inline]
pub fn mm256_slli_si256(a: m256i, imm8: i32) -> m256i {
    fn_imm8_arg1!(avx2_psll_dq, a.as_i64x4(), imm8 * 8).as_m256i()
}

// vpsllvd
// __m128i _mm_sllv_epi32 (__m128i a, __m128i count)
// vpsllvd
// __m256i _mm256_sllv_epi32 (__m256i a, __m256i count)
// vpsllvq
// __m128i _mm_sllv_epi64 (__m128i a, __m128i count)
// vpsllvq
// __m256i _mm256_sllv_epi64 (__m256i a, __m256i count)
// vpsraw
// __m256i _mm256_sra_epi16 (__m256i a, __m128i count)
// vpsrad
// __m256i _mm256_sra_epi32 (__m256i a, __m128i count)
// vpsraw
// __m256i _mm256_srai_epi16 (__m256i a, int imm8)
// vpsrad
// __m256i _mm256_srai_epi32 (__m256i a, int imm8)
// vpsravd
// __m128i _mm_srav_epi32 (__m128i a, __m128i count)
// vpsravd
// __m256i _mm256_srav_epi32 (__m256i a, __m256i count)
// vpsrlw
// __m256i _mm256_srl_epi16 (__m256i a, __m128i count)
// vpsrld
// __m256i _mm256_srl_epi32 (__m256i a, __m128i count)
// vpsrlq
// __m256i _mm256_srl_epi64 (__m256i a, __m128i count)
// vpsrlw
// __m256i _mm256_srli_epi16 (__m256i a, int imm8)
// vpsrld
// __m256i _mm256_srli_epi32 (__m256i a, int imm8)
// vpsrlq
// __m256i _mm256_srli_epi64 (__m256i a, int imm8)

// vpsrldq
// __m256i _mm256_srli_si256 (__m256i a, const int imm8)
#[inline]
pub fn mm256_srli_si256(a: m256i, imm8: i32) -> m256i {
    fn_imm8_arg1!(avx2_psrl_dq, a.as_i64x4(), imm8 * 8).as_m256i()
}

// vpsrlvd
// __m128i _mm_srlv_epi32 (__m128i a, __m128i count)
// vpsrlvd
// __m256i _mm256_srlv_epi32 (__m256i a, __m256i count)
// vpsrlvq
// __m128i _mm_srlv_epi64 (__m128i a, __m128i count)
// vpsrlvq
// __m256i _mm256_srlv_epi64 (__m256i a, __m256i count)
// vmovntdqa
// __m256i _mm256_stream_load_si256 (__m256i const* mem_addr)
// vpsubw
// __m256i _mm256_sub_epi16 (__m256i a, __m256i b)
// vpsubd
// __m256i _mm256_sub_epi32 (__m256i a, __m256i b)
// vpsubq
// __m256i _mm256_sub_epi64 (__m256i a, __m256i b)
// vpsubb
// __m256i _mm256_sub_epi8 (__m256i a, __m256i b)
// vpsubsw
// __m256i _mm256_subs_epi16 (__m256i a, __m256i b)
// vpsubsb
// __m256i _mm256_subs_epi8 (__m256i a, __m256i b)
// vpsubusw
// __m256i _mm256_subs_epu16 (__m256i a, __m256i b)
// vpsubusb
// __m256i _mm256_subs_epu8 (__m256i a, __m256i b)

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

    fn seq16() -> m256i {
        mm256_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    }
    fn seq16_128() -> m128i {
        mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8)
    }
    fn mseq16() -> m256i {
        mm256_setr_epi16(-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16)
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

    fn seqps_128() -> m128 { mm_setr_ps(1.0, 2.0, 3.0, 4.0) }
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
        assert_eq!(mm256_slli_si256(seq8(), 3).as_i8x32().as_array(),
                   [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                    0, 0, 0, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]);
        assert_eq!(mm256_srli_si256(seq8(), 3).as_i8x32().as_array(),
                   [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0,
                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 0, 0, 0]);
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
        //assert_eq!(mm256_mul_epi32(seq32(), mseq32()).as_i64x4().as_array(),
        //           [-1, -9, -25, -49]);
        //assert_eq!(mm256_mul_epu32(seq32(), mseq32()).as_u64x4().as_array(),
        //           [1 * (-1i64 as u64), 3 * (-3i64 as u64), 5 * (-5i64 as u64), 7 * (-7i64 as u64)]);

        let x = mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7);
        let y = mm256_setr_epi16(0, 1, 2, 3, -4, -5, -6, -7, 0, 1, 2, 3, -4, -5, -6, -7);

        //assert_eq!(mm256_mulhi_epi16(x, y).as_i16x16().as_array(),
        //           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        //assert_eq!(mm256_mulhi_epu16(x, y).as_u16x16().as_array(),
        //           [0, 0, 0, 0, !0, !0, !0, !0, 0, 0, 0, 0, !0, !0, !0, !0]);

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
}
