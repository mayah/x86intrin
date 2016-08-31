use super::*;
use super::{simd_and, simd_or, simd_xor};

extern "platform-intrinsic" {
    fn x86_mm256_abs_epi8(x: i8x32) -> i8x32;
    fn x86_mm256_abs_epi16(x: i16x16) -> i16x16;
    fn x86_mm256_abs_epi32(x: i32x8) -> i32x8;
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
// vpaddd
// __m256i _mm256_add_epi32 (__m256i a, __m256i b)
// vpaddq
// __m256i _mm256_add_epi64 (__m256i a, __m256i b)
// vpaddb
// __m256i _mm256_add_epi8 (__m256i a, __m256i b)
// vpaddsw
// __m256i _mm256_adds_epi16 (__m256i a, __m256i b)
// vpaddsb
// __m256i _mm256_adds_epi8 (__m256i a, __m256i b)
// vpaddusw
// __m256i _mm256_adds_epu16 (__m256i a, __m256i b)
// vpaddusb
// __m256i _mm256_adds_epu8 (__m256i a, __m256i b)
// vpalignr
// __m256i _mm256_alignr_epi8 (__m256i a, __m256i b, const int count)

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
// vpavgb
// __m256i _mm256_avg_epu8 (__m256i a, __m256i b)
// vpblendw
// __m256i _mm256_blend_epi16 (__m256i a, __m256i b, const int imm8)
// vpblendd
// __m128i _mm_blend_epi32 (__m128i a, __m128i b, const int imm8)
// vpblendd
// __m256i _mm256_blend_epi32 (__m256i a, __m256i b, const int imm8)
// vpblendvb
// __m256i _mm256_blendv_epi8 (__m256i a, __m256i b, __m256i mask)
// vpbroadcastb
// __m128i _mm_broadcastb_epi8 (__m128i a)
// vpbroadcastb
// __m256i _mm256_broadcastb_epi8 (__m128i a)
// vpbroadcastd
// __m128i _mm_broadcastd_epi32 (__m128i a)
// vpbroadcastd
// __m256i _mm256_broadcastd_epi32 (__m128i a)
// vpbroadcastq
// __m128i _mm_broadcastq_epi64 (__m128i a)
// vpbroadcastq
// __m256i _mm256_broadcastq_epi64 (__m128i a)
// movddup
// __m128d _mm_broadcastsd_pd (__m128d a)
// vbroadcastsd
// __m256d _mm256_broadcastsd_pd (__m128d a)
// vbroadcasti128
// __m256i _mm256_broadcastsi128_si256 (__m128i a)
// vbroadcastss
// __m128 _mm_broadcastss_ps (__m128 a)
// vbroadcastss
// __m256 _mm256_broadcastss_ps (__m128 a)
// vpbroadcastw
// __m128i _mm_broadcastw_epi16 (__m128i a)
// vpbroadcastw
// __m256i _mm256_broadcastw_epi16 (__m128i a)
// vpslldq
// __m256i _mm256_bslli_epi128 (__m256i a, const int imm8)
// vpsrldq
// __m256i _mm256_bsrli_epi128 (__m256i a, const int imm8)
// vpcmpeqw
// __m256i _mm256_cmpeq_epi16 (__m256i a, __m256i b)
// vpcmpeqd
// __m256i _mm256_cmpeq_epi32 (__m256i a, __m256i b)
// vpcmpeqq
// __m256i _mm256_cmpeq_epi64 (__m256i a, __m256i b)
// vpcmpeqb
// __m256i _mm256_cmpeq_epi8 (__m256i a, __m256i b)
// vpcmpgtw
// __m256i _mm256_cmpgt_epi16 (__m256i a, __m256i b)
// vpcmpgtd
// __m256i _mm256_cmpgt_epi32 (__m256i a, __m256i b)
// vpcmpgtq
// __m256i _mm256_cmpgt_epi64 (__m256i a, __m256i b)
// vpcmpgtb
// __m256i _mm256_cmpgt_epi8 (__m256i a, __m256i b)
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
// vphaddw
// __m256i _mm256_hadd_epi16 (__m256i a, __m256i b)
// vphaddd
// __m256i _mm256_hadd_epi32 (__m256i a, __m256i b)
// vphaddsw
// __m256i _mm256_hadds_epi16 (__m256i a, __m256i b)
// vphsubw
// __m256i _mm256_hsub_epi16 (__m256i a, __m256i b)
// vphsubd
// __m256i _mm256_hsub_epi32 (__m256i a, __m256i b)
// vphsubsw
// __m256i _mm256_hsubs_epi16 (__m256i a, __m256i b)
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
// vpmaddwd
// __m256i _mm256_madd_epi16 (__m256i a, __m256i b)
// vpmaddubsw
// __m256i _mm256_maddubs_epi16 (__m256i a, __m256i b)
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
// vpmaxsd
// __m256i _mm256_max_epi32 (__m256i a, __m256i b)
// vpmaxsb
// __m256i _mm256_max_epi8 (__m256i a, __m256i b)
// vpmaxuw
// __m256i _mm256_max_epu16 (__m256i a, __m256i b)
// vpmaxud
// __m256i _mm256_max_epu32 (__m256i a, __m256i b)
// vpmaxub
// __m256i _mm256_max_epu8 (__m256i a, __m256i b)
// vpminsw
// __m256i _mm256_min_epi16 (__m256i a, __m256i b)
// vpminsd
// __m256i _mm256_min_epi32 (__m256i a, __m256i b)
// vpminsb
// __m256i _mm256_min_epi8 (__m256i a, __m256i b)
// vpminuw
// __m256i _mm256_min_epu16 (__m256i a, __m256i b)
// vpminud
// __m256i _mm256_min_epu32 (__m256i a, __m256i b)
// vpminub
// __m256i _mm256_min_epu8 (__m256i a, __m256i b)
// vpmovmskb
// int _mm256_movemask_epi8 (__m256i a)
// vmpsadbw
// __m256i _mm256_mpsadbw_epu8 (__m256i a, __m256i b, const int imm8)
// vpmuldq
// __m256i _mm256_mul_epi32 (__m256i a, __m256i b)
// vpmuludq
// __m256i _mm256_mul_epu32 (__m256i a, __m256i b)
// vpmulhw
// __m256i _mm256_mulhi_epi16 (__m256i a, __m256i b)
// vpmulhuw
// __m256i _mm256_mulhi_epu16 (__m256i a, __m256i b)
// vpmulhrsw
// __m256i _mm256_mulhrs_epi16 (__m256i a, __m256i b)
// vpmullw
// __m256i _mm256_mullo_epi16 (__m256i a, __m256i b)
// vpmulld
// __m256i _mm256_mullo_epi32 (__m256i a, __m256i b)

// vpor
// __m256i _mm256_or_si256 (__m256i a, __m256i b)
pub fn mm256_or_si256(a: m256i, b: m256i) -> m256i {
    unsafe { simd_or(a, b) }
}

// vpacksswb
// __m256i _mm256_packs_epi16 (__m256i a, __m256i b)
// vpackssdw
// __m256i _mm256_packs_epi32 (__m256i a, __m256i b)
// vpackuswb
// __m256i _mm256_packus_epi16 (__m256i a, __m256i b)
// vpackusdw
// __m256i _mm256_packus_epi32 (__m256i a, __m256i b)
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
// vpunpckhdq
// __m256i _mm256_unpackhi_epi32 (__m256i a, __m256i b)
// vpunpckhqdq
// __m256i _mm256_unpackhi_epi64 (__m256i a, __m256i b)
// vpunpckhbw
// __m256i _mm256_unpackhi_epi8 (__m256i a, __m256i b)
// vpunpcklwd
// __m256i _mm256_unpacklo_epi16 (__m256i a, __m256i b)
// vpunpckldq
// __m256i _mm256_unpacklo_epi32 (__m256i a, __m256i b)
// vpunpcklqdq
// __m256i _mm256_unpacklo_epi64 (__m256i a, __m256i b)
// vpunpcklbw
// __m256i _mm256_unpacklo_epi8 (__m256i a, __m256i b)

// vpxor
// __m256i _mm256_xor_si256 (__m256i a, __m256i b)
pub fn mm256_xor_si256(a: m256i, b: m256i) -> m256i {
    unsafe { simd_xor(a, b) }
}

#[cfg(test)]
mod tests {
    use super::super::*;

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
}
