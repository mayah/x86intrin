use super::*;
use super::{simd_add, simd_div,
            simd_and, simd_xor,
            simd_eq, simd_ge, simd_gt, simd_lt, simd_le, simd_ne,
            simd_shuffle4};

extern {
    // See http://x86.renejeschke.de/html/file_module_x86_id_37.html
    #[link_name = "llvm.x86.sse.cmp.ps"]
    pub fn sse_cmp_ps(a: m128, b: m128, c: i8) -> m128;
    #[link_name = "llvm.x86.sse.cmp.ss"]
    pub fn sse_cmp_ss(a: m128, b: m128, c: i8) -> m128;

    #[link_name = "llvm.x86.sse.comieq.ss"]
    pub fn sse_comieq_ss(a: m128, b: m128) -> i32;
    #[link_name = "llvm.x86.sse.comilt.ss"]
    pub fn sse_comilt_ss(a: m128, b: m128) -> i32;
    #[link_name = "llvm.x86.sse.comile.ss"]
    pub fn sse_comile_ss(a: m128, b: m128) -> i32;
    #[link_name = "llvm.x86.sse.comigt.ss"]
    pub fn sse_comigt_ss(a: m128, b: m128) -> i32;
    #[link_name = "llvm.x86.sse.comige.ss"]
    pub fn sse_comige_ss(a: m128, b: m128) -> i32;
    #[link_name = "llvm.x86.sse.comineq.ss"]
    pub fn sse_comineq_ss(a: m128, b: m128) -> i32;
    #[link_name = "llvm.x86.sse.ucomieq.ss"]
    pub fn sse_ucomieq_ss(a: m128, b: m128) -> i32;
    #[link_name = "llvm.x86.sse.ucomilt.ss"]
    pub fn sse_ucomilt_ss(a: m128, b: m128) -> i32;
    #[link_name = "llvm.x86.sse.ucomile.ss"]
    pub fn sse_ucomile_ss(a: m128, b: m128) -> i32;
    #[link_name = "llvm.x86.sse.ucomigt.ss"]
    pub fn sse_ucomigt_ss(a: m128, b: m128) -> i32;
    #[link_name = "llvm.x86.sse.ucomige.ss"]
    pub fn sse_ucomige_ss(a: m128, b: m128) -> i32;
    #[link_name = "llvm.x86.sse.ucomineq.ss"]
    pub fn sse_ucomineq_ss(a: m128, b: m128) -> i32;

    #[link_name = "llvm.x86.sse.cvtss2si"]
    pub fn sse_cvtss2si(a: m128) -> i32;
    #[link_name = "llvm.x86.sse.cvttss2si"]
    pub fn sse_cvttss2si(a: m128) -> i32;
    #[link_name = "llvm.x86.sse.cvtss2si64"]
    pub fn sse_cvtss2si64(a: m128) -> i64;
    #[link_name = "llvm.x86.sse.cvttss2si64"]
    pub fn sse_cvttss2si64(a: m128) -> i64;
    #[link_name = "llvm.x86.sse.cvtsi2ss"]
    pub fn sse_cvtsi2ss(a: m128, b: i32) -> m128;
    #[link_name = "llvm.x86.sse.cvtsi642ss"]
    pub fn sse_cvtsi642ss(a: m128, b: i64) -> m128;
}

// addps
// __m128 _mm_add_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_add_ps(a: m128, b: m128) -> m128 {
    unsafe { simd_add(a, b) }
}

// addss
// __m128 _mm_add_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_add_ss(a: m128, b: m128) -> m128 {
    a.as_f32x4().insert(0, (a.as_f32x4().extract(0) + b.as_f32x4().extract(0))).as_m128()
}

// andps
// __m128 _mm_and_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_and_ps(a: m128, b: m128) -> m128 {
    let ai = a.as_m128i();
    let bi = b.as_m128i();
    unsafe { simd_and(ai, bi).as_m128() }
}

// andnps
// __m128 _mm_andnot_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_andnot_ps(a: m128, b: m128) -> m128 {
    let ones = i32x4(!0, !0, !0, !0).as_m128i();
    let ai = a.as_m128i();
    let bi = b.as_m128i();
    unsafe { simd_and(simd_xor(ai, ones), bi).as_m128() }
}

// pavgw
// __m64 _mm_avg_pu16 (__m64 a, __m64 b)
// pavgb
// __m64 _mm_avg_pu8 (__m64 a, __m64 b)

// cmpps
// __m128 _mm_cmpeq_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_cmpeq_ps(a: m128, b: m128) -> m128 {
    let x: m128i = unsafe { simd_eq(a, b) };
    x.as_m128()
}

// cmpss
// __m128 _mm_cmpeq_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_cmpeq_ss(a: m128, b: m128) -> m128 {
    unsafe { sse_cmp_ss(a, b, 0) }
}

// cmpps
// __m128 _mm_cmpge_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_cmpge_ps(a: m128, b: m128) -> m128 {
    let x: m128i = unsafe { simd_ge(a, b) };
    x.as_m128()
}

// cmpss
// __m128 _mm_cmpge_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_cmpge_ss(a: m128, b: m128) -> m128 {
    unsafe {
        simd_shuffle4(a, sse_cmp_ss(b, a, 2), [4, 1, 2, 3])
    }
}

// cmpps
// __m128 _mm_cmpgt_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_cmpgt_ps(a: m128, b: m128) -> m128 {
    let x: m128i = unsafe { simd_gt(a, b) };
    x.as_m128()
}

// cmpss
// __m128 _mm_cmpgt_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_cmpgt_ss(a: m128, b: m128) -> m128 {
    unsafe {
        simd_shuffle4(a, sse_cmp_ss(b, a, 1), [4, 1, 2, 3])
    }
}

// cmpps
// __m128 _mm_cmple_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_cmple_ps(a: m128, b: m128) -> m128 {
    let x: m128i = unsafe { simd_le(a, b) };
    x.as_m128()
}

// cmpss
// __m128 _mm_cmple_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_cmple_ss(a: m128, b: m128) -> m128 {
    unsafe { sse_cmp_ss(a, b, 2) }
}

// cmpps
// __m128 _mm_cmplt_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_cmplt_ps(a: m128, b: m128) -> m128 {
    let x: m128i = unsafe { simd_lt(a, b) };
    x.as_m128()
}

// cmpss
// __m128 _mm_cmplt_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_cmplt_ss(a: m128, b: m128) -> m128 {
    unsafe { sse_cmp_ss(a, b, 1) }
}

// cmpps
// __m128 _mm_cmpneq_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_cmpneq_ps(a: m128, b: m128) -> m128 {
    let x: m128i = unsafe { simd_ne(a, b) };
    x.as_m128()
}

// cmpss
// __m128 _mm_cmpneq_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_cmpneq_ss(a: m128, b: m128) -> m128 {
    unsafe { sse_cmp_ss(a, b, 4) }
}

// cmpps
// __m128 _mm_cmpnge_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_cmpnge_ps(a: m128, b: m128) -> m128 {
    unsafe { sse_cmp_ps(b, a, 6) }
}

// cmpss
// __m128 _mm_cmpnge_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_cmpnge_ss(a: m128, b: m128) -> m128 {
    unsafe {
        simd_shuffle4(a, sse_cmp_ss(b, a, 6), [4, 1, 2, 3])
    }
}

// cmpps
// __m128 _mm_cmpngt_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_cmpngt_ps(a: m128, b: m128) -> m128 {
    unsafe { sse_cmp_ps(b, a, 5) }
}

// cmpss
// __m128 _mm_cmpngt_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_cmpngt_ss(a: m128, b: m128) -> m128 {
    unsafe {
        simd_shuffle4(a, sse_cmp_ss(b, a, 5), [4, 1, 2, 3])
    }
}

// cmpps
// __m128 _mm_cmpnle_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_cmpnle_ps(a: m128, b: m128) -> m128 {
    unsafe { sse_cmp_ps(a, b, 6) }
}

// cmpss
// __m128 _mm_cmpnle_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_cmpnle_ss(a: m128, b: m128) -> m128 {
    unsafe { sse_cmp_ss(a, b, 6) }
}

// cmpps
// __m128 _mm_cmpnlt_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_cmpnlt_ps(a: m128, b: m128) -> m128 {
    unsafe { sse_cmp_ps(a, b, 5) }
}

// cmpss
// __m128 _mm_cmpnlt_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_cmpnlt_ss(a: m128, b: m128) -> m128 {
    unsafe { sse_cmp_ss(a, b, 5) }
}

// cmpps
// __m128 _mm_cmpord_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_cmpord_ps(a: m128, b: m128) -> m128 {
    unsafe { sse_cmp_ps(a, b, 7) }
}

// cmpss
// __m128 _mm_cmpord_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_cmpord_ss(a: m128, b: m128) -> m128 {
    unsafe { sse_cmp_ss(a, b, 7) }
}

// cmpps
// __m128 _mm_cmpunord_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_cmpunord_ps(a: m128, b: m128) -> m128 {
    unsafe { sse_cmp_ps(a, b, 3) }
}

// cmpss
// __m128 _mm_cmpunord_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_cmpunord_ss(a: m128, b: m128) -> m128 {
    unsafe { sse_cmp_ss(a, b, 3) }
}

// comiss
// int _mm_comieq_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_comieq_ss(a: m128, b: m128) -> i32 {
    unsafe { sse_comieq_ss(a, b) }
}

// comiss
// int _mm_comige_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_comige_ss(a: m128, b: m128) -> i32 {
    unsafe { sse_comige_ss(a, b) }
}

// comiss
// int _mm_comigt_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_comigt_ss(a: m128, b: m128) -> i32 {
    unsafe { sse_comigt_ss(a, b) }
}

// comiss
// int _mm_comile_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_comile_ss(a: m128, b: m128) -> i32 {
    unsafe { sse_comile_ss(a, b) }
}

// comiss
// int _mm_comilt_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_comilt_ss(a: m128, b: m128) -> i32 {
    unsafe { sse_comilt_ss(a, b) }
}

// comiss
// int _mm_comineq_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_comineq_ss(a: m128, b: m128) -> i32 {
    unsafe { sse_comineq_ss(a, b) }
}

// cvtpi2ps
// __m128 _mm_cvt_pi2ps (__m128 a, __m64 b)
// cvtps2pi
// __m64 _mm_cvt_ps2pi (__m128 a)

// cvtsi2ss
// __m128 _mm_cvt_si2ss (__m128 a, int b)
#[inline]
pub fn mm_cvt_si2ss(a: m128, b: i32) -> m128 {
    unsafe { sse_cvtsi2ss(a, b) }
}

// cvtss2si
// int _mm_cvt_ss2si (__m128 a)
#[inline]
pub fn mm_cvt_ss2si(a: m128) -> i32 {
    mm_cvtss_si32(a)
}

// ...
// __m128 _mm_cvtpi16_ps (__m64 a)
// cvtpi2ps
// __m128 _mm_cvtpi32_ps (__m128 a, __m64 b)
// ...
// __m128 _mm_cvtpi32x2_ps (__m64 a, __m64 b)
// ...
// __m128 _mm_cvtpi8_ps (__m64 a)
// ...
// __m64 _mm_cvtps_pi16 (__m128 a)
// cvtps2pi
// __m64 _mm_cvtps_pi32 (__m128 a)
// ...
// __m64 _mm_cvtps_pi8 (__m128 a)
// ...
// __m128 _mm_cvtpu16_ps (__m64 a)
// ...
// __m128 _mm_cvtpu8_ps (__m64 a)

// cvtsi2ss
// __m128 _mm_cvtsi32_ss (__m128 a, int b)
#[inline]
pub fn mm_cvtsi32_ss(a: m128, b: i32) -> m128 {
    unsafe { sse_cvtsi2ss(a, b) }
}

// cvtsi2ss
// __m128 _mm_cvtsi64_ss (__m128 a, __int64 b)
#[inline]
pub fn mm_cvtsi64_ss(a: m128, b: i64) -> m128 {
    unsafe { sse_cvtsi642ss(a, b) }
}

// movss
// float _mm_cvtss_f32 (__m128 a)
#[inline]
pub fn mm_cvtss_f32(a: m128) -> f32 {
    a.as_f32x4().extract(0)
}

// cvtss2si
// int _mm_cvtss_si32 (__m128 a)
#[inline]
pub fn mm_cvtss_si32(a: m128) -> i32 {
    unsafe { sse_cvtss2si(a) }
}

// cvtss2si
// __int64 _mm_cvtss_si64 (__m128 a)
#[inline]
pub fn mm_cvtss_si64(a: m128) -> i64 {
    unsafe { sse_cvtss2si64(a) }
}

// cvttps2pi
// __m64 _mm_cvtt_ps2pi (__m128 a)

// cvttss2si
// int _mm_cvtt_ss2si (__m128 a)
#[inline]
pub fn mm_cvtt_ss2si(a: m128) -> i32 {
    mm_cvttss_si32(a)
}

// cvttps2pi
// __m64 _mm_cvttps_pi32 (__m128 a)

// cvttss2si
// int _mm_cvttss_si32 (__m128 a)
#[inline]
pub fn mm_cvttss_si32(a: m128) -> i32 {
    unsafe { sse_cvttss2si(a) }
}

// cvttss2si
// __int64 _mm_cvttss_si64 (__m128 a)
#[inline]
pub fn mm_cvttss_si64(a: m128) -> i64 {
    unsafe { sse_cvttss2si64(a) }
}

// divps
// __m128 _mm_div_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_div_ps(a: m128, b: m128) -> m128 {
    unsafe { simd_div(a, b) }
}

// divss
// __m128 _mm_div_ss (__m128 a, __m128 b)
// pextrw
// int _mm_extract_pi16 (__m64 a, int imm8)
// unsigned int _MM_GET_EXCEPTION_MASK ()
// unsigned int _MM_GET_EXCEPTION_STATE ()
// unsigned int _MM_GET_FLUSH_ZERO_MODE ()
// unsigned int _MM_GET_ROUNDING_MODE ()
// stmxcsr
// unsigned int _mm_getcsr (void)
// pinsrw
// __m64 _mm_insert_pi16 (__m64 a, int i, int imm8)
// movaps
// __m128 _mm_load_ps (float const* mem_addr)
// ...
// __m128 _mm_load_ps1 (float const* mem_addr)
// movss
// __m128 _mm_load_ss (float const* mem_addr)
// ...
// __m128 _mm_load1_ps (float const* mem_addr)
// movhps
// __m128 _mm_loadh_pi (__m128 a, __m64 const* mem_addr)
// movlps
// __m128 _mm_loadl_pi (__m128 a, __m64 const* mem_addr)
// ...
// __m128 _mm_loadr_ps (float const* mem_addr)
// movups
// __m128 _mm_loadu_ps (float const* mem_addr)
// maskmovq
// void _mm_maskmove_si64 (__m64 a, __m64 mask, char* mem_addr)
// maskmovq
// void _m_maskmovq (__m64 a, __m64 mask, char* mem_addr)
// pmaxsw
// __m64 _mm_max_pi16 (__m64 a, __m64 b)
// maxps
// __m128 _mm_max_ps (__m128 a, __m128 b)
// pmaxub
// __m64 _mm_max_pu8 (__m64 a, __m64 b)
// maxss
// __m128 _mm_max_ss (__m128 a, __m128 b)
// pminsw
// __m64 _mm_min_pi16 (__m64 a, __m64 b)
// minps
// __m128 _mm_min_ps (__m128 a, __m128 b)
// pminub
// __m64 _mm_min_pu8 (__m64 a, __m64 b)
// minss
// __m128 _mm_min_ss (__m128 a, __m128 b)
// movss
// __m128 _mm_move_ss (__m128 a, __m128 b)
// movhlps
// __m128 _mm_movehl_ps (__m128 a, __m128 b)
// movlhps
// __m128 _mm_movelh_ps (__m128 a, __m128 b)
// pmovmskb
// int _mm_movemask_pi8 (__m64 a)
// movmskps
// int _mm_movemask_ps (__m128 a)
// mulps
// __m128 _mm_mul_ps (__m128 a, __m128 b)
// mulss
// __m128 _mm_mul_ss (__m128 a, __m128 b)
// pmulhuw
// __m64 _mm_mulhi_pu16 (__m64 a, __m64 b)
// orps
// __m128 _mm_or_ps (__m128 a, __m128 b)
// pavgb
// __m64 _m_pavgb (__m64 a, __m64 b)
// pavgw
// __m64 _m_pavgw (__m64 a, __m64 b)
// pextrw
// int _m_pextrw (__m64 a, int imm8)
// pinsrw
// __m64 _m_pinsrw (__m64 a, int i, int imm8)
// pmaxsw
// __m64 _m_pmaxsw (__m64 a, __m64 b)
// pmaxub
// __m64 _m_pmaxub (__m64 a, __m64 b)
// pminsw
// __m64 _m_pminsw (__m64 a, __m64 b)
// pminub
// __m64 _m_pminub (__m64 a, __m64 b)
// pmovmskb
// int _m_pmovmskb (__m64 a)
// pmulhuw
// __m64 _m_pmulhuw (__m64 a, __m64 b)
// prefetchnta, prefetcht0, prefetcht1, prefetcht2
// void _mm_prefetch (char const* p, int i)
// psadbw
// __m64 _m_psadbw (__m64 a, __m64 b)
// pshufw
// __m64 _m_pshufw (__m64 a, int imm8)
// rcpps
// __m128 _mm_rcp_ps (__m128 a)
// rcpss
// __m128 _mm_rcp_ss (__m128 a)
// rsqrtps
// __m128 _mm_rsqrt_ps (__m128 a)
// rsqrtss
// __m128 _mm_rsqrt_ss (__m128 a)
// psadbw
// __m64 _mm_sad_pu8 (__m64 a, __m64 b)
// void _MM_SET_EXCEPTION_MASK (unsigned int a)
// void _MM_SET_EXCEPTION_STATE (unsigned int a)
// void _MM_SET_FLUSH_ZERO_MODE (unsigned int a)

// ...
// __m128 _mm_set_ps (float e3, float e2, float e1, float e0)
#[inline]
pub fn mm_set_ps(e3: f32, e2: f32, e1: f32, e0: f32) -> m128 {
    m128(e0, e1, e2, e3)
}

// ...
// __m128 _mm_set_ps1 (float a)
#[inline]
pub fn mm_set_ps1(a: f32) -> m128 {
    m128(a, a, a, a)
}

// void _MM_SET_ROUNDING_MODE (unsigned int a)

// ...
// __m128 _mm_set_ss (float a)
#[inline]
pub fn mm_set_ss(a: f32) -> m128 {
    m128(a, 0.0, 0.0, 0.0)
}

// ...
// __m128 _mm_set1_ps (float a)
#[inline]
pub fn mm_set1_ps(a: f32) -> m128 {
    m128(a, a, a, a)
}

// ldmxcsr
// void _mm_setcsr (unsigned int a)

// ...
// __m128 _mm_setr_ps (float e3, float e2, float e1, float e0)
#[inline]
pub fn mm_setr_ps(e0: f32, e1: f32, e2: f32, e3: f32) -> m128 {
    m128(e0, e1, e2, e3)
}

// xorps
// __m128 _mm_setzero_ps (void)
#[inline]
pub fn mm_setzero_ps() -> m128 {
    m128(0.0, 0.0, 0.0, 0.0)
}

// sfence
// void _mm_sfence (void)
// pshufw
// __m64 _mm_shuffle_pi16 (__m64 a, int imm8)
// shufps
// __m128 _mm_shuffle_ps (__m128 a, __m128 b, unsigned int imm8)
// sqrtps
// __m128 _mm_sqrt_ps (__m128 a)
// sqrtss
// __m128 _mm_sqrt_ss (__m128 a)
// movaps
// void _mm_store_ps (float* mem_addr, __m128 a)
// ...
// void _mm_store_ps1 (float* mem_addr, __m128 a)
// movss
// void _mm_store_ss (float* mem_addr, __m128 a)
// ...
// void _mm_store1_ps (float* mem_addr, __m128 a)
// movhps
// void _mm_storeh_pi (__m64* mem_addr, __m128 a)
// movlps
// void _mm_storel_pi (__m64* mem_addr, __m128 a)
// ...
// void _mm_storer_ps (float* mem_addr, __m128 a)
// movups
// void _mm_storeu_ps (float* mem_addr, __m128 a)
// movntq
// void _mm_stream_pi (__m64* mem_addr, __m64 a)
// movntps
// void _mm_stream_ps (float* mem_addr, __m128 a)
// subps
// __m128 _mm_sub_ps (__m128 a, __m128 b)
// subss
// __m128 _mm_sub_ss (__m128 a, __m128 b)
// ...
// _MM_TRANSPOSE4_PS (__m128 row0, __m128 row1, __m128 row2, __m128 row3)

// ucomiss
// int _mm_ucomieq_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_ucomieq_ss(a: m128, b: m128) -> i32 {
    unsafe { sse_ucomieq_ss(a, b) }
}

// ucomiss
// int _mm_ucomige_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_ucomige_ss(a: m128, b: m128) -> i32 {
    unsafe { sse_ucomige_ss(a, b) }
}

// ucomiss
// int _mm_ucomigt_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_ucomigt_ss(a: m128, b: m128) -> i32 {
    unsafe { sse_ucomigt_ss(a, b) }
}

// ucomiss
// int _mm_ucomile_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_ucomile_ss(a: m128, b: m128) -> i32 {
    unsafe { sse_ucomile_ss(a, b) }
}

// ucomiss
// int _mm_ucomilt_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_ucomilt_ss(a: m128, b: m128) -> i32 {
    unsafe { sse_ucomilt_ss(a, b) }
}

// ucomiss
// int _mm_ucomineq_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_ucomineq_ss(a: m128, b: m128) -> i32 {
    unsafe { sse_ucomineq_ss(a, b) }
}

// unpckhps
// __m128 _mm_unpackhi_ps (__m128 a, __m128 b)
// unpcklps
// __m128 _mm_unpacklo_ps (__m128 a, __m128 b)
// xorps
// __m128 _mm_xor_ps (__m128 a, __m128 b)

#[cfg(test)]
mod tests {
    use std;
    use super::super::*;

    #[test]
    fn test_mm_add_ps() {
        let x = mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let y = mm_add_ps(x, x).as_f32x4();

        assert_eq!(y.as_array(), [2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_mm_add_ss() {
        let x = mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let y = mm_add_ss(x, x).as_f32x4();

        assert_eq!(y.as_array(), [2.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_mm_and_ps() {
        let x = i32x4(0x1, 0x2, 0x3, 0x4).as_m128();
        let y = i32x4(0x3, 0x4, 0x5, 0x6).as_m128();

        let z1 = mm_and_ps(x, y).as_m128i().as_i32x4();
        let z2 = mm_andnot_ps(x, y).as_m128i().as_i32x4();

        assert_eq!(z1.as_array(), [0x1 & 0x3, 0x2 & 0x4, 0x3 & 0x5, 0x4 & 0x6]);
        assert_eq!(z2.as_array(), [!0x1 & 0x3, !0x2 & 0x4, !0x3 & 0x5, !0x4 & 0x6]);
    }

    #[test]
    fn test_mm_set() {
        let x1 = mm_set_ps(1.0, 2.0, 3.0, 4.0).as_f32x4();
        let x2 = mm_set_ps1(5.0).as_f32x4();
        let x3 = mm_set_ss(6.0).as_f32x4();
        let x4 = mm_set1_ps(7.0).as_f32x4();
        let x5 = mm_setr_ps(1.0, 2.0, 3.0, 4.0).as_f32x4();
        let x6 = mm_setzero_ps().as_f32x4();

        assert_eq!(x1.as_array(), [4.0, 3.0, 2.0, 1.0]);
        assert_eq!(x2.as_array(), [5.0, 5.0, 5.0, 5.0]);
        assert_eq!(x3.as_array(), [6.0, 0.0, 0.0, 0.0]);
        assert_eq!(x4.as_array(), [7.0, 7.0, 7.0, 7.0]);
        assert_eq!(x5.as_array(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(x6.as_array(), [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_mm_cmp_ps() {
        let x = mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let y = mm_setr_ps(2.0, 2.0, 2.0, 2.0);
        let z = mm_setr_ps(std::f32::NAN, std::f32::NAN, std::f32::NAN, std::f32::NAN);

        let xy_eq = mm_cmpeq_ps(x, y).as_m128i().as_i32x4();
        let xy_ge = mm_cmpge_ps(x, y).as_m128i().as_i32x4();
        let xy_gt = mm_cmpgt_ps(x, y).as_m128i().as_i32x4();
        let xy_le = mm_cmple_ps(x, y).as_m128i().as_i32x4();
        let xy_lt = mm_cmplt_ps(x, y).as_m128i().as_i32x4();
        let xy_ne = mm_cmpneq_ps(x, y).as_m128i().as_i32x4();
        let xy_nge = mm_cmpnge_ps(x, y).as_m128i().as_i32x4();
        let xy_ngt = mm_cmpngt_ps(x, y).as_m128i().as_i32x4();
        let xy_nle = mm_cmpnle_ps(x, y).as_m128i().as_i32x4();
        let xy_nlt = mm_cmpnlt_ps(x, y).as_m128i().as_i32x4();
        let xy_ord = mm_cmpord_ps(x, y).as_m128i().as_i32x4();
        let xy_uno = mm_cmpunord_ps(x, y).as_m128i().as_i32x4();

        assert_eq!(xy_eq.as_array(), [ 0, !0,  0,  0]);
        assert_eq!(xy_ge.as_array(), [ 0, !0, !0, !0]);
        assert_eq!(xy_gt.as_array(), [ 0,  0, !0, !0]);
        assert_eq!(xy_le.as_array(), [!0, !0,  0,  0]);
        assert_eq!(xy_lt.as_array(), [!0,  0,  0,  0]);
        assert_eq!(xy_ne.as_array(), [!0,  0, !0, !0]);
        assert_eq!(xy_nge.as_array(), [!0,  0,  0,  0]);
        assert_eq!(xy_ngt.as_array(), [!0, !0,  0,  0]);
        assert_eq!(xy_nle.as_array(), [ 0,  0, !0, !0]);
        assert_eq!(xy_nlt.as_array(), [ 0, !0, !0, !0]);
        assert_eq!(xy_ord.as_array(), [!0, !0, !0, !0]);
        assert_eq!(xy_uno.as_array(), [ 0,  0,  0,  0]);

        let xz_eq = mm_cmpeq_ps(x, z).as_m128i().as_i32x4();
        let xz_ge = mm_cmpge_ps(x, z).as_m128i().as_i32x4();
        let xz_gt = mm_cmpgt_ps(x, z).as_m128i().as_i32x4();
        let xz_le = mm_cmple_ps(x, z).as_m128i().as_i32x4();
        let xz_lt = mm_cmplt_ps(x, z).as_m128i().as_i32x4();
        let xz_ne = mm_cmpneq_ps(x, z).as_m128i().as_i32x4();
        let xz_nge = mm_cmpnge_ps(x, z).as_m128i().as_i32x4();
        let xz_ngt = mm_cmpngt_ps(x, z).as_m128i().as_i32x4();
        let xz_nle = mm_cmpnle_ps(x, z).as_m128i().as_i32x4();
        let xz_nlt = mm_cmpnlt_ps(x, z).as_m128i().as_i32x4();
        let xz_ord = mm_cmpord_ps(x, z).as_m128i().as_i32x4();
        let xz_uno = mm_cmpunord_ps(x, z).as_m128i().as_i32x4();

        assert_eq!(xz_eq.as_array(), [ 0,  0,  0,  0]);
        assert_eq!(xz_ge.as_array(), [ 0,  0,  0,  0]);
        assert_eq!(xz_gt.as_array(), [ 0,  0,  0,  0]);
        assert_eq!(xz_le.as_array(), [ 0,  0,  0,  0]);
        assert_eq!(xz_lt.as_array(), [ 0,  0,  0,  0]);
        assert_eq!(xz_ne.as_array(), [!0, !0, !0, !0]);
        assert_eq!(xz_nge.as_array(), [!0, !0, !0, !0]);
        assert_eq!(xz_ngt.as_array(), [!0, !0, !0, !0]);
        assert_eq!(xz_nle.as_array(), [!0, !0, !0, !0]);
        assert_eq!(xz_nlt.as_array(), [!0, !0, !0, !0]);
        assert_eq!(xz_ord.as_array(), [ 0,  0,  0,  0]);
        assert_eq!(xz_uno.as_array(), [!0, !0, !0, !0]);
    }

    #[test]
    fn test_mm_cmp_ss() {
        let x = mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let y = mm_setr_ps(2.0, 2.0, 2.0, 2.0);
        let z = mm_setr_ps(std::f32::NAN, std::f32::NAN, std::f32::NAN, std::f32::NAN);

        let xy_eq = mm_cmpeq_ss(x, y);
        let xy_ge = mm_cmpge_ss(x, y);
        let xy_gt = mm_cmpgt_ss(x, y);
        let xy_le = mm_cmple_ss(x, y);
        let xy_lt = mm_cmplt_ss(x, y);
        let xy_ne = mm_cmpneq_ss(x, y);
        let xy_nge = mm_cmpnge_ss(x, y);
        let xy_ngt = mm_cmpngt_ss(x, y);
        let xy_nle = mm_cmpnle_ss(x, y);
        let xy_nlt = mm_cmpnlt_ss(x, y);
        let xy_ord = mm_cmpord_ss(x, y);
        let xy_uno = mm_cmpunord_ss(x, y);

        assert_eq!(xy_eq.as_m128i().as_i32x4().extract(0), 0);
        assert_eq!(xy_ge.as_m128i().as_i32x4().extract(0), 0);
        assert_eq!(xy_gt.as_m128i().as_i32x4().extract(0), 0);
        assert_eq!(xy_le.as_m128i().as_i32x4().extract(0), !0);
        assert_eq!(xy_lt.as_m128i().as_i32x4().extract(0), !0);
        assert_eq!(xy_ne.as_m128i().as_i32x4().extract(0), !0);
        assert_eq!(xy_nge.as_m128i().as_i32x4().extract(0), !0);
        assert_eq!(xy_ngt.as_m128i().as_i32x4().extract(0), !0);
        assert_eq!(xy_nle.as_m128i().as_i32x4().extract(0), 0);
        assert_eq!(xy_nlt.as_m128i().as_i32x4().extract(0), 0);
        assert_eq!(xy_ord.as_m128i().as_i32x4().extract(0), !0);
        assert_eq!(xy_uno.as_m128i().as_i32x4().extract(0), 0);

        for i in 1 .. 4 {
            assert_eq!(xy_eq.as_f32x4().extract(i), x.as_f32x4().extract(i));
            assert_eq!(xy_ge.as_f32x4().extract(i), x.as_f32x4().extract(i), "i={}", i);
            assert_eq!(xy_gt.as_f32x4().extract(i), x.as_f32x4().extract(i), "i={}", i);
            assert_eq!(xy_le.as_f32x4().extract(i), x.as_f32x4().extract(i));
            assert_eq!(xy_lt.as_f32x4().extract(i), x.as_f32x4().extract(i));
            assert_eq!(xy_ne.as_f32x4().extract(i), x.as_f32x4().extract(i));
            assert_eq!(xy_nge.as_f32x4().extract(i), x.as_f32x4().extract(i), "i={}", i);
            assert_eq!(xy_ngt.as_f32x4().extract(i), x.as_f32x4().extract(i));
            assert_eq!(xy_nle.as_f32x4().extract(i), x.as_f32x4().extract(i));
            assert_eq!(xy_nlt.as_f32x4().extract(i), x.as_f32x4().extract(i));
            assert_eq!(xy_ord.as_f32x4().extract(i), x.as_f32x4().extract(i));
            assert_eq!(xy_uno.as_f32x4().extract(i), x.as_f32x4().extract(i));
        }

        let xz_eq = mm_cmpeq_ss(x, z);
        let xz_ge = mm_cmpge_ss(x, z);
        let xz_gt = mm_cmpgt_ss(x, z);
        let xz_le = mm_cmple_ss(x, z);
        let xz_lt = mm_cmplt_ss(x, z);
        let xz_ne = mm_cmpneq_ss(x, z);
        let xz_nge = mm_cmpnge_ss(x, z);
        let xz_ngt = mm_cmpngt_ss(x, z);
        let xz_nle = mm_cmpnle_ss(x, z);
        let xz_nlt = mm_cmpnlt_ss(x, z);
        let xz_ord = mm_cmpord_ss(x, z);
        let xz_uno = mm_cmpunord_ss(x, z);

        assert_eq!(xz_eq.as_m128i().as_i32x4().extract(0), 0);
        assert_eq!(xz_ge.as_m128i().as_i32x4().extract(0), 0);
        assert_eq!(xz_gt.as_m128i().as_i32x4().extract(0), 0);
        assert_eq!(xz_le.as_m128i().as_i32x4().extract(0), 0);
        assert_eq!(xz_lt.as_m128i().as_i32x4().extract(0), 0);
        assert_eq!(xz_ne.as_m128i().as_i32x4().extract(0), !0);
        assert_eq!(xz_nge.as_m128i().as_i32x4().extract(0), !0);
        assert_eq!(xz_ngt.as_m128i().as_i32x4().extract(0), !0);
        assert_eq!(xz_nle.as_m128i().as_i32x4().extract(0), !0);
        assert_eq!(xz_nlt.as_m128i().as_i32x4().extract(0), !0);
        assert_eq!(xz_ord.as_m128i().as_i32x4().extract(0), 0);
        assert_eq!(xz_uno.as_m128i().as_i32x4().extract(0), !0);
    }

    #[test]
    fn test_mm_comi_ss() {
        let x = mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let y = mm_setr_ps(2.0, 2.0, 2.0, 2.0);

        assert_eq!(mm_comieq_ss(x, x), 1);
        assert_eq!(mm_comige_ss(x, x), 1);
        assert_eq!(mm_comigt_ss(x, x), 0);
        assert_eq!(mm_comile_ss(x, x), 1);
        assert_eq!(mm_comilt_ss(x, x), 0);
        assert_eq!(mm_comineq_ss(x, x), 0);

        assert_eq!(mm_ucomieq_ss(x, x), 1);
        assert_eq!(mm_ucomige_ss(x, x), 1);
        assert_eq!(mm_ucomigt_ss(x, x), 0);
        assert_eq!(mm_ucomile_ss(x, x), 1);
        assert_eq!(mm_ucomilt_ss(x, x), 0);
        assert_eq!(mm_ucomineq_ss(x, x), 0);

        assert_eq!(mm_comieq_ss(x, y), 0);
        assert_eq!(mm_comige_ss(x, y), 0);
        assert_eq!(mm_comigt_ss(x, y), 0);
        assert_eq!(mm_comile_ss(x, y), 1);
        assert_eq!(mm_comilt_ss(x, y), 1);
        assert_eq!(mm_comineq_ss(x, y), 1);

        assert_eq!(mm_ucomieq_ss(x, y), 0);
        assert_eq!(mm_ucomige_ss(x, y), 0);
        assert_eq!(mm_ucomigt_ss(x, y), 0);
        assert_eq!(mm_ucomile_ss(x, y), 1);
        assert_eq!(mm_ucomilt_ss(x, y), 1);
        assert_eq!(mm_ucomineq_ss(x, y), 1);

        assert_eq!(mm_comieq_ss(y, x), 0);
        assert_eq!(mm_comige_ss(y, x), 1);
        assert_eq!(mm_comigt_ss(y, x), 1);
        assert_eq!(mm_comile_ss(y, x), 0);
        assert_eq!(mm_comilt_ss(y, x), 0);
        assert_eq!(mm_comineq_ss(y, x), 1);

        assert_eq!(mm_ucomieq_ss(y, x), 0);
        assert_eq!(mm_ucomige_ss(y, x), 1);
        assert_eq!(mm_ucomigt_ss(y, x), 1);
        assert_eq!(mm_ucomile_ss(y, x), 0);
        assert_eq!(mm_ucomilt_ss(y, x), 0);
        assert_eq!(mm_ucomineq_ss(y, x), 1);

        // TODO(mayah): Hmm, hitting this behavior change?
        // https://llvm.org/bugs/show_bug.cgi?id=28510
        //
        // let z = mm_setr_ps(std::f32::NAN, std::f32::NAN, std::f32::NAN, std::f32::NAN);
        // assert_eq!(mm_comieq_ss(x, z), 1);
        // assert_eq!(mm_comige_ss(x, z), 1);
        // assert_eq!(mm_comigt_ss(x, z), 1);
        // assert_eq!(mm_comile_ss(x, z), 1);
        // assert_eq!(mm_comilt_ss(x, z), 1);
        // assert_eq!(mm_comineq_ss(x, z), 1);
        //
        // assert_eq!(mm_ucomieq_ss(x, z), 1);
        // assert_eq!(mm_ucomige_ss(x, z), 1);
        // assert_eq!(mm_ucomigt_ss(x, z), 1);
        // assert_eq!(mm_ucomile_ss(x, z), 1);
        // assert_eq!(mm_ucomilt_ss(x, z), 1);
        // assert_eq!(mm_ucomineq_ss(x, z), 1);
    }

    #[test]
    fn test_cvt() {
        let x = mm_setr_ps(1.7, 2.0, 3.0, 4.0);

        assert_eq!(mm_cvt_si2ss(x, 5).as_f32x4().as_array(), [5.0, 2.0, 3.0, 4.0]);
        assert_eq!(mm_cvt_ss2si(x), 2);
        assert_eq!(mm_cvtsi32_ss(x, 6).as_f32x4().as_array(), [6.0, 2.0, 3.0, 4.0]);
        assert_eq!(mm_cvtsi64_ss(x, 7).as_f32x4().as_array(), [7.0, 2.0, 3.0, 4.0]);
        assert_eq!(mm_cvtss_f32(x), 1.7);
        assert_eq!(mm_cvtss_si32(x), 2);
        assert_eq!(mm_cvtss_si64(x), 2);
        // for cvtt, truncate will be used.
        assert_eq!(mm_cvtt_ss2si(x), 1);
        assert_eq!(mm_cvttss_si32(x), 1);
        assert_eq!(mm_cvttss_si64(x), 1);
    }

    #[test]
    fn test_div() {
        let x = mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let y = mm_setr_ps(2.0, 2.0, 2.0, 2.0);
        let z = mm_div_ps(x, y);

        assert_eq!(z.as_f32x4().as_array(), [0.5, 1.0, 1.5, 2.0]);
    }
}
