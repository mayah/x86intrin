use std;
use super::*;
use super::{simd_add, simd_sub, simd_mul, simd_div,
            simd_and, simd_or, simd_xor,
            simd_eq, simd_ge, simd_gt, simd_lt, simd_le, simd_ne,
            simd_shuffle4};

extern "platform-intrinsic" {
    fn x86_mm_max_ps(a: m128, b: m128) -> m128;
    fn x86_mm_min_ps(a: m128, b: m128) -> m128;

    fn x86_mm_rcp_ps(x: m128) -> m128;
    fn x86_mm_rsqrt_ps(x: m128) -> m128;
    fn x86_mm_sqrt_ps(x: m128) -> m128;
}

extern {
    // See http://x86.renejeschke.de/html/file_module_x86_id_37.html
    #[link_name = "llvm.x86.sse.cmp.ps"]
    fn sse_cmp_ps(a: m128, b: m128, c: i8) -> m128;
    #[link_name = "llvm.x86.sse.cmp.ss"]
    fn sse_cmp_ss(a: m128, b: m128, c: i8) -> m128;

    #[link_name = "llvm.x86.sse.comieq.ss"]
    fn sse_comieq_ss(a: m128, b: m128) -> i32;
    #[link_name = "llvm.x86.sse.comilt.ss"]
    fn sse_comilt_ss(a: m128, b: m128) -> i32;
    #[link_name = "llvm.x86.sse.comile.ss"]
    fn sse_comile_ss(a: m128, b: m128) -> i32;
    #[link_name = "llvm.x86.sse.comigt.ss"]
    fn sse_comigt_ss(a: m128, b: m128) -> i32;
    #[link_name = "llvm.x86.sse.comige.ss"]
    fn sse_comige_ss(a: m128, b: m128) -> i32;
    #[link_name = "llvm.x86.sse.comineq.ss"]
    fn sse_comineq_ss(a: m128, b: m128) -> i32;
    #[link_name = "llvm.x86.sse.ucomieq.ss"]
    fn sse_ucomieq_ss(a: m128, b: m128) -> i32;
    #[link_name = "llvm.x86.sse.ucomilt.ss"]
    fn sse_ucomilt_ss(a: m128, b: m128) -> i32;
    #[link_name = "llvm.x86.sse.ucomile.ss"]
    fn sse_ucomile_ss(a: m128, b: m128) -> i32;
    #[link_name = "llvm.x86.sse.ucomigt.ss"]
    fn sse_ucomigt_ss(a: m128, b: m128) -> i32;
    #[link_name = "llvm.x86.sse.ucomige.ss"]
    fn sse_ucomige_ss(a: m128, b: m128) -> i32;
    #[link_name = "llvm.x86.sse.ucomineq.ss"]
    fn sse_ucomineq_ss(a: m128, b: m128) -> i32;

    #[link_name = "llvm.x86.sse.cvtss2si"]
    fn sse_cvtss2si(a: m128) -> i32;
    #[link_name = "llvm.x86.sse.cvttss2si"]
    fn sse_cvttss2si(a: m128) -> i32;
    #[link_name = "llvm.x86.sse.cvtss2si64"]
    fn sse_cvtss2si64(a: m128) -> i64;
    #[link_name = "llvm.x86.sse.cvttss2si64"]
    fn sse_cvttss2si64(a: m128) -> i64;
    #[link_name = "llvm.x86.sse.cvtsi2ss"]
    fn sse_cvtsi2ss(a: m128, b: i32) -> m128;
    #[link_name = "llvm.x86.sse.cvtsi642ss"]
    fn sse_cvtsi642ss(a: m128, b: i64) -> m128;

    #[link_name = "llvm.x86.sse.max.ss"]
    fn sse_max_ss(a: m128, b: m128) -> m128;
    #[link_name = "llvm.x86.sse.min.ss"]
    fn sse_min_ss(a: m128, b: m128) -> m128;

    #[link_name = "llvm.x86.sse.movmsk.ps"]
    fn sse_movmsk_ps(a: m128) -> i32;

    #[link_name = "llvm.x86.sse.rcp.ss"]
    fn sse_rcp_ss(a: m128) -> m128;
    #[link_name = "llvm.x86.sse.rsqrt.ss"]
    fn sse_rsqrt_ss(a: m128) -> m128;
    #[link_name = "llvm.x86.sse.sqrt.ss"]
    fn sse_sqrt_ss(a: m128) -> m128;

    #[link_name = "llvm.x86.sse.stmxcsr"]
    fn sse_stmxcsr(a: *mut i8) -> ();
    #[link_name = "llvm.x86.sse.ldmxcsr"]
    fn sse_ldmxcsr(a: *const i8) -> ();

    #[link_name = "llvm.x86.sse.storeu.ps"]
    fn sse_storeu_ps(a: *mut i8, b: m128) -> ();

    #[link_name = "llvm.x86.sse.sfence"]
    fn sse_sfence() -> ();
}

pub const MM_HINT_T0: i32 = 3;
pub const MM_HINT_T1: i32 = 2;
pub const MM_HINT_T2: i32 = 1;
pub const MM_HINT_NTA: i32 = 0;

// #define _MM_SHUFFLE(z, y, x, w) (((z) << 6) | ((y) << 4) | ((x) << 2) | (w))
#[inline]
pub fn mm_shuffle(z: u32, y: u32, x: u32, w: u32) -> u32 {
    (z << 6) | (y << 4) | (x << 2) | w
}

pub const MM_EXCEPT_INVALID:   u32 = 0x0001;
pub const MM_EXCEPT_DENORM:    u32 = 0x0002;
pub const MM_EXCEPT_DIV_ZERO:  u32 = 0x0004;
pub const MM_EXCEPT_OVERFLOW:  u32 = 0x0008;
pub const MM_EXCEPT_UNDERFLOW: u32 = 0x0010;
pub const MM_EXCEPT_INEXACT:   u32 = 0x0020;
pub const MM_EXCEPT_MASK:      u32 = 0x003f;

pub const MM_MASK_INVALID:   u32 = 0x0080;
pub const MM_MASK_DENORM:    u32 = 0x0100;
pub const MM_MASK_DIV_ZERO:  u32 = 0x0200;
pub const MM_MASK_OVERFLOW:  u32 = 0x0400;
pub const MM_MASK_UNDERFLOW: u32 = 0x0800;
pub const MM_MASK_INEXACT:   u32 = 0x1000;
pub const MM_MASK_MASK:      u32 = 0x1f80;

pub const MM_ROUND_NEAREST:     u32 = 0x0000;
pub const MM_ROUND_DOWN:        u32 = 0x2000;
pub const MM_ROUND_UP:          u32 = 0x4000;
pub const MM_ROUND_TOWARD_ZERO: u32 = 0x6000;
pub const MM_ROUND_MASK:        u32 = 0x6000;

pub const MM_FLUSH_ZERO_MASK: u32 = 0x8000;
pub const MM_FLUSH_ZERO_ON:   u32 = 0x8000;
pub const MM_FLUSH_ZERO_OFF:  u32 = 0x0000;

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

// cmpps
// __m128 _mm_cmpeq_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_cmpeq_ps(a: m128, b: m128) -> m128 {
    let x: i32x4 = unsafe { simd_eq(a, b) };
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
    let x: i32x4 = unsafe { simd_ge(a, b) };
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
    let x: i32x4 = unsafe { simd_gt(a, b) };
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
    let x: i32x4 = unsafe { simd_le(a, b) };
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
    let x: i32x4 = unsafe { simd_lt(a, b) };
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
    let x: i32x4 = unsafe { simd_ne(a, b) };
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

// cvttss2si
// int _mm_cvtt_ss2si (__m128 a)
#[inline]
pub fn mm_cvtt_ss2si(a: m128) -> i32 {
    mm_cvttss_si32(a)
}

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
#[inline]
pub fn mm_div_ss(a: m128, b: m128) -> m128 {
    a.as_f32x4().insert(0, (a.as_f32x4().extract(0) / b.as_f32x4().extract(0))).as_m128()
}

// unsigned int _MM_GET_EXCEPTION_MASK ()
#[inline]
pub fn mm_get_exception_mask() -> u32 {
    mm_getcsr() & MM_MASK_MASK
}

// unsigned int _MM_GET_EXCEPTION_STATE ()
#[inline]
pub fn mm_get_exception_state() -> u32 {
    mm_getcsr() & MM_EXCEPT_MASK
}

// unsigned int _MM_GET_FLUSH_ZERO_MODE ()
#[inline]
pub fn mm_get_flush_zero_mode() -> u32 {
    mm_getcsr() & MM_FLUSH_ZERO_MASK
}

// unsigned int _MM_GET_ROUNDING_MODE ()
#[inline]
pub fn mm_get_rounding_mode() -> u32 {
    mm_getcsr() & MM_ROUND_MASK
}

// stmxcsr
// unsigned int _mm_getcsr (void)
#[inline]
pub fn mm_getcsr() -> u32 {
    unsafe {
        let mut x: u32 = std::mem::uninitialized();
        sse_stmxcsr(&mut x as *mut u32 as *mut i8);
        x
    }
}

// movaps
// __m128 _mm_load_ps (float const* mem_addr)
#[inline]
pub unsafe fn mm_load_ps(mem_addr: *const f32) -> m128 {
    *(mem_addr as *const m128)
}

// ...
// __m128 _mm_load_ps1 (float const* mem_addr)
#[inline]
pub unsafe fn mm_load_ps1(mem_addr: *const f32) -> m128 {
    mm_load1_ps(mem_addr)
}

// movss
// __m128 _mm_load_ss (float const* mem_addr)
#[inline]
pub unsafe fn mm_load_ss(mem_addr: *const f32) -> m128 {
    // TODO(mayah): mem_addr might be unaligned?
    let v = *mem_addr;
    f32x4(v, 0.0, 0.0, 0.0).as_m128()
}

// ...
// __m128 _mm_load1_ps (float const* mem_addr)
#[inline]
pub unsafe fn mm_load1_ps(mem_addr: *const f32) -> m128 {
    // TODO(mayah): mem_addr might be unaligned?
    let v = *mem_addr;
    f32x4(v, v, v, v).as_m128()
}

// ...
// __m128 _mm_loadr_ps (float const* mem_addr)
#[inline]
pub unsafe fn mm_loadr_ps(mem_addr: *const f32) -> m128 {
    let a = mm_load_ps(mem_addr);
    simd_shuffle4(a, a, [3, 2, 1, 0])
}

// movups
// __m128 _mm_loadu_ps (float const* mem_addr)
#[inline]
pub unsafe fn mm_loadu_ps(mem_addr: *const f32) -> m128 {
    let mut result: m128 = std::mem::uninitialized();

    let src_p = mem_addr as *const u8;
    let dst_p = &mut result as *mut m128 as *mut u8;
    std::ptr::copy_nonoverlapping(src_p, dst_p, 16);

    result
}

// maxps
// __m128 _mm_max_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_max_ps(a: m128, b: m128) -> m128 {
    unsafe { x86_mm_max_ps(a, b) }
}

// maxss
// __m128 _mm_max_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_max_ss(a: m128, b: m128) -> m128 {
    unsafe { sse_max_ss(a, b) }
}

// minps
// __m128 _mm_min_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_min_ps(a: m128, b: m128) -> m128 {
    unsafe { x86_mm_min_ps(a, b) }
}

// minss
// __m128 _mm_min_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_min_ss(a: m128, b: m128) -> m128 {
    unsafe { sse_min_ss(a, b) }
}

// movss
// __m128 _mm_move_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_move_ss(a: m128, b: m128) -> m128 {
    unsafe { simd_shuffle4(a, b, [4, 1, 2, 3]) }
}

// movhlps
// __m128 _mm_movehl_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_movehl_ps(a: m128, b: m128) -> m128 {
    unsafe { simd_shuffle4(a, b, [6, 7, 2, 3]) }
}

// movlhps
// __m128 _mm_movelh_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_movelh_ps(a: m128, b: m128) -> m128 {
    unsafe { simd_shuffle4(a, b, [0, 1, 4, 5]) }
}

// movmskps
// int _mm_movemask_ps (__m128 a)
#[inline]
pub fn mm_movemask_ps(a: m128) -> i32 {
    unsafe { sse_movmsk_ps(a) }
}

// mulps
// __m128 _mm_mul_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_mul_ps(a: m128, b: m128) -> m128 {
    unsafe { simd_mul(a, b) }
}

// mulss
// __m128 _mm_mul_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_mul_ss(a: m128, b: m128) -> m128 {
    a.as_f32x4().insert(0, (a.as_f32x4().extract(0) * b.as_f32x4().extract(0))).as_m128()
}

// orps
// __m128 _mm_or_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_or_ps(a: m128, b: m128) -> m128 {
    let ai = a.as_m128i();
    let bi = b.as_m128i();
    unsafe { simd_or(ai, bi).as_m128() }
}

// rcpps
// __m128 _mm_rcp_ps (__m128 a)
#[inline]
pub fn mm_rcp_ps(a: m128) -> m128 {
    unsafe { x86_mm_rcp_ps(a) }
}

// rcpss
// __m128 _mm_rcp_ss (__m128 a)
#[inline]
pub fn mm_rcp_ss(a: m128) -> m128 {
    unsafe { sse_rcp_ss(a) }
}

// rsqrtps
// __m128 _mm_rsqrt_ps (__m128 a)
#[inline]
pub fn mm_rsqrt_ps(a: m128) -> m128 {
    unsafe { x86_mm_rsqrt_ps(a) }
}

// rsqrtss
// __m128 _mm_rsqrt_ss (__m128 a)
#[inline]
pub fn mm_rsqrt_ss(a: m128) -> m128 {
    unsafe { sse_rsqrt_ss(a) }
}

// void _MM_SET_EXCEPTION_MASK (unsigned int a)
#[inline]
pub fn mm_set_exception_mask(a: u32) {
    mm_setcsr((mm_getcsr() & !MM_MASK_MASK) | a)
}

// void _MM_SET_EXCEPTION_STATE (unsigned int a)
#[inline]
pub fn mm_set_exception_state(a: u32) {
    mm_setcsr((mm_getcsr() & !MM_EXCEPT_MASK) | a)
}

// void _MM_SET_FLUSH_ZERO_MODE (unsigned int a)
#[inline]
pub fn mm_set_flush_zero_mode(a: u32) {
    mm_setcsr((mm_getcsr() & !MM_FLUSH_ZERO_MASK) | a)
}

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
#[inline]
pub fn mm_set_rounding_mode(a: u32) {
    mm_setcsr((mm_getcsr() & !MM_ROUND_MASK) | a)
}

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
#[inline]
pub fn mm_setcsr(a: u32) {
    unsafe { sse_ldmxcsr(&a as *const u32 as *const i8) }
}

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
#[inline]
pub fn mm_sfense() {
    unsafe { sse_sfence() }
}

// shufps
// __m128 _mm_shuffle_ps (__m128 a, __m128 b, unsigned int imm8)
#[inline]
pub fn mm_shuffle_ps(a: m128, b: m128, imm8: u32) -> m128 {
    // Grr...

    macro_rules! shuffle4 {
        ($a: expr, $b: expr, $c: expr, $d: expr) => {
            unsafe { simd_shuffle4(a, b, [$a, $b, $c, $d]) }
        }
    }

    macro_rules! shuffle3 {
        ($a: expr, $b: expr, $c: expr) => {
            match (imm8 >> 6) & 0x3 {
                0 => shuffle4!($a, $b, $c, 0),
                1 => shuffle4!($a, $b, $c, 1),
                2 => shuffle4!($a, $b, $c, 2),
                3 => shuffle4!($a, $b, $c, 3),
                _ => unreachable!()
            }
        }
    }

    macro_rules! shuffle2 {
        ($a: expr, $b: expr) => {
            match (imm8 >> 4) & 0x3 {
                0 => shuffle3!($a, $b, 0),
                1 => shuffle3!($a, $b, 1),
                2 => shuffle3!($a, $b, 2),
                3 => shuffle3!($a, $b, 3),
                _ => unreachable!()
            }
        }
    }

    macro_rules! shuffle1 {
        ($a: expr) => {
            match (imm8 >> 2) & 0x3 {
                0 => shuffle2!($a, 0),
                1 => shuffle2!($a, 1),
                2 => shuffle2!($a, 2),
                3 => shuffle2!($a, 3),
                _ => unreachable!()
            }
        }
    }

    macro_rules! shuffle0 {
        () => {
            match (imm8 >> 0) & 0x3 {
                0 => shuffle1!(0),
                1 => shuffle1!(1),
                2 => shuffle1!(2),
                3 => shuffle1!(3),
                _ => unreachable!()
            }
        }
    }

    shuffle0!()
}

// sqrtps
// __m128 _mm_sqrt_ps (__m128 a)
#[inline]
pub fn mm_sqrt_ps(a: m128) -> m128 {
    unsafe { x86_mm_sqrt_ps(a) }
}

// sqrtss
// __m128 _mm_sqrt_ss (__m128 a)
#[inline]
pub fn mm_sqrt_ss(a: m128) -> m128 {
    unsafe { sse_sqrt_ss(a) }
}

// movaps
// void _mm_store_ps (float* mem_addr, __m128 a)
#[inline]
pub unsafe fn mm_store_ps(mem_addr: *mut f32, a: m128) {
    let p = mem_addr as *mut m128;
    *p = a
}

// ...
// void _mm_store_ps1 (float* mem_addr, __m128 a)
#[inline]
pub unsafe fn mm_store_ps1(mem_addr: *mut f32, a: m128) {
    mm_store1_ps(mem_addr, a)
}

// movss
// void _mm_store_ss (float* mem_addr, __m128 a)
#[inline]
pub unsafe fn mm_store_ss(mem_addr: *mut f32, a: m128) {
    *mem_addr = *(&a as *const m128 as *const f32);
}

// ...
// void _mm_store1_ps (float* mem_addr, __m128 a)
#[inline]
pub unsafe fn mm_store1_ps(mem_addr: *mut f32, a: m128) {
    let x = simd_shuffle4(a, a, [0, 0, 0, 0]);
    mm_store_ps(mem_addr, x)
}

// ...
// void _mm_storer_ps (float* mem_addr, __m128 a)
#[inline]
pub unsafe fn mm_storer_ps(mem_addr: *mut f32, a: m128) {
    let x = simd_shuffle4(a, a, [3, 2, 1, 0]);
    mm_store_ps(mem_addr, x)
}

// movups
// void _mm_storeu_ps (float* mem_addr, __m128 a)
#[inline]
pub unsafe fn mm_storeu_ps(mem_addr: *mut f32, a: m128) {
    sse_storeu_ps(mem_addr as *mut i8, a)
}

// movntps
// void _mm_stream_ps (float* mem_addr, __m128 a)
#[inline]
#[allow(unused_variables)]
pub unsafe fn mm_stream_ps(mem_addr: *mut f32, a: m128) {
    // TODO(mayah): No __builtin_ia32_movntps equivalent in rust?
    unimplemented!()
}

// subps
// __m128 _mm_sub_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_sub_ps(a: m128, b: m128) -> m128 {
    unsafe { simd_sub(a, b) }
}

// subss
// __m128 _mm_sub_ss (__m128 a, __m128 b)
#[inline]
pub fn mm_sub_ss(a: m128, b: m128) -> m128 {
    a.as_f32x4().insert(0, (a.as_f32x4().extract(0) - b.as_f32x4().extract(0))).as_m128()
}

// ...
// _MM_TRANSPOSE4_PS (__m128 row0, __m128 row1, __m128 row2, __m128 row3)
#[inline]
pub fn mm_transpose4_ps(row0: &mut m128, row1: &mut m128, row2: &mut m128, row3: &mut m128) {
    let tmp0 = mm_unpacklo_ps(*row0, *row1);
    let tmp2 = mm_unpacklo_ps(*row2, *row3);
    let tmp1 = mm_unpackhi_ps(*row0, *row1);
    let tmp3 = mm_unpackhi_ps(*row2, *row3);
    *row0 = mm_movelh_ps(tmp0, tmp2);
    *row1 = mm_movehl_ps(tmp2, tmp0);
    *row2 = mm_movelh_ps(tmp1, tmp3);
    *row3 = mm_movehl_ps(tmp3, tmp1);
}

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
#[inline]
pub fn mm_unpackhi_ps(a: m128, b: m128) -> m128 {
    unsafe { simd_shuffle4(a, b, [2, 6, 3, 7]) }
}

// unpcklps
// __m128 _mm_unpacklo_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_unpacklo_ps(a: m128, b: m128) -> m128 {
    unsafe { simd_shuffle4(a, b, [0, 4, 1, 5]) }
}

// xorps
// __m128 _mm_xor_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_xor_ps(a: m128, b: m128) -> m128 {
    let ai = a.as_m128i();
    let bi = b.as_m128i();
    unsafe { simd_xor(ai, bi).as_m128() }
}

// MMX methods
// pavgw
// __m64 _mm_avg_pu16 (__m64 a, __m64 b)
// pavgb
// __m64 _mm_avg_pu8 (__m64 a, __m64 b)
// cvtpi2ps
// __m128 _mm_cvt_pi2ps (__m128 a, __m64 b)
// cvtps2pi
// __m64 _mm_cvt_ps2pi (__m128 a)
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
// cvttps2pi
// __m64 _mm_cvtt_ps2pi (__m128 a)
// cvttps2pi
// __m64 _mm_cvttps_pi32 (__m128 a)
// pextrw
// int _mm_extract_pi16 (__m64 a, int imm8)
// pinsrw
// __m64 _mm_insert_pi16 (__m64 a, int i, int imm8)
// maskmovq
// void _mm_maskmove_si64 (__m64 a, __m64 mask, char* mem_addr)
// maskmovq
// void _m_maskmovq (__m64 a, __m64 mask, char* mem_addr)
// pmaxsw
// __m64 _mm_max_pi16 (__m64 a, __m64 b)
// pmaxub
// __m64 _mm_max_pu8 (__m64 a, __m64 b)
// pminsw
// __m64 _mm_min_pi16 (__m64 a, __m64 b)
// pminub
// __m64 _mm_min_pu8 (__m64 a, __m64 b)
// pmovmskb
// int _mm_movemask_pi8 (__m64 a)
// pmulhuw
// __m64 _mm_mulhi_pu16 (__m64 a, __m64 b)
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
// psadbw
// __m64 _mm_sad_pu8 (__m64 a, __m64 b)
// movntq
// void _mm_stream_pi (__m64* mem_addr, __m64 a)
// movhps
// __m128 _mm_loadh_pi (__m128 a, __m64 const* mem_addr)
// movlps
// __m128 _mm_loadl_pi (__m128 a, __m64 const* mem_addr)
// movhps
// void _mm_storeh_pi (__m64* mem_addr, __m128 a)
// movlps
// void _mm_storel_pi (__m64* mem_addr, __m128 a)

#[cfg(test)]
mod tests {
    use std;
    use super::super::*;

    #[test]
    fn test_mm_arith_ps() {
        let x = mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let y = mm_setr_ps(2.0, 4.0, 2.0, 2.0);

        assert_eq!(mm_add_ps(x, y).as_f32x4().as_array(), [3.0, 6.0, 5.0, 6.0]);
        assert_eq!(mm_sub_ps(x, y).as_f32x4().as_array(), [-1.0, -2.0, 1.0, 2.0]);
        assert_eq!(mm_mul_ps(x, y).as_f32x4().as_array(), [2.0, 8.0, 6.0, 8.0]);
        assert_eq!(mm_div_ps(x, y).as_f32x4().as_array(), [0.5, 0.5, 1.5, 2.0]);
    }

    #[test]
    fn test_mm_arith_ss() {
        let x = mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let y = mm_setr_ps(2.0, 4.0, 2.0, 2.0);

        assert_eq!(mm_add_ss(x, y).as_f32x4().as_array(), [3.0, 2.0, 3.0, 4.0]);
        assert_eq!(mm_sub_ss(x, y).as_f32x4().as_array(), [-1.0, 2.0, 3.0, 4.0]);
        assert_eq!(mm_mul_ss(x, y).as_f32x4().as_array(), [2.0, 2.0, 3.0, 4.0]);
        assert_eq!(mm_div_ss(x, y).as_f32x4().as_array(), [0.5, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_mm_math_ps() {
        let x = mm_setr_ps(1.0, 2.0, 3.0, 4.0);

        let expected_rcp = [1.0 / 1.0, 1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0];
        let expected_rsqrt = [1.0 / 1.0f32.sqrt(), 1.0 / 2.0f32.sqrt(), 1.0 / 3.0f32.sqrt(), 1.0 / 4.0f32.sqrt()];
        let expected_sqrt = [1.0f32.sqrt(), 2.0f32.sqrt(), 3.0f32.sqrt(), 4.0f32.sqrt()];

        let actual_rcp = mm_rcp_ps(x).as_f32x4().as_array();
        let actual_rsqrt = mm_rsqrt_ps(x).as_f32x4().as_array();
        let actual_sqrt = mm_sqrt_ps(x).as_f32x4().as_array();

        for i in 0 .. 4 {
            let a = actual_rcp[i];
            let e = expected_rcp[i];
            assert!((a - e).abs() <= 1.5 * (1.0 / 4096.0));
        }
        for i in 0 .. 4 {
            let a = actual_rsqrt[i];
            let e = expected_rsqrt[i];
            assert!((a - e).abs() <= 1.5 * (1.0 / 4096.0));
        }
        for i in 0 .. 4 {
            let a = actual_sqrt[i];
            let e = expected_sqrt[i];
            assert!((a - e).abs() <= 1.5 * (1.0 / 4096.0));
        }
    }

    #[test]
    fn test_mm_math_ss() {
        let x = mm_setr_ps(3.0, 2.0, 3.0, 4.0);

        let actual_rcp_ss = mm_rcp_ss(x).as_f32x4().as_array();
        let actual_rsqrt_ss = mm_rsqrt_ss(x).as_f32x4().as_array();
        let actual_sqrt_ss = mm_sqrt_ss(x).as_f32x4().as_array();

        let expected_rcp = 1.0 / 3.0;
        let expected_rsqrt = 1.0 / 3.0f32.sqrt();
        let expected_sqrt = 3.0f32.sqrt();

        assert!((actual_rcp_ss[0] - expected_rcp).abs() <= 1.5 * (1.0 / 4096.0));
        assert!((actual_rsqrt_ss[0] - expected_rsqrt).abs() <= 1.5 * (1.0 / 4096.0));
        assert!((actual_sqrt_ss[0] - expected_sqrt).abs() <= 1.5 * (1.0 / 4096.0));

        // TODO(mayah): this doesn't pass in debug build.
        // x(1), x(2), x(3) should remain the original value.
        //
        // for i in 1 .. 4 {
        //    assert_eq!(actual_rcp_ss[i], x.as_f32x4().extract(i));
        //    assert_eq!(actual_rsqrt_ss[i], x.as_f32x4().extract(i));
        //    assert_eq!(actual_sqrt_ss[i], x.as_f32x4().extract(i));
        // }
    }

    #[test]
    fn test_mm_logic_ps() {
        let x = i32x4(0x1, 0x2, 0x3, 0x4).as_m128();
        let y = i32x4(0x3, 0x4, 0x5, 0x6).as_m128();

        assert_eq!(mm_and_ps(x, y).as_m128i().as_i32x4().as_array(),
                   [0x1 & 0x3, 0x2 & 0x4, 0x3 & 0x5, 0x4 & 0x6]);
        assert_eq!(mm_or_ps(x, y).as_m128i().as_i32x4().as_array(),
                   [0x1 | 0x3, 0x2 | 0x4, 0x3 | 0x5, 0x4 | 0x6]);
        assert_eq!(mm_xor_ps(x, y).as_m128i().as_i32x4().as_array(),
                   [0x1 ^ 0x3, 0x2 ^ 0x4, 0x3 ^ 0x5, 0x4 ^ 0x6]);
        assert_eq!(mm_andnot_ps(x, y).as_m128i().as_i32x4().as_array(),
                   [!0x1 & 0x3, !0x2 & 0x4, !0x3 & 0x5, !0x4 & 0x6]);
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
    fn test_minmax() {
        let x = mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let y = mm_setr_ps(3.0, 2.0, 1.0, 0.0);

        assert_eq!(mm_max_ps(x, y).as_f32x4().as_array(), [3.0, 2.0, 3.0, 4.0]);
        assert_eq!(mm_max_ss(x, y).as_f32x4().as_array(), [3.0, 2.0, 3.0, 4.0]);
        assert_eq!(mm_min_ps(x, y).as_f32x4().as_array(), [1.0, 2.0, 1.0, 0.0]);
        assert_eq!(mm_min_ss(x, y).as_f32x4().as_array(), [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_move() {
        let x = mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let y = mm_setr_ps(3.0, 2.0, 1.0, 0.0);

        assert_eq!(mm_move_ss(x, y).as_f32x4().as_array(), [3.0, 2.0, 3.0, 4.0]);
        assert_eq!(mm_movehl_ps(x, y).as_f32x4().as_array(), [1.0, 0.0, 3.0, 4.0]);
        assert_eq!(mm_movelh_ps(x, y).as_f32x4().as_array(), [1.0, 2.0, 3.0, 2.0]);
    }

    #[test]
    fn test_movemask() {
        let x = mm_setr_ps(1.0, 2.0, -3.0, -4.0);
        assert_eq!(mm_movemask_ps(x), (1 << 2) | (1 << 3));
    }

    #[test]
    fn test_unpack() {
        let x = mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let y = mm_setr_ps(5.0, 6.0, 7.0, 8.0);

        assert_eq!(mm_unpackhi_ps(x, y).as_f32x4().as_array(), [3.0, 7.0, 4.0, 8.0]);
        assert_eq!(mm_unpacklo_ps(x, y).as_f32x4().as_array(), [1.0, 5.0, 2.0, 6.0]);
    }

    #[test]
    fn test_store() {
        let ps = mm_setr_ps(1.0, 2.0, 3.0, 4.0);

        let mut buf: [f32; 4] = [0.0; 4];
        let p: *mut f32 = unsafe { std::mem::transmute(&mut buf) };

        unsafe { mm_store_ps(p, ps) };
        assert_eq!(buf, [1.0, 2.0, 3.0, 4.0]);

        unsafe { mm_store1_ps(p, ps) };
        assert_eq!(buf, [1.0, 1.0, 1.0, 1.0]);

        unsafe { mm_storer_ps(p, ps) };
        assert_eq!(buf, [4.0, 3.0, 2.0, 1.0]);

        unsafe { mm_store_ss(p, ps) };
        assert_eq!(buf, [1.0, 3.0, 2.0, 1.0]);

        unsafe { mm_storeu_ps(p, ps) };
        assert_eq!(buf, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_load() {
        let buf: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let p = &buf as *const [f32; 4] as *const f32;

        assert_eq!(unsafe { mm_load_ps(p) }.as_f32x4().as_array(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(unsafe { mm_load_ps1(p) }.as_f32x4().as_array(), [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(unsafe { mm_load_ss(p) }.as_f32x4().as_array(), [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(unsafe { mm_load1_ps(p) }.as_f32x4().as_array(), [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(unsafe { mm_loadr_ps(p) }.as_f32x4().as_array(), [4.0, 3.0, 2.0, 1.0]);
        assert_eq!(unsafe { mm_loadu_ps(p) }.as_f32x4().as_array(), [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_transpose4() {
        let mut row0 = mm_setr_ps( 1.0,  2.0,  3.0,  4.0);
        let mut row1 = mm_setr_ps( 5.0,  6.0,  7.0,  8.0);
        let mut row2 = mm_setr_ps( 9.0, 10.0, 11.0, 12.0);
        let mut row3 = mm_setr_ps(13.0, 14.0, 15.0, 16.0);
        mm_transpose4_ps(&mut row0, &mut row1, &mut row2, &mut row3);

        assert_eq!(row0.as_f32x4().as_array(), [1.0, 5.0,  9.0, 13.0]);
        assert_eq!(row1.as_f32x4().as_array(), [2.0, 6.0, 10.0, 14.0]);
        assert_eq!(row2.as_f32x4().as_array(), [3.0, 7.0, 11.0, 15.0]);
        assert_eq!(row3.as_f32x4().as_array(), [4.0, 8.0, 12.0, 16.0]);
    }
}
