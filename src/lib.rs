// Intrinsic list is here.
// Web: https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=133
// XML: https://software.intel.com/sites/landingpage/IntrinsicsGuide/files/data-3.3.14.xml

#![feature(link_llvm_intrinsics)]
#![feature(platform_intrinsics)]
#![feature(repr_simd)]
#![feature(simd_ffi)]

extern "platform-intrinsic" {
    fn simd_and<T>(x: T, y: T) -> T;
    fn simd_or<T>(x: T, y: T) -> T;
    fn simd_xor<T>(x: T, y: T) -> T;
    fn simd_cast<T, U>(x: T) -> U;
    fn simd_extract<T, U>(x: T, idx: u32) -> U;
}

#[inline]
unsafe fn bitcast<T, U>(x: T) -> U {
    debug_assert!(std::mem::size_of::<T>() == std::mem::size_of::<U>());
    std::mem::transmute_copy(&x)
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct i64x2(i64, i64);

impl i64x2 {
    #[inline]
    pub fn new(r0: i64, r1: i64) -> i64x2 {
        i64x2(r0, r1)
    }

    #[inline]
    pub fn extract(self, idx: usize) -> i64 {
        debug_assert!(idx < 2);
        unsafe { simd_extract(self, idx as u32) }
    }

    #[inline]
    pub fn as_m128i(self) -> m128i {
        unsafe { bitcast(self) }
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct i32x4(i32, i32, i32, i32);

impl i32x4 {
    #[inline]
    pub fn new(r0: i32, r1: i32, r2: i32, r3: i32) -> i32x4 {
        i32x4(r0, r1, r2, r3)
    }

    #[inline]
    pub fn extract(self, idx: usize) -> i32 {
        debug_assert!(idx < 4);
        unsafe { simd_extract(self, idx as u32) }
    }

    #[inline]
    pub fn as_m128i(self) -> m128i {
        unsafe { bitcast(self) }
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct i16x8(i16, i16, i16, i16, i16, i16, i16, i16);

impl i16x8 {
    #[inline]
    pub fn new(r0: i16, r1: i16, r2: i16, r3: i16,
               r4: i16, r5: i16, r6: i16, r7: i16) -> i16x8 {
        i16x8(r0, r1, r2, r3, r4, r5, r6, r7)
    }

    #[inline]
    pub fn extract(self, idx: usize) -> i16 {
        debug_assert!(idx < 8);
        unsafe { simd_extract(self, idx as u32) }
    }

    #[inline]
    pub fn as_m128i(self) -> m128i {
        unsafe { bitcast(self) }
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct u16x8(u16, u16, u16, u16, u16, u16, u16, u16);

impl u16x8 {
    #[inline]
    pub fn new(r0: u16, r1: u16, r2: u16, r3: u16,
               r4: u16, r5: u16, r6: u16, r7: u16) -> u16x8 {
        u16x8(r0, r1, r2, r3, r4, r5, r6, r7)
    }

    #[inline]
    pub fn extract(self, idx: usize) -> u16 {
        debug_assert!(idx < 8);
        unsafe { simd_extract(self, idx as u32) }
    }

    #[inline]
    pub fn as_m128i(self) -> m128i {
        unsafe { bitcast(self) }
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct i8x16(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8);

impl i8x16 {
    #[inline]
    pub fn new(r0: i8, r1: i8,  r2: i8,  r3: i8,  r4: i8,  r5: i8,  r6: i8,  r7: i8,
               r8: i8, r9: i8, r10: i8, r11: i8, r12: i8, r13: i8, r14: i8, r15: i8) -> i8x16 {
        i8x16(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15)
    }

    #[inline]
    pub fn extract(self, idx: usize) -> i8 {
        debug_assert!(idx < 16);
        unsafe { simd_extract(self, idx as u32) }
    }

    #[inline]
    pub fn as_m128i(self) -> m128i {
        unsafe { bitcast(self) }
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct m128i(i64, i64);

impl m128i {
    #[inline]
    pub fn as_i64x2(self) -> i64x2 {
        unsafe { simd_cast(self) }
    }

    #[inline]
    pub fn as_i32x4(self) -> i32x4 {
        unsafe { bitcast(self) }
    }

    #[inline]
    pub fn as_i16x8(self) -> i16x8 {
        unsafe { bitcast(self) }
    }
}

extern {
    #[link_name = "llvm.x86.sse2.pslli.w"]
    pub fn sse2_pslli_w(a: i16x8, b: i32) -> i16x8;
    #[link_name = "llvm.x86.sse2.psrli.w"]
    pub fn sse2_psrli_w(a: i16x8, b: i32) -> i16x8;
}

#[inline]
pub fn mm_setzero_si128() -> m128i {
    m128i(0, 0)
}

#[inline]
pub fn mm_set_epi32(r3: i32, r2: i32, r1: i32, r0: i32) -> m128i {
    i32x4(r0, r1, r2, r3).as_m128i()
}

#[inline]
pub fn mm_setr_epi32(r0: i32, r1: i32, r2: i32, r3: i32) -> m128i {
    i32x4(r0, r1, r2, r3).as_m128i()
}

#[inline]
pub fn mm_set_epi16(r7: i16, r6: i16, r5: i16, r4: i16,
                    r3: i16, r2: i16, r1: i16, r0: i16) -> m128i {
    i16x8(r0, r1, r2, r3, r4, r5, r6, r7).as_m128i()
}

#[inline]
pub fn mm_setr_epi16(r0: i16, r1: i16, r2: i16, r3: i16,
                     r4: i16, r5: i16, r6: i16, r7: i16) -> m128i {
    i16x8(r0, r1, r2, r3, r4, r5, r6, r7).as_m128i()
}

#[inline]
pub fn mm_and_si128(a: m128i, b: m128i) -> m128i {
    unsafe { simd_and(a, b) }
}

#[inline]
pub fn mm_or_si128(a: m128i, b: m128i) -> m128i {
    unsafe { simd_or(a, b) }
}

#[inline]
pub fn mm_xor_si128(a: m128i, b: m128i) -> m128i {
    unsafe { simd_xor(a, b) }
}

#[inline]
pub fn mm_andnot_si128(a: m128i, b: m128i) -> m128i {
    let ones = i64x2::new(!0, !0).as_m128i();
    mm_and_si128(mm_xor_si128(a, ones), b)
}

#[inline]
pub fn mm_slli_epi16(a: m128i, imm8: i32) -> m128i {
    unsafe { bitcast(sse2_pslli_w(a.as_i16x8(), imm8)) }
}

#[inline]
pub fn mm_srli_epi16(a: m128i, imm8: i32) -> m128i {
    unsafe { bitcast(sse2_psrli_w(a.as_i16x8(), imm8)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_i64x2() {
        let x = i64x2::new(3, 9);
        assert_eq!(x.extract(0), 3);
        assert_eq!(x.extract(1), 9);
    }

    #[test]
    fn base_i32x4() {
        let x = i32x4::new(1, 2, 3, 4);
        assert_eq!(x.extract(0), 1);
        assert_eq!(x.extract(1), 2);
        assert_eq!(x.extract(2), 3);
        assert_eq!(x.extract(3), 4);
    }

    #[test]
    fn base_i16x8() {
        let x = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq!(x.extract(0), 1);
        assert_eq!(x.extract(1), 2);
        assert_eq!(x.extract(2), 3);
        assert_eq!(x.extract(3), 4);
        assert_eq!(x.extract(4), 5);
        assert_eq!(x.extract(5), 6);
        assert_eq!(x.extract(6), 7);
        assert_eq!(x.extract(7), 8);
    }

    #[test]
    fn base_u16x8() {
        let x = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq!(x.extract(0), 1);
        assert_eq!(x.extract(1), 2);
        assert_eq!(x.extract(2), 3);
        assert_eq!(x.extract(3), 4);
        assert_eq!(x.extract(4), 5);
        assert_eq!(x.extract(5), 6);
        assert_eq!(x.extract(6), 7);
        assert_eq!(x.extract(7), 8);
    }

    #[test]
    fn base_i8x16() {
        let x = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        for i in 0 .. 16 {
            assert_eq!(x.extract(i), (i + 1) as i8);
        }
    }

    #[test]
    fn test_mm_setzero_si128() {
        let zero = mm_setzero_si128().as_i64x2();
        assert_eq!(zero.extract(0), 0);
        assert_eq!(zero.extract(1), 0);
    }

    #[test]
    fn test_mm_set_epi32() {
        let x = mm_set_epi32(1, 2, 3, 4).as_i32x4();
        assert_eq!(x.extract(0), 4);
        assert_eq!(x.extract(1), 3);
        assert_eq!(x.extract(2), 2);
        assert_eq!(x.extract(3), 1);
    }

    #[test]
    fn test_mm_setr_epi32() {
        let x = mm_setr_epi32(1, 2, 3, 4).as_i32x4();
        assert_eq!(x.extract(0), 1);
        assert_eq!(x.extract(1), 2);
        assert_eq!(x.extract(2), 3);
        assert_eq!(x.extract(3), 4);
    }

    #[test]
    fn test_mm_set_epi16() {
        let x = mm_set_epi16(1, 2, 3, 4, 5, 6, 7, 8).as_i16x8();
        assert_eq!(x.extract(0), 8);
        assert_eq!(x.extract(1), 7);
        assert_eq!(x.extract(2), 6);
        assert_eq!(x.extract(3), 5);
        assert_eq!(x.extract(4), 4);
        assert_eq!(x.extract(5), 3);
        assert_eq!(x.extract(6), 2);
        assert_eq!(x.extract(7), 1);
    }

    #[test]
    fn test_mm_setr_epi16() {
        let x = mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8).as_i16x8();
        assert_eq!(x.extract(0), 1);
        assert_eq!(x.extract(1), 2);
        assert_eq!(x.extract(2), 3);
        assert_eq!(x.extract(3), 4);
        assert_eq!(x.extract(4), 5);
        assert_eq!(x.extract(5), 6);
        assert_eq!(x.extract(6), 7);
        assert_eq!(x.extract(7), 8);
    }

    #[test]
    fn test_mm_and_si128() {
        let x = mm_setr_epi32(0x3F, 0x7E, 0x13, 0xFF);
        let y = mm_setr_epi32(0x53, 0x8C, 0xFF, 0x17);
        let z = mm_and_si128(x, y).as_i32x4();
        assert_eq!(z.extract(0), 0x3F & 0x53);
        assert_eq!(z.extract(1), 0x7E & 0x8C);
        assert_eq!(z.extract(2), 0x13 & 0xFF);
        assert_eq!(z.extract(3), 0xFF & 0x17);
    }

    #[test]
    fn test_mm_or_si128() {
        let x = mm_setr_epi32(0x3F, 0x7E, 0x13, 0xFF);
        let y = mm_setr_epi32(0x53, 0x8C, 0xFF, 0x17);
        let z = mm_or_si128(x, y).as_i32x4();
        assert_eq!(z.extract(0), 0x3F | 0x53);
        assert_eq!(z.extract(1), 0x7E | 0x8C);
        assert_eq!(z.extract(2), 0x13 | 0xFF);
        assert_eq!(z.extract(3), 0xFF | 0x17);
    }

    #[test]
    fn test_mm_xor_si128() {
        let x = mm_setr_epi32(0x3F, 0x7E, 0x13, 0xFF);
        let y = mm_setr_epi32(0x53, 0x8C, 0xFF, 0x17);
        let z = mm_xor_si128(x, y).as_i32x4();
        assert_eq!(z.extract(0), 0x3F ^ 0x53);
        assert_eq!(z.extract(1), 0x7E ^ 0x8C);
        assert_eq!(z.extract(2), 0x13 ^ 0xFF);
        assert_eq!(z.extract(3), 0xFF ^ 0x17);
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
}
