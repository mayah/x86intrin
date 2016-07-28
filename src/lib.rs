// Intrinsic list is here.
// Web: https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=133
// XML: https://software.intel.com/sites/landingpage/IntrinsicsGuide/files/data-3.3.14.xml

#![feature(repr_simd)]
#![feature(platform_intrinsics)]

extern "platform-intrinsic" {
    fn simd_and<T>(x: T, y: T) -> T;
    fn simd_or<T>(x: T, y: T) -> T;
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
#[repr(simd)]
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
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(simd)]
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
#[repr(simd)]
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
pub fn mm_and_si128(a: m128i, b: m128i) -> m128i {
    unsafe { simd_and(a, b) }
}

#[inline]
pub fn mm_or_si128(a: m128i, b: m128i) -> m128i {
    unsafe { simd_or(a, b) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_i32x4() {
        let x = i32x4::new(1, 2, 3, 4);
        assert_eq!(x.extract(0), 1);
        assert_eq!(x.extract(1), 2);
        assert_eq!(x.extract(2), 3);
        assert_eq!(x.extract(3), 4);
    }

    #[test]
    fn basic_i64x2() {
        let x = i64x2::new(3, 9);
        assert_eq!(x.extract(0), 3);
        assert_eq!(x.extract(1), 9);
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
}
