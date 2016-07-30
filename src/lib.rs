// Intrinsic list is here.
// Web: https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=133
// XML: https://software.intel.com/sites/landingpage/IntrinsicsGuide/files/data-3.3.14.xml

#![feature(cfg_target_feature)]
#![feature(link_llvm_intrinsics)]
#![feature(platform_intrinsics)]
#![feature(repr_simd)]
#![feature(simd_ffi)]

extern "platform-intrinsic" {
    fn simd_add<T>(x: T, y: T) -> T;

    fn simd_and<T>(x: T, y: T) -> T;
    fn simd_or<T>(x: T, y: T) -> T;
    fn simd_xor<T>(x: T, y: T) -> T;
    fn simd_extract<T, U>(x: T, idx: u32) -> U;

    fn simd_shuffle16<T, U>(x: T, y: T, idx: [u32; 16]) -> U;
}

#[inline]
unsafe fn bitcast<T, U>(x: T) -> U {
    debug_assert!(std::mem::size_of::<T>() == std::mem::size_of::<U>());
    std::mem::transmute_copy(&x)
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct m128i(i32, i32, i32, i32);

impl m128i {
    #[inline]
    pub fn as_i64x2(self) -> i64x2 { unsafe { bitcast(self) } }
    #[inline]
    pub fn as_u64x2(self) -> u64x2 { unsafe { bitcast(self) } }
    #[inline]
    pub fn as_i32x4(self) -> i32x4 { unsafe { bitcast(self) } }
    #[inline]
    pub fn as_u32x4(self) -> u32x4 { unsafe { bitcast(self) } }
    #[inline]
    pub fn as_i16x8(self) -> i16x8 { unsafe { bitcast(self) } }
    #[inline]
    pub fn as_u16x8(self) -> u16x8 { unsafe { bitcast(self) } }
    #[inline]
    pub fn as_i8x16(self) -> i8x16 { unsafe { bitcast(self) } }
    #[inline]
    pub fn as_u8x16(self) -> u8x16 { unsafe { bitcast(self) } }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct m128(f32, f32, f32, f32);

impl m128 {
    pub fn as_f32x4(self) -> f32x4 { unsafe { bitcast(self) } }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct m128d(f64, f64);

impl m128d {
    pub fn as_f64x2(self) -> f64x2 { unsafe { bitcast(self) } }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct i64x2(i64, i64);

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct u64x2(u64, u64);

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct f64x2(f64, f64);

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct i32x4(i32, i32, i32, i32);

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct u32x4(u32, u32, u32, u32);

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct f32x4(f32, f32, f32, f32);

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct i16x8(i16, i16, i16, i16, i16, i16, i16, i16);

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct u16x8(u16, u16, u16, u16, u16, u16, u16, u16);

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct i8x16(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8);

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct u8x16(u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8);

macro_rules! simd_type {
    ($name: ident, $elem: ident, $size: expr) => {
        impl $name {
            #[inline]
            pub fn extract(self, idx: usize) -> $elem {
                debug_assert!(idx < $size);
                unsafe { simd_extract(self, idx as u32) }
            }

            #[inline]
            pub fn as_m128i(self) -> m128i {
                unsafe { bitcast(self) }
            }

            #[inline]
            pub fn as_m128(self) -> m128 {
                unsafe { bitcast(self) }
            }

            #[inline]
            pub fn as_m128d(self) -> m128d {
                unsafe { bitcast(self) }
            }
        }
    }
}

simd_type! { i64x2, i64, 2 }
simd_type! { u64x2, u64, 2 }
simd_type! { f64x2, f64, 2 }
simd_type! { i32x4, i32, 4 }
simd_type! { u32x4, u32, 4 }
simd_type! { f32x4, f32, 4 }
simd_type! { i16x8, i16, 8 }
simd_type! { u16x8, u16, 8 }
simd_type! { i8x16, i8, 16 }
simd_type! { u8x16, u8, 16 }

#[cfg(any(feature = "doc", target_feature = "sse"))]
pub mod sse;
#[cfg(any(feature = "doc", target_feature = "sse"))]
pub use sse::*;

#[cfg(any(feature = "doc", target_feature = "sse2"))]
pub mod sse2;
#[cfg(any(feature = "doc", target_feature = "sse2"))]
pub use sse2::*;

#[cfg(any(feature = "doc", target_feature = "sse3"))]
pub mod sse3;
#[cfg(any(feature = "doc", target_feature = "sse3"))]
pub use sse3::*;

#[cfg(any(feature = "doc", target_feature = "ssse3"))]
pub mod ssse3;
#[cfg(any(feature = "doc", target_feature = "ssse3"))]
pub use ssse3::*;

#[cfg(any(feature = "doc", target_feature = "sse4.1"))]
pub mod sse41;
#[cfg(any(feature = "doc", target_feature = "sse4.1"))]
pub use sse41::*;

#[cfg(any(feature = "doc", target_feature = "sse4.2"))]
pub mod sse42;
#[cfg(any(feature = "doc", target_feature = "sse4.2"))]
pub use sse42::*;

#[cfg(any(feature = "doc", target_feature = "avx"))]
pub mod avx;
#[cfg(any(feature = "doc", target_feature = "avx"))]
pub use avx::*;

#[cfg(any(feature = "doc", target_feature = "avx2"))]
pub mod avx2;
#[cfg(any(feature = "doc", target_feature = "avx2"))]
pub use avx2::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_m128i_ops() {
        let x = m128i(1, 2, 3, 4);
        let y = m128i(2, 3, 4, 5);

        let xy_and = (x & y).as_i32x4();
        let xy_or  = (x | y).as_i32x4();
        let xy_xor = (x ^ y).as_i32x4();

        assert_eq!(xy_and.extract(0), 1 & 2);
        assert_eq!(xy_and.extract(1), 2 & 3);
        assert_eq!(xy_and.extract(2), 3 & 4);
        assert_eq!(xy_and.extract(3), 4 & 5);

        assert_eq!(xy_or.extract(0), 1 | 2);
        assert_eq!(xy_or.extract(1), 2 | 3);
        assert_eq!(xy_or.extract(2), 3 | 4);
        assert_eq!(xy_or.extract(3), 4 | 5);

        assert_eq!(xy_xor.extract(0), 1 ^ 2);
        assert_eq!(xy_xor.extract(1), 2 ^ 3);
        assert_eq!(xy_xor.extract(2), 3 ^ 4);
        assert_eq!(xy_xor.extract(3), 4 ^ 5);
    }

    #[test]
    fn basic_i64x2() {
        let x = i64x2(3, 9);
        assert_eq!(x.extract(0), 3);
        assert_eq!(x.extract(1), 9);
    }

    #[test]
    fn base_i32x4() {
        let x = i32x4(1, 2, 3, 4);
        assert_eq!(x.extract(0), 1);
        assert_eq!(x.extract(1), 2);
        assert_eq!(x.extract(2), 3);
        assert_eq!(x.extract(3), 4);
    }

    #[test]
    fn base_i16x8() {
        let x = i16x8(1, 2, 3, 4, 5, 6, 7, 8);
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
        let x = u16x8(1, 2, 3, 4, 5, 6, 7, 8);
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
        let x = i8x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        for i in 0 .. 16 {
            assert_eq!(x.extract(i), (i + 1) as i8);
        }
    }
}
