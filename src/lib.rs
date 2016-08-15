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
    fn simd_sub<T>(x: T, y: T) -> T;
    fn simd_mul<T>(x: T, y: T) -> T;
    fn simd_div<T>(x: T, y: T) -> T;

    fn simd_and<T>(x: T, y: T) -> T;
    fn simd_or<T>(x: T, y: T) -> T;
    fn simd_xor<T>(x: T, y: T) -> T;

    fn simd_eq<T, U>(x: T, y: T) -> U;
    fn simd_ge<T, U>(x: T, y: T) -> U;
    fn simd_gt<T, U>(x: T, y: T) -> U;
    fn simd_lt<T, U>(x: T, y: T) -> U;
    fn simd_le<T, U>(x: T, y: T) -> U;
    fn simd_ne<T, U>(x: T, y: T) -> U;

    fn simd_extract<T, U>(x: T, idx: u32) -> U;
    fn simd_insert<T, U>(x: T, idx: u32, v: U) -> T;

    fn simd_shuffle2<T, U>(x: T, y: T, idx: [u32; 2]) -> U;
    fn simd_shuffle4<T, U>(x: T, y: T, idx: [u32; 4]) -> U;
    fn simd_shuffle8<T, U>(x: T, y: T, idx: [u32; 8]) -> U;
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
pub struct m128i(i64, i64);

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

    #[inline]
    pub fn as_m128i(self) -> m128i { unsafe { bitcast(self) } }
    #[inline]
    pub fn as_m128(self) -> m128 { unsafe { bitcast(self) } }
    #[inline]
    pub fn as_m128d(self) -> m128d { unsafe { bitcast(self) } }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct m128(f32, f32, f32, f32);

impl m128 {
    #[inline]
    pub fn as_f32x4(self) -> f32x4 { unsafe { bitcast(self) } }

    #[inline]
    pub fn as_m128i(self) -> m128i { unsafe { bitcast(self) } }
    #[inline]
    pub fn as_m128(self) -> m128 { unsafe { bitcast(self) } }
    #[inline]
    pub fn as_m128d(self) -> m128d { unsafe { bitcast(self) } }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct m128d(f64, f64);

impl m128d {
    #[inline]
    pub fn as_f64x2(self) -> f64x2 { unsafe { bitcast(self) } }

    #[inline]
    pub fn as_m128i(self) -> m128i { unsafe { bitcast(self) } }
    #[inline]
    pub fn as_m128(self) -> m128 { unsafe { bitcast(self) } }
    #[inline]
    pub fn as_m128d(self) -> m128d { unsafe { bitcast(self) } }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct m256i(i32, i32, i32, i32, i32, i32, i32, i32);

impl m256i {
    #[inline]
    pub fn as_i64x4(self) -> i64x4 { unsafe { bitcast(self) } }
    #[inline]
    pub fn as_u64x4(self) -> u64x4 { unsafe { bitcast(self) } }
    #[inline]
    pub fn as_i32x8(self) -> i32x8 { unsafe { bitcast(self) } }
    #[inline]
    pub fn as_u32x8(self) -> u32x8 { unsafe { bitcast(self) } }
    #[inline]
    pub fn as_i16x16(self) -> i16x16 { unsafe { bitcast(self) } }
    #[inline]
    pub fn as_u16x16(self) -> u16x16 { unsafe { bitcast(self) } }
    #[inline]
    pub fn as_i8x32(self) -> i8x32 { unsafe { bitcast(self) } }
    #[inline]
    pub fn as_u8x32(self) -> u8x32 { unsafe { bitcast(self) } }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct m256(f32, f32, f32, f32, f32, f32, f32, f32);

impl m256 {
    #[inline]
    pub fn as_f32x8(self) -> f32x8 { unsafe { bitcast(self) } }

    #[inline]
    pub fn as_m256i(self) -> m256i { unsafe { bitcast(self) } }
    #[inline]
    pub fn as_m256(self) -> m256 { unsafe { bitcast(self) } }
    #[inline]
    pub fn as_m256d(self) -> m256d { unsafe { bitcast(self) } }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct m256d(f64, f64, f64, f64);

impl m256d {
    #[inline]
    pub fn as_f64x4(self) -> f64x4 { unsafe { bitcast(self) } }

    #[inline]
    pub fn as_m256i(self) -> m256i { unsafe { bitcast(self) } }
    #[inline]
    pub fn as_m256(self) -> m256 { unsafe { bitcast(self) } }
    #[inline]
    pub fn as_m256d(self) -> m256d { unsafe { bitcast(self) } }
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

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct i64x4(i64, i64, i64, i64);

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct u64x4(u64, u64, u64, u64);

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct f64x4(f64, f64, f64, f64);

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct i32x8(i32, i32, i32, i32, i32, i32, i32, i32);

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct u32x8(u32, u32, u32, u32, u32, u32, u32, u32);

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct f32x8(f32, f32, f32, f32, f32, f32, f32, f32);

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct i16x16(i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16);

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct u16x16(u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16);

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct i8x32(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8,
                 i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8);

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(C, simd)]
pub struct u8x32(u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8,
                 u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8);

macro_rules! simd_128_type {
    ($name: ident, $elem: ident, $size: expr) => {
        impl $name {
            #[inline]
            pub fn extract(self, idx: usize) -> $elem {
                debug_assert!(idx < $size);
                unsafe { simd_extract(self, idx as u32) }
            }

            #[inline]
            pub fn insert(self, idx: usize, v: $elem) -> $name {
                debug_assert!(idx < $size);
                unsafe { simd_insert(self, idx as u32, v) }
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

            #[inline]
            pub fn as_array(self) -> [$elem; $size] {
                let mut data: [$elem; $size];
                unsafe {
                    data = std::mem::uninitialized();
                    for i in 0 .. $size {
                        data[i] = self.extract(i)
                    }
                };
                data
            }
        }
    }
}

simd_128_type! { i64x2, i64, 2 }
simd_128_type! { u64x2, u64, 2 }
simd_128_type! { f64x2, f64, 2 }
simd_128_type! { i32x4, i32, 4 }
simd_128_type! { u32x4, u32, 4 }
simd_128_type! { f32x4, f32, 4 }
simd_128_type! { i16x8, i16, 8 }
simd_128_type! { u16x8, u16, 8 }
simd_128_type! { i8x16, i8, 16 }
simd_128_type! { u8x16, u8, 16 }

macro_rules! simd_256_type {
    ($name: ident, $elem: ident, $size: expr) => {
        impl $name {
            #[inline]
            pub fn extract(self, idx: usize) -> $elem {
                debug_assert!(idx < $size);
                unsafe { simd_extract(self, idx as u32) }
            }

            #[inline]
            pub fn insert(self, idx: usize, v: $elem) -> $name {
                debug_assert!(idx < $size);
                unsafe { simd_insert(self, idx as u32, v) }
            }

            #[inline]
            pub fn as_m256i(self) -> m256i {
                unsafe { bitcast(self) }
            }

            #[inline]
            pub fn as_m256(self) -> m256 {
                unsafe { bitcast(self) }
            }

            #[inline]
            pub fn as_m256d(self) -> m256d {
                unsafe { bitcast(self) }
            }

            #[inline]
            pub fn as_array(self) -> [$elem; $size] {
                let mut data: [$elem; $size];
                unsafe {
                    data = std::mem::uninitialized();
                    for i in 0 .. $size {
                        data[i] = self.extract(i)
                    }
                };
                data
            }
        }
    }
}

simd_256_type! { i64x4, i64, 4 }
simd_256_type! { u64x4, u64, 4 }
simd_256_type! { f64x4, f64, 4 }
simd_256_type! { i32x8, i32, 8 }
simd_256_type! { u32x8, u32, 8 }
simd_256_type! { f32x8, f32, 8 }
simd_256_type! { i16x16, i16, 16 }
simd_256_type! { u16x16, u16, 16 }
simd_256_type! { i8x32, i8, 32 }
simd_256_type! { u8x32, u8, 32 }

#[macro_use]
pub mod util;

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
        let x = i32x4(1, 2, 3, 4).as_m128i();
        let y = i32x4(2, 3, 4, 5).as_m128i();

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
    fn basic_i32x4() {
        let x = i32x4(1, 2, 3, 4);
        assert_eq!(x.extract(0), 1);
        assert_eq!(x.extract(1), 2);
        assert_eq!(x.extract(2), 3);
        assert_eq!(x.extract(3), 4);
    }

    #[test]
    fn basic_f32x4() {
        let x = f32x4(1.0, 2.0, 3.0, 4.0);

        let y = x.insert(0, 9.0);
        assert_eq!(x.extract(0), 1.0);
        assert_eq!(y.extract(0), 9.0);
    }

    #[test]
    fn basic_i16x8() {
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
    fn basic_u16x8() {
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
    fn basic_i8x16() {
        let x = i8x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        for i in 0 .. 16 {
            assert_eq!(x.extract(i), (i + 1) as i8);
        }
    }
}
