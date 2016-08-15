use super::*;
use super::{simd_gt};

extern {
    #[link_name = "llvm.x86.sse42.pcmpestrm128"]
    fn sse42_pcmpestrm128(a: i8x16, b: i32, c: i8x16, d: i32, e: u8) -> i8x16;
    #[link_name = "llvm.x86.sse42.pcmpestri128"]
    fn sse42_pcmpestri128(a: i8x16, b: i32, c: i8x16, d: i32, e: u8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpestria128"]
    fn sse42_pcmpestria128(a: i8x16, b: i32, c: i8x16, d: i32, e: u8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpestric128"]
    fn sse42_pcmpestric128(a: i8x16, b: i32, c: i8x16, d: i32, e: u8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpestrio128"]
    fn sse42_pcmpestrio128(a: i8x16, b: i32, c: i8x16, d: i32, e: u8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpestris128"]
    fn sse42_pcmpestris128(a: i8x16, b: i32, c: i8x16, d: i32, e: u8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpestriz128"]
    fn sse42_pcmpestriz128(a: i8x16, b: i32, c: i8x16, d: i32, e: u8) -> i32;

    #[link_name = "llvm.x86.sse42.pcmpistrm128"]
    fn sse42_pcmpistrm128(a: i8x16, b: i8x16, c: u8) -> i8x16;
    #[link_name = "llvm.x86.sse42.pcmpistri128"]
    fn sse42_pcmpistri128(a: i8x16, b: i8x16, c: u8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpistria128"]
    fn sse42_pcmpistria128(a: i8x16, b: i8x16, c: u8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpistric128"]
    fn sse42_pcmpistric128(a: i8x16, b: i8x16, c: u8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpistrio128"]
    fn sse42_pcmpistrio128(a: i8x16, b: i8x16, c: u8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpistris128"]
    fn sse42_pcmpistris128(a: i8x16, b: i8x16, c: u8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpistriz128"]
    fn sse42_pcmpistriz128(a: i8x16, b: i8x16, c: u8) -> i32;

    #[link_name = "llvm.x86.sse42.crc32.32.8"]
    fn sse42_crc32_32_8(a: u32, b: u8) -> u32;
    #[link_name = "llvm.x86.sse42.crc32.32.16"]
    fn sse42_crc32_32_16(a: u32, b: u16) -> u32;
    #[link_name = "llvm.x86.sse42.crc32.32.32"]
    fn sse42_crc32_32_32(a: u32, b: u32) -> u32;
    #[link_name = "llvm.x86.sse42.crc32.64.64"]
    fn sse42_crc32_64_64(a: u64, b: u64) -> u64;
}

// #define _SIDD_UBYTE_OPS                 0x00
// #define _SIDD_UWORD_OPS                 0x01
// #define _SIDD_SBYTE_OPS                 0x02
// #define _SIDD_SWORD_OPS                 0x03
pub const SIDD_UBYTE_OPS: i32 = 0x00;
pub const SIDD_UWORD_OPS: i32 = 0x01;
pub const SIDD_SBYTE_OPS: i32 = 0x02;
pub const SIDD_SWORD_OPS: i32 = 0x03;

// #define _SIDD_CMP_EQUAL_ANY             0x00
// #define _SIDD_CMP_RANGES                0x04
// #define _SIDD_CMP_EQUAL_EACH            0x08
// #define _SIDD_CMP_EQUAL_ORDERED         0x0c
pub const SIDD_CMP_EQUAL_ANY: i32 = 0x00;
pub const SIDD_CMP_RANGES: i32 = 0x04;
pub const SIDD_CMP_EQUAL_EACH: i32 = 0x08;
pub const SIDD_CMP_EQUAL_ORDERED: i32 = 0x0c;

//#define _SIDD_POSITIVE_POLARITY         0x00
//#define _SIDD_NEGATIVE_POLARITY         0x10
//#define _SIDD_MASKED_POSITIVE_POLARITY  0x20
//#define _SIDD_MASKED_NEGATIVE_POLARITY  0x30
pub const SIDD_POSITIVE_POLARITY: i32 = 0x00;
pub const SIDD_NEGATIVE_POLARITY: i32 = 0x10;
pub const SIDD_MASKED_POSITIVE_POLARITY: i32 = 0x20;
pub const SIDD_MASKED_NEGATIVE_POLARITY: i32 = 0x30;

// #define _SIDD_LEAST_SIGNIFICANT         0x00
// #define _SIDD_MOST_SIGNIFICANT          0x40
pub const SIDD_LEAST_SIGNIFICANT: i32 = 0x00;
pub const SIDD_MOST_SIGNIFICANT: i32 = 0x40;

// #define _SIDD_BIT_MASK                  0x00
// #define _SIDD_UNIT_MASK                 0x40
pub const SIDD_BIT_MASK: i8 = 0x00;
pub const SIDD_UNIT_MASK: i8 = 0x40;

// int _mm_cmpestra (__m128i a, int la, __m128i b, int lb, const int imm8)
#[inline]
pub fn mm_cmpestra(a: m128i, la: i32, b: m128i, lb: i32, imm8: i32) -> i32 {
    fn_imm8_arg4!(sse42_pcmpestria128, a.as_i8x16(), la, b.as_i8x16(), lb, imm8 as u8)
}

// pcmpestri
// int _mm_cmpestrc (__m128i a, int la, __m128i b, int lb, const int imm8)
#[inline]
pub fn mm_cmpestrc(a: m128i, la: i32, b: m128i, lb: i32, imm8: i32) -> i32 {
    fn_imm8_arg4!(sse42_pcmpestric128, a.as_i8x16(), la, b.as_i8x16(), lb, imm8 as u8)
}

// pcmpestri
// int _mm_cmpestri (__m128i a, int la, __m128i b, int lb, const int imm8)
#[inline]
pub fn mm_cmpestri(a: m128i, la: i32, b: m128i, lb: i32, imm8: i32) -> i32 {
    fn_imm8_arg4!(sse42_pcmpestri128, a.as_i8x16(), la, b.as_i8x16(), lb, imm8 as u8)
}

// pcmpestrm
// __m128i _mm_cmpestrm (__m128i a, int la, __m128i b, int lb, const int imm8)
#[inline]
pub fn mm_cmpestrm(a: m128i, la: i32, b: m128i, lb: i32, imm8: i32) -> m128i {
    fn_imm8_arg4!(sse42_pcmpestrm128, a.as_i8x16(), la, b.as_i8x16(), lb, imm8 as u8).as_m128i()
}

// pcmpestri
// int _mm_cmpestro (__m128i a, int la, __m128i b, int lb, const int imm8)
#[inline]
pub fn mm_cmpestro(a: m128i, la: i32, b: m128i, lb: i32, imm8: i32) -> i32 {
    fn_imm8_arg4!(sse42_pcmpestrio128, a.as_i8x16(), la, b.as_i8x16(), lb, imm8 as u8)
}

// pcmpestri
// int _mm_cmpestrs (__m128i a, int la, __m128i b, int lb, const int imm8)
#[inline]
pub fn mm_cmpestrs(a: m128i, la: i32, b: m128i, lb: i32, imm8: i32) -> i32 {
    fn_imm8_arg4!(sse42_pcmpestris128, a.as_i8x16(), la, b.as_i8x16(), lb, imm8 as u8)
}

// pcmpestri
// int _mm_cmpestrz (__m128i a, int la, __m128i b, int lb, const int imm8)
#[inline]
pub fn mm_cmpestrz(a: m128i, la: i32, b: m128i, lb: i32, imm8: i32) -> i32 {
    fn_imm8_arg4!(sse42_pcmpestriz128, a.as_i8x16(), la, b.as_i8x16(), lb, imm8 as u8)
}

// pcmpgtq
// __m128i _mm_cmpgt_epi64 (__m128i a, __m128i b)
#[inline]
pub fn mm_cmpgt_epi64(a: m128i, b: m128i) -> m128i {
    let x: i64x2 = unsafe { simd_gt(a.as_i64x2(), b.as_i64x2()) };
    x.as_m128i()
}

// pcmpistri
// int _mm_cmpistra (__m128i a, __m128i b, const int imm8)
#[inline]
pub fn mm_cmpistra(a: m128i, b: m128i, imm8: i32) -> i32 {
    fn_imm8_arg2!(sse42_pcmpistria128, a.as_i8x16(), b.as_i8x16(), imm8 as u8)
}

// pcmpistri
// int _mm_cmpistrc (__m128i a, __m128i b, const int imm8)
#[inline]
pub fn mm_cmpistrc(a: m128i, b: m128i, imm8: i32) -> i32 {
    fn_imm8_arg2!(sse42_pcmpistric128, a.as_i8x16(), b.as_i8x16(), imm8 as u8)
}

// pcmpistri
// int _mm_cmpistri (__m128i a, __m128i b, const int imm8)
#[inline]
pub fn mm_cmpistri(a: m128i, b: m128i, imm8: i32) -> i32 {
    fn_imm8_arg2!(sse42_pcmpistri128, a.as_i8x16(), b.as_i8x16(), imm8 as u8)
}

// pcmpistrm
// __m128i _mm_cmpistrm (__m128i a, __m128i b, const int imm8)
#[inline]
pub fn mm_cmpistrm(a: m128i, b: m128i, imm8: i32) -> m128i {
    fn_imm8_arg2!(sse42_pcmpistrm128, a.as_i8x16(), b.as_i8x16(), imm8 as u8).as_m128i()
}

// pcmpistri
// int _mm_cmpistro (__m128i a, __m128i b, const int imm8)
#[inline]
pub fn mm_cmpistro(a: m128i, b: m128i, imm8: i32) -> i32 {
    fn_imm8_arg2!(sse42_pcmpistrio128, a.as_i8x16(), b.as_i8x16(), imm8 as u8)
}

// pcmpistri
// int _mm_cmpistrs (__m128i a, __m128i b, const int imm8)
#[inline]
pub fn mm_cmpistrs(a: m128i, b: m128i, imm8: i32) -> i32 {
    fn_imm8_arg2!(sse42_pcmpistris128, a.as_i8x16(), b.as_i8x16(), imm8 as u8)
}

// pcmpistri
// int _mm_cmpistrz (__m128i a, __m128i b, const int imm8)
#[inline]
pub fn mm_cmpistrz(a: m128i, b: m128i, imm8: i32) -> i32 {
    fn_imm8_arg2!(sse42_pcmpistriz128, a.as_i8x16(), b.as_i8x16(), imm8 as u8)
}

// crc32
// unsigned int _mm_crc32_u16 (unsigned int crc, unsigned short v)
#[inline]
pub fn mm_crc32_u16(crc: u32, v: u16) -> u32 {
    unsafe { sse42_crc32_32_16(crc, v) }
}

// crc32
// unsigned int _mm_crc32_u32 (unsigned int crc, unsigned int v)
#[inline]
pub fn mm_crc32_u32(crc: u32, v: u32) -> u32 {
    unsafe { sse42_crc32_32_32(crc, v) }
}

// crc32
// unsigned __int64 _mm_crc32_u64 (unsigned __int64 crc, unsigned __int64 v)
#[inline]
pub fn mm_crc32_u64(crc: u64, v: u64) -> u64 {
    unsafe { sse42_crc32_64_64(crc, v) }
}

// crc32
// unsigned int _mm_crc32_u8 (unsigned int crc, unsigned char v)
#[inline]
pub fn mm_crc32_u8(crc: u32, v: u8) -> u32 {
    unsafe { sse42_crc32_32_8(crc, v) }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test_crc() {
        assert_eq!(mm_crc32_u8(1, 100), 1412925310);
        assert_eq!(mm_crc32_u16(1, 1000), 3870914500);
        assert_eq!(mm_crc32_u32(1, 50000), 971731851);
        assert_eq!(mm_crc32_u64(0x000011115555AAAA, 0x88889999EEEE3333), 0x0000000016F57621);
    }

    #[test]
    fn test_cmpgt() {
        let a = i64x2(1, 2).as_m128i();
        let b = i64x2(1, 1).as_m128i();

        assert_eq!(mm_cmpgt_epi64(a, b).as_i64x2().as_array(), [0, !0]);
    }

    #[test]
    fn test_cmpestr() {
        // NOTE: SIDD_LEAST_SIGNIFICANT sets the same bit as SIDD_BIT_MASK
        let mode: i32 = SIDD_UWORD_OPS | SIDD_CMP_EQUAL_EACH | SIDD_LEAST_SIGNIFICANT;

        let mut a = mm_set1_epi16(0xCCCCu16 as i16);
        let b = mm_set1_epi16(0x3333u16 as i16);

        assert_eq!(mm_cmpestra(a, 8, b, -8, mode), 1);
        assert_eq!(mm_cmpestrc(a, 8, b, 8, mode), 0);

        a = a.as_u16x8().insert(7, 0x3333).as_m128i();
        a = a.as_u16x8().insert(5, 0x3333).as_m128i();

        assert_eq!(mm_cmpestri(a, 8, b, 8, mode), 5);

        // NOTE: mode has SIDD_LEAST_SIGNIFICANT set which equals SIDD_BIT_MASK
        let m = mm_cmpestrm(a, 8, b, 8, mode);
        assert_eq!(m.as_i64x2().as_array(), [0xa0, 0x00]);

        assert_eq!(mm_cmpestro(a, 8, b, 8, mode), 0);
        a = a.as_u16x8().insert(0, 0x3333).as_m128i();

        assert_eq!(mm_cmpestro(a, 8, b, 8, mode), 1);
        assert_eq!(mm_cmpestrs(a, 8, b, 8, mode), 0);
        assert_eq!(mm_cmpestrs(a, 7, b, 8, mode), 1);
        assert_eq!(mm_cmpestrz(a, 8, b, 8, mode), 0);
        assert_eq!(mm_cmpestrz(a, 8, b, 7, mode), 1);
    }
}
