use super::*;
use super::{simd_gt};

extern {
    #[link_name = "llvm.x86.sse42.crc32.32.8"]
    fn sse42_crc32_32_8(a: u32, b: u8) -> u32;
    #[link_name = "llvm.x86.sse42.crc32.32.16"]
    fn sse42_crc32_32_16(a: u32, b: u16) -> u32;
    #[link_name = "llvm.x86.sse42.crc32.32.32"]
    fn sse42_crc32_32_32(a: u32, b: u32) -> u32;
    #[link_name = "llvm.x86.sse42.crc32.64.64"]
    fn sse42_crc32_64_64(a: u64, b: u64) -> u64;
}

// int _mm_cmpestra (__m128i a, int la, __m128i b, int lb, const int imm8)
// pcmpestri
// int _mm_cmpestrc (__m128i a, int la, __m128i b, int lb, const int imm8)
// pcmpestri
// int _mm_cmpestri (__m128i a, int la, __m128i b, int lb, const int imm8)
// pcmpestrm
// __m128i _mm_cmpestrm (__m128i a, int la, __m128i b, int lb, const int imm8)
// pcmpestri
// int _mm_cmpestro (__m128i a, int la, __m128i b, int lb, const int imm8)
// pcmpestri
// int _mm_cmpestrs (__m128i a, int la, __m128i b, int lb, const int imm8)
// pcmpestri
// int _mm_cmpestrz (__m128i a, int la, __m128i b, int lb, const int imm8)

// pcmpgtq
// __m128i _mm_cmpgt_epi64 (__m128i a, __m128i b)
#[inline]
pub fn mm_cmpgt_epi64(a: m128i, b: m128i) -> m128i {
    let x: i64x2 = unsafe { simd_gt(a.as_i64x2(), b.as_i64x2()) };
    x.as_m128i()
}

// pcmpistri
// int _mm_cmpistra (__m128i a, __m128i b, const int imm8)
// pcmpistri
// int _mm_cmpistrc (__m128i a, __m128i b, const int imm8)
// pcmpistri
// int _mm_cmpistri (__m128i a, __m128i b, const int imm8)
// pcmpistrm
// __m128i _mm_cmpistrm (__m128i a, __m128i b, const int imm8)
// pcmpistri
// int _mm_cmpistro (__m128i a, __m128i b, const int imm8)
// pcmpistri
// int _mm_cmpistrs (__m128i a, __m128i b, const int imm8)
// pcmpistri
// int _mm_cmpistrz (__m128i a, __m128i b, const int imm8)

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
}
