use super::*;
use super::{simd_shuffle4};

extern "platform-intrinsic" {
    fn x86_mm_addsub_pd(x: m128d, y: m128d) -> m128d;
    fn x86_mm_addsub_ps(x: m128, y: m128) -> m128;
    fn x86_mm_hadd_pd(x: m128d, y: m128d) -> m128d;
    fn x86_mm_hadd_ps(x: m128, y: m128) -> m128;
    fn x86_mm_hsub_pd(x: m128d, y: m128d) -> m128d;
    fn x86_mm_hsub_ps(x: m128, y: m128) -> m128;

    fn simd_shuffle2<T, U>(x: T, y: T, idx: [u32; 2]) -> U;
}

// addsubpd
// __m128d _mm_addsub_pd (__m128d a, __m128d b)
#[inline]
pub fn mm_addsub_pd(a: m128d, b: m128d) -> m128d {
    unsafe { x86_mm_addsub_pd(a, b) }
}

// addsubps
// __m128 _mm_addsub_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_addsub_ps(a: m128, b: m128) -> m128 {
    unsafe { x86_mm_addsub_ps(a, b) }
}

// haddpd
// __m128d _mm_hadd_pd (__m128d a, __m128d b)
#[inline]
pub fn mm_hadd_pd(a: m128d, b: m128d) -> m128d {
    unsafe { x86_mm_hadd_pd(a, b) }
}

// haddps
// __m128 _mm_hadd_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_hadd_ps(a: m128, b: m128) -> m128 {
    unsafe { x86_mm_hadd_ps(a, b) }
}

// hsubpd
// __m128d _mm_hsub_pd (__m128d a, __m128d b)
#[inline]
pub fn mm_hsub_pd(a: m128d, b: m128d) -> m128d {
    unsafe { x86_mm_hsub_pd(a, b) }
}

// hsubps
// __m128 _mm_hsub_ps (__m128 a, __m128 b)
#[inline]
pub fn mm_hsub_ps(a: m128, b: m128) -> m128 {
    unsafe { x86_mm_hsub_ps(a, b) }
}

// lddqu
// __m128i _mm_lddqu_si128 (__m128i const* mem_addr)
// movddup
// __m128d _mm_loaddup_pd (double const* mem_addr)

// movddup
// __m128d _mm_movedup_pd (__m128d a)
#[inline]
pub fn mm_movedup_pd(a: m128d) -> m128d {
    let a64 = a.as_f64x2();
    unsafe { simd_shuffle2(a64, a64, [0, 0]) }
}

// movshdup
// __m128 _mm_movehdup_ps (__m128 a)
#[inline]
pub fn mm_movehdup_ps(a: m128) -> m128 {
    let a32 = a.as_f32x4();
    unsafe { simd_shuffle4(a32, a32, [1, 1, 3, 3]) }
}

// movsldup
// __m128 _mm_moveldup_ps (__m128 a)
#[inline]
pub fn mm_moveldup_ps(a: m128) -> m128 {
    let a32 = a.as_f32x4();
    unsafe { simd_shuffle4(a32, a32, [0, 0, 2, 2]) }
}

#[cfg(test)]
mod test {
    use super::super::*;

    #[test]
    fn test_arith_ps() {
        let x = mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let y = mm_setr_ps(2.0, 1.0, 2.0, 4.0);

        assert_eq!(mm_addsub_ps(x, y).as_f32x4().as_array(), [-1.0, 3.0, 1.0, 8.0]);
        assert_eq!(mm_hadd_ps(x, y).as_f32x4().as_array(), [3.0, 7.0, 3.0, 6.0]);
        assert_eq!(mm_hsub_ps(x, y).as_f32x4().as_array(), [-1.0, -1.0, 1.0, -2.0]);
    }

    #[test]
    fn test_arith_pd() {
        let x = mm_setr_pd(1.0, 2.0);
        let y = mm_setr_pd(3.0, 5.0);

        assert_eq!(mm_addsub_pd(x, y).as_f64x2().as_array(), [-2.0, 7.0]);
        assert_eq!(mm_hadd_pd(x, y).as_f64x2().as_array(), [3.0, 8.0]);
        assert_eq!(mm_hsub_pd(x, y).as_f64x2().as_array(), [-1.0, -2.0]);
    }

    #[test]
    fn test_move_pd() {
        let x = mm_setr_pd(1.0, 2.0);
        assert_eq!(mm_movedup_pd(x).as_f64x2().as_array(), [1.0, 1.0]);
    }

    #[test]
    fn test_move_ps() {
        let x = mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        assert_eq!(mm_movehdup_ps(x).as_f32x4().as_array(), [2.0, 2.0, 4.0, 4.0]);
        assert_eq!(mm_moveldup_ps(x).as_f32x4().as_array(), [1.0, 1.0, 3.0, 3.0]);
    }
}
