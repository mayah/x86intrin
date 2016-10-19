extern "platform-intrinsic" {
    fn x86_bmi2_bzhi_32(x: u32, y: u32) -> u32;
    fn x86_bmi2_bzhi_64(x: u64, y: u64) -> u64;
    fn x86_bmi2_pdep_32(x: u32, y: u32) -> u32;
    fn x86_bmi2_pdep_64(x: u64, y: u64) -> u64;
    fn x86_bmi2_pext_32(x: u32, y: u32) -> u32;
    fn x86_bmi2_pext_64(x: u64, y: u64) -> u64;
}

#[inline]
pub fn bzhi_u32(a: u32, index: u32) -> u32 {
    unsafe { x86_bmi2_bzhi_32(a, index) }
}

#[inline]
pub fn bzhi_u64(a: u64, index: u64) -> u64 {
    unsafe { x86_bmi2_bzhi_64(a, index) }
}

#[inline]
pub fn pdep_u32(a: u32, b: u32) -> u32 {
    unsafe { x86_bmi2_pdep_32(a, b) }
}

#[inline]
pub fn pdep_u64(a: u64, b: u64) -> u64 {
    unsafe { x86_bmi2_pdep_64(a, b) }
}

pub fn pext_u32(a: u32, b: u32) -> u32 {
    unsafe { x86_bmi2_pext_32(a, b) }
}

#[inline]
pub fn pext_u64(a: u64, b: u64) -> u64 {
    unsafe { x86_bmi2_pext_64(a, b) }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test_bzhi() {
        assert_eq!(bzhi_u32(0xFFFFFFFF, 3), 0x7);
        assert_eq!(bzhi_u32(0xFFFF7777, 20), 0xF7777);
        assert_eq!(bzhi_u64(0xFFFFFFFFFFFFFFFF, 3), 0x7);
        assert_eq!(bzhi_u64(0xFFFFFFFF77777777, 36), 0xF77777777);
    }

    #[test]
    fn test_pdep() {
        assert_eq!(pdep_u32(0xAA, 0xC7), 0x42);
        assert_eq!(pdep_u64(0xAA, 0xC7), 0x42);
    }

    #[test]
    fn test_pext() {
        assert_eq!(pext_u32(0xAA, 0xC7), 0x12);
        assert_eq!(pext_u64(0xAA, 0xC7), 0x12);
    }
}
