#![feature(cfg_target_feature)]

extern crate x86intrin;

#[cfg(target_feature = "ssse3")]
mod x {
    use x86intrin::*;

    #[inline(never)]
    fn palignr_2(a: m128i, b: m128i) -> m128i {
        mm_alignr_epi8(a, b, 2)
    }

    pub fn run() {
        let a = mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = mm_setr_epi8(17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);

        println!("{:?}", palignr_2(a, b).as_i8x16().as_array());
    }
}

#[cfg(not(target_feature = "ssse3"))]
mod x {
    pub fn run() {
        println!("build with target_feature=ssse3");
    }
}

fn main() {
    x::run()
}
