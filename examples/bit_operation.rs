extern crate x86intrin;

use x86intrin::*;

#[inline(never)]
fn xy_and(x: m128i, y: m128i) -> m128i {
    mm_and_si128(x, y)
}

#[inline(never)]
fn xy_or(x: m128i, y: m128i) -> m128i {
    mm_or_si128(x, y)
}

#[inline(never)]
fn xy_xor(x: m128i, y: m128i) -> m128i {
    mm_xor_si128(x, y)
}

#[inline(never)]
fn xy_andnot(x: m128i, y: m128i) -> m128i {
    mm_andnot_si128(x, y)
}

fn main() {
    let x = mm_setr_epi32(1, 2, 3, 4);
    let y = mm_setr_epi32(1, 3, 7, 15);

    let xya = xy_and(x, y).as_i32x4();
    let xyo = xy_or(x, y).as_i32x4();
    let xyx = xy_xor(x, y).as_i32x4();
    let xyn = xy_andnot(x, y).as_i32x4();

    println!(" x & y = {} {} {} {}", xya.extract(0), xya.extract(1), xya.extract(2), xya.extract(3));
    println!(" x | y = {} {} {} {}", xyo.extract(0), xyo.extract(1), xyo.extract(2), xyo.extract(3));
    println!(" x ^ y = {} {} {} {}", xyx.extract(0), xyx.extract(1), xyx.extract(2), xyx.extract(3));
    println!("!x & y = {} {} {} {}", xyn.extract(0), xyn.extract(1), xyn.extract(2), xyn.extract(3));
}
