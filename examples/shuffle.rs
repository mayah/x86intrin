extern crate x86intrin;

use x86intrin::*;

#[inline(never)]
fn shuffle(x: m128, y: m128, imm8: u32) -> m128 {
    mm_shuffle_ps(x, y, imm8)
}

fn main() {
    let x = mm_setr_ps(1.0, 2.0, 3.0, 4.0);
    let y = mm_setr_ps(5.0, 6.0, 7.0, 8.0);

    let z = shuffle(x, y, 0x3F).as_f32x4().as_array();

    println!("z = {} {} {} {}", z[0], z[1], z[2], z[3]);
}
