extern crate x86intrin;

use x86intrin::*;

#[inline(never)]
fn rcp_ps(x: m128) -> m128 {
    mm_rcp_ps(x)
}

#[inline(never)]
fn rcp_ss(x: m128) -> m128 {
    mm_rcp_ss(x)
}

fn main() {
    let x = mm_setr_ps(3.0, 2.0, 3.0, 4.0);

    let x_rcp_ps = rcp_ps(x).as_f32x4();
    let x_rcp_ss = rcp_ss(x).as_f32x4();

    println!("x_rcp_ps(x) = {} {} {} {}", x_rcp_ps.extract(0), x_rcp_ps.extract(1), x_rcp_ps.extract(2), x_rcp_ps.extract(3));
    println!("x_rcp_ss(x) = {} {} {} {}", x_rcp_ss.extract(0), x_rcp_ss.extract(1), x_rcp_ss.extract(2), x_rcp_ss.extract(3));
}
