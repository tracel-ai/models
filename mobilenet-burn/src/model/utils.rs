use core::cmp::max;

pub fn make_divisble(v: f32, divisor: i32) -> i32 {
    let min_value = divisor;
    let new_v = (v + divisor as f32 / 2.0) as i32 / divisor * divisor;
    let new_v = max(new_v, min_value);
    // make sure that round down does not go down by more than 10%
    if new_v < (0.9 * v) as i32 {
        new_v + divisor
    } else {
        new_v
    }
}
