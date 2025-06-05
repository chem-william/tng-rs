const MAX31BIT: i32 = 2147483647;
const SIGN32BIT: i32 = 2147483648;

/// Positive double to 32 bit fixed point value
///
/// c version: Ptncg_ud_to_fix_t
pub(crate) fn double_to_u32_fixed_point(d: f64, max: f64) -> u32 {
    let d_clamped = d.clamp(0.0, max);

    // Scale into [0..u32::MAX] and round
    let scaled = (d_clamped / max) * (u32::MAX as f64);
    if scaled >= (u32::MAX as f64) {
        u32::MAX
    } else {
        scaled.round() as u32
    }
}

/// Returns an `i32` whose highest bit is the sign and whose lower 31 bits
/// are the scaled magnitude ([0..2^31-1]).
///
/// c version: Ptngc_d_to_fix_t
pub(crate) fn double_to_i32_signed_fixed_point(d: f64, max: f64) -> i32 {
    // Handle sign
    let mut d_abs = d;
    let mut sign_flag = false;
    if d_abs.is_sign_negative() {
        sign_flag = true;
        d_abs = -d_abs;
    }

    let clamped = if d_abs > max { max } else { d_abs };

    // Scale to [0..MAX31BIT], truncating (like C cast from double to int)
    let mut val = ((clamped / max) * (MAX31BIT as f64)) as i32;

    // Guard against rounding overshoot
    if val > MAX31BIT {
        val = MAX31BIT;
    }

    // If negative, set the sign bit
    if sign_flag {
        val |= SIGN32BIT;
    }

    val
}

/// 32 bit fixed point value to positive double
///
/// c version: Ptncg_fix_t_to_ud
pub(crate) fn u32_fixed_to_f64(fixed: u32, max: f64) -> f64 {
    (fixed as f64) * (max / (u32::MAX as f64))
}

/// Signed 32 bit fixed point value to double
///
/// c version: Ptngc_fix_t_to_d
pub(crate) fn i32_fixed_to_f64(fixed: i32, max: f64) -> f64 {
    // Extract sign bit
    let negative = (fixed & SIGN32BIT) != 0;
    // Mask off the sign bit to get the magnitude in [0..MAX31BIT]
    let magnitude = (fixed & MAX31BIT) as f64;
    // Scale that magnitude into [0..max]
    let scaled = magnitude * (max / (MAX31BIT as f64));
    if negative { -scaled } else { scaled }
}

/// Convert a floating point variable to two 32 bit integers with range
/// -2.1e9 to 2.1e9 and precision somewhere around 1e-9
///
/// c version: Ptngc_d_to_i32x2
pub(crate) fn f64_to_i32_fixed_pair(d: f64) -> (i32, u32) {
    // Handle sign & work with absolute value
    let mut abs = d;
    let mut sign_flag = false;
    if abs.is_sign_negative() {
        sign_flag = true;
        abs = -abs;
    }

    // Split into integer part (ent) and fractional part
    let ent_f = abs.floor();
    let frac = abs - ent_f;

    // Clamp the integer part at MAX31BIT if it’s too big, then cast
    let mut hi = if ent_f > (MAX31BIT as f64) {
        MAX31BIT
    } else {
        ent_f as i32
    };
    // Set the sign bit if needed
    if sign_flag {
        hi |= SIGN32BIT;
    }

    let lo = double_to_u32_fixed_point(frac, 1.0);

    (hi, lo)
}

/// Convert two 32 bit integers to a floating point variable
/// -2.1e9 to 2.1e9 and precision to somewhere around 1e-0
pub(crate) fn i32_fixed_pair_to_f64(hi: i32, lo: u32) -> f64 {
    let negative = (hi & SIGN32BIT) != 0;
    // Mask away the sign bit to get the absolute integer part
    let magnitude_hi = (hi & MAX31BIT) as f64;

    let ent = magnitude_hi as f64;
    let frac = u32_fixed_to_f64(lo, 1.);
    let val = ent + frac;

    if negative { -val } else { val }
}
