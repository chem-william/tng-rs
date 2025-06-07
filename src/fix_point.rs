const MAX31BIT: u32 = 2147483647;
const SIGN32BIT: u32 = 2147483648;

/// Positive double to 32 bit fixed point value
///
/// c version: Ptncg_ud_to_fix_t
pub(crate) fn f64_to_u32_fixed_point(d: f64, max: f64) -> u32 {
    let d_clamped = d.clamp(0.0, max);

    // Scale into [0..u32::MAX] and round
    let scaled = (d_clamped / max) * (u32::MAX as f64);
    if scaled > (u32::MAX as f64) {
        u32::MAX
    } else {
        scaled as u32
    }
}

/// Returns an `u32` whose highest bit is the sign and whose lower 31 bits
/// are the scaled magnitude ([0..2^31-1]).
///
/// c version: Ptngc_d_to_fix_t
pub(crate) fn f64_to_u32_signed_fixed_point(d: f64, max: f64) -> u32 {
    // compute ratio, clamped into [−1.0..1.0]
    let ratio = (d / max).clamp(-1.0, 1.0);

    // magnitude = floor(|ratio| * MAX31BIT), guaranteed ≤ MAX31BIT
    let mag_f = (ratio.abs() * (MAX31BIT as f64)).floor();
    let mag = mag_f as u32; // now in [0..MAX31BIT]

    // if negative, set the sign bit; otherwise leave it as-is
    if ratio.is_sign_negative() {
        mag | SIGN32BIT
    } else {
        mag
    }
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
pub(crate) fn fix_u32_fixed_to_f64(fixed: u32, max: f64) -> f64 {
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
pub(crate) fn f64_to_i32_fixed_pair(d: f64) -> (u32, u32) {
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
        ent_f as u32
    };
    // Set the sign bit if needed
    if sign_flag {
        hi = hi as u32 | SIGN32BIT;
    }

    let lo = f64_to_u32_fixed_point(frac, 1.0);

    (hi, lo)
}

/// Convert two 32 bit integers to a floating point variable
/// -2.1e9 to 2.1e9 and precision to somewhere around 1e-0
///
/// c version: Ptngc_i32x2_to_d
pub(crate) fn u32_fixed_pair_to_f64(hi: u32, lo: u32) -> f64 {
    let negative = (hi & SIGN32BIT) != 0;
    // Mask away the sign bit to get the absolute integer part
    let magnitude_hi = (hi & MAX31BIT) as f64;

    let ent = magnitude_hi;
    let frac = u32_fixed_to_f64(lo, 1.);
    let val = ent + frac;

    if negative { -val } else { val }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    use super::*;

    #[test]
    fn test_double_to_u32_fixed_point() {
        let max = 10.0;
        let tests = [
            -5.0,
            0.0,
            5.0,
            10.0,
            15.0,
            3.421234251,
            4.234,
            8.123618970983,
        ];
        let expected = [
            0, 0, 2147483647, 4294967295, 4294967295, 1469408921, 1818489152, 3489067779,
        ];
        for (&d, exp) in tests.iter().zip(expected) {
            let f = f64_to_u32_fixed_point(d, max);
            assert_eq!(f, exp);
        }
    }

    #[test]
    fn test_f64_to_fix_i32() {
        let max = 10.0;
        let tests = [-15.0, -5.0, 0.0, 5.0, 10.0, 15.0];
        let expected_hi = [
            4294967295, 3221225471, 0, 1073741823, 2147483647, 2147483647,
        ];
        let expected_mag = [
            2147483647, 1073741823, 0, 1073741823, 2147483647, 2147483647,
        ];
        let expected_sign = [-1, -1, 1, 1, 1, 1];
        for (((&d, exp_hi), exp_mag), exp_sign) in tests
            .iter()
            .zip(expected_hi)
            .zip(expected_mag)
            .zip(expected_sign)
        {
            let f = f64_to_u32_signed_fixed_point(d, max);
            let sign = if ((f & SIGN32BIT) as i32) != 0 { -1 } else { 1 };
            let mag = f & MAX31BIT;
            assert_eq!(f, exp_hi);
            assert_eq!(mag, exp_mag);
            assert_eq!(sign, exp_sign);
        }
    }

    #[test]
    fn test_fix_u32_to_ud() {
        let max = 10.0;
        let tests = [0u32, u32::MAX / 2, u32::MAX];
        let expected = [0.0, 5.0, 10.0];
        for (&f, exp_val) in tests.iter().zip(expected) {
            let d = u32_fixed_to_f64(f, max);
            assert_approx_eq!(d, exp_val);
        }
    }

    #[test]
    fn test_fix_i32_to_f64() {
        let max = 10.0;
        let tests = [
            MAX31BIT / 2,
            (MAX31BIT / 2) | SIGN32BIT,
            MAX31BIT,
            MAX31BIT | SIGN32BIT,
        ];
        let expected_fix = [1073741823, 3221225471, 2147483647, 4294967295];
        let expected_val = [5.0, -5.0, 10.0, -10.0];
        for ((&f, exp_fix), exp_val) in tests.iter().zip(expected_fix).zip(expected_val) {
            let d = fix_u32_fixed_to_f64(f, max);
            assert_eq!(exp_fix, f);
            assert_approx_eq!(exp_val, d);
        }
    }

    #[test]
    fn test_f64_to_i32x2_and_back() {
        let tests = [
            -12.345678,
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
            12.345678,
            123456789.0,
        ];
        for &d in &tests {
            let (hi, lo) = f64_to_i32_fixed_pair(d);
            let roundtrip = u32_fixed_pair_to_f64(hi, lo);
            assert_approx_eq!(d, roundtrip);
        }
    }
}
