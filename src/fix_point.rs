use std::ops::{BitAnd, BitOr, BitOrAssign, Shl};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixT(u32);

impl FixT {
    pub(crate) const MAX32BIT: u32 = 4_294_967_295; // (1 << 32) - 1
    pub(crate) const MAX31BIT: u32 = 2_147_483_647; // (1 << 31) - 1
    pub(crate) const SIGN32BIT: u32 = 2_147_483_648; // 1 << 31

    /// Positive `f64` to 32 bit fixed point value
    ///
    /// C API: `Ptncg_ud_to_fix_t`
    pub fn from_f64_unsigned(d: f64, max: f64) -> Self {
        let d_clamped = d.clamp(0.0, max);

        // Scale into [0..u32::MAX] and round
        let scaled = (d_clamped / max) * f64::from(u32::MAX);
        if scaled > f64::from(u32::MAX) {
            FixT(u32::MAX)
        } else {
            FixT(scaled as u32)
        }
    }

    /// `f64` to signed 32 bit fixed point value
    ///
    /// C API: `Ptngc_d_to_fix_t`
    pub fn from_f64_signed(d: f64, max: f64) -> Self {
        // compute ratio, clamped into [−1.0..1.0]
        let ratio = (d / max).clamp(-1.0, 1.0);

        // magnitude = floor(|ratio| * MAX31BIT), guaranteed ≤ MAX31BIT
        let mag_f = (ratio.abs() * f64::from(Self::MAX31BIT)).floor();
        let mag = mag_f as u32; // now in [0..MAX31BIT]

        // if negative, set the sign bit; otherwise leave it as-is
        if ratio.is_sign_negative() {
            FixT(mag | Self::SIGN32BIT)
        } else {
            FixT(mag)
        }
    }

    /// 32 bit fixed point value to positive `f64`
    ///
    /// C API: `Ptncg_fix_t_to_ud`
    pub(crate) fn to_f64_unsigned(self, max: f64) -> f64 {
        f64::from(self.0) * (max / f64::from(u32::MAX))
    }

    /// Signed 32 bit fixed point value to `f64`
    ///
    /// C API: `Ptngc_fix_t_to_d`
    pub(crate) fn to_f64_signed(self, max: f64) -> f64 {
        f64::from(self.0) * (max / f64::from(Self::MAX32BIT))
    }
}

impl From<u32> for FixT {
    /// Will wrap around to 0 if bigger than [`FixT::MAX31BIT`] (2147483647)
    fn from(value: u32) -> Self {
        Self(value & FixT::MAX31BIT)
    }
}

impl From<FixT> for u32 {
    fn from(f: FixT) -> u32 {
        f.0
    }
}

/// This is very much like C
impl From<FixT> for i32 {
    fn from(value: FixT) -> Self {
        value.0 as i32
    }
}

impl BitAnd for FixT {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        FixT::from(self.0 & rhs.0)
    }
}

impl Shl for FixT {
    type Output = Self;

    fn shl(self, rhs: Self) -> Self::Output {
        FixT::from(self.0 << rhs.0)
    }
}

impl BitOr for FixT {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        FixT::from(self.0 | rhs.0)
    }
}

impl BitOrAssign for FixT {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = FixT::from(self.0 | rhs.0);
    }
}

impl std::fmt::Display for FixT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Convert a floating point variable to two 32 bit integers with range
/// -2.1e9 to 2.1e9 and precision somewhere around 1e-9
///
/// C API: `Ptngc_d_to_i32x2`
pub(crate) fn f64_to_fixt_pair(d: f64) -> (FixT, FixT) {
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
    let mut hi = if ent_f > f64::from(FixT::MAX31BIT) {
        FixT::MAX31BIT
    } else {
        ent_f as u32
    };
    // Set the sign bit if needed
    if sign_flag {
        hi |= FixT::SIGN32BIT;
    }

    let lo = FixT::from_f64_unsigned(frac, 1.0);

    (FixT(hi), lo)
}

/// Convert two 32 bit integers to a floating point variable
/// -2.1e9 to 2.1e9 and precision to somewhere around 1e-0
///
/// C API: `Ptngc_i32x2_to_d`
pub(crate) fn fixt_pair_to_f64(hi: FixT, lo: FixT) -> f64 {
    let negative = (u32::from(hi) & FixT::SIGN32BIT) != 0;
    // Mask away the sign bit to get the absolute integer part
    let magnitude_hi = if negative {
        u32::from(hi) & FixT::MAX31BIT
    } else {
        u32::from(hi)
    };

    let ent = f64::from(magnitude_hi);
    let frac = lo.to_f64_signed(1.0);
    let val = ent + frac;

    if negative { -val } else { val }
}
