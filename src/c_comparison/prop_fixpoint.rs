use crate::{
    ffi,
    fix_point::{FixT, f64_to_fixt_pair, fixt_pair_to_f64},
};
use proptest::prelude::*;

fn fix_t_to_u32(v: ffi::FixT) -> u32 {
    v as u32
}

proptest! {
    /// from_f64_unsigned must match C's Ptngc_ud_to_fix_t.
    /// d is generated relative to max so we proportionally cover the in-range
    /// interior [0, max], below-zero, and above-max (all clamp) cases.
    #[test]
    fn prop_unsigned_matches_c(
        max in 0.01f64..=1000.0f64,
        d_ratio in -0.5f64..=1.5f64,
    ) {
        let d = d_ratio * max;
        let rust: u32 = FixT::from_f64_unsigned(d, max).into();
        let c = fix_t_to_u32(unsafe { ffi::Ptngc_ud_to_fix_t(d, max) });
        prop_assert_eq!(rust, c, "d={}, max={}", d, max);
    }

    /// from_f64_signed must match C's Ptngc_d_to_fix_t.
    /// d is generated relative to max so interior [-max, max] and both
    /// out-of-range clamp directions are proportionally covered.
    #[test]
    fn prop_signed_matches_c(
        max in 0.01f64..=1000.0f64,
        d_ratio in -1.5f64..=1.5f64,
    ) {
        let d = d_ratio * max;
        let rust: u32 = FixT::from_f64_signed(d, max).into();
        let c = fix_t_to_u32(unsafe { ffi::Ptngc_d_to_fix_t(d, max) });
        prop_assert_eq!(rust, c, "d={}, max={}", d, max);
    }

    /// Unsigned encode then decode stays within one quantum of the original.
    #[test]
    fn prop_unsigned_roundtrip(
        max in 0.01f64..=1000.0f64,
        d_ratio in 0.0f64..=1.0f64,
    ) {
        let d = d_ratio * max;
        let encoded = FixT::from_f64_unsigned(d, max);
        let decoded = encoded.to_f64_unsigned(max);
        // Truncation means decoded <= d; error is at most one quantum.
        let quantum = max / f64::from(u32::MAX);
        prop_assert!(
            d - decoded >= -1e-12 && d - decoded <= quantum + 1e-12,
            "roundtrip error: d={d}, decoded={decoded}, quantum={quantum}",
        );
    }

    /// f64_to_fixt_pair must match C's Ptngc_d_to_i32x2.
    /// Weighted mix: 3/4 of cases use small values (common MD range),
    /// 1/4 use values up to MAX31BIT, so small-value precision is well covered.
    #[test]
    fn prop_fixt_pair_matches_c(
        d_abs in prop_oneof![
            3 => 0.0f64..1000.0f64,
            1 => 1000.0f64..=(f64::from(FixT::MAX31BIT)),
        ],
        negative in proptest::bool::ANY,
    ) {
        let d = if negative { -d_abs } else { d_abs };

        let (rust_hi, rust_lo) = f64_to_fixt_pair(d);
        let rust_hi_u32: u32 = rust_hi.into();
        let rust_lo_u32: u32 = rust_lo.into();

        let mut c_hi: ffi::FixT = 0;
        let mut c_lo: ffi::FixT = 0;
        unsafe { ffi::Ptngc_d_to_i32x2(d, &raw mut c_hi, &raw mut c_lo) };

        prop_assert_eq!(rust_hi_u32, fix_t_to_u32(c_hi), "hi mismatch for d={}", d);
        prop_assert_eq!(rust_lo_u32, fix_t_to_u32(c_lo), "lo mismatch for d={}", d);
    }

    /// fixt_pair roundtrip: f64 -> (hi, lo) -> f64 recovers.
    /// Same weighted distribution as prop_fixt_pair_matches_c.
    #[test]
    fn prop_fixt_pair_roundtrip(
        d_abs in prop_oneof![
            3 => 0.0f64..1000.0f64,
            1 => 1000.0f64..=(f64::from(FixT::MAX31BIT)),
        ],
        negative in proptest::bool::ANY,
    ) {
        let d = if negative { -d_abs } else { d_abs };

        let (hi, lo) = f64_to_fixt_pair(d);
        let recovered = fixt_pair_to_f64(hi, lo);

        // Fractional quantum is 1/u32::MAX approx. 2.33e-10; 1e-9 is generous.
        prop_assert!(
            (recovered - d).abs() < 1e-9,
            "roundtrip error too large: d={d}, recovered={recovered}, diff={}",
            (recovered - d).abs(),
        );
    }
}
