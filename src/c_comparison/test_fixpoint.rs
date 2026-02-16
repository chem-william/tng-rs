use crate::ffi;
use crate::fix_point::{FixT, f64_to_fixt_pair, fixt_pair_to_f64};

/// Truncate a C `fix_t` (unsigned long, 64-bit on LP64) to 32 bits,
/// matching the Rust `FixT` which wraps `u32`.
fn fix_t_to_u32(v: ffi::FixT) -> u32 {
    v as u32
}

#[test]
fn ud_to_fix_t_matches_c() {
    let max = 10.0;
    let test_values = [
        -5.0,
        0.0,
        5.0,
        10.0,
        15.0,
        3.421234251,
        4.234,
        8.123618970983,
    ];

    for &d in &test_values {
        let rust_result: u32 = FixT::from_f64_unsigned(d, max).into();
        let c_result = fix_t_to_u32(unsafe { ffi::Ptngc_ud_to_fix_t(d, max) });
        assert_eq!(
            rust_result, c_result,
            "ud_to_fix_t mismatch for d={d}, max={max}"
        );
    }
}

#[test]
fn d_to_fix_t_matches_c() {
    let max = 10.0;
    let test_values = [-15.0, -5.0, 0.0, 5.0, 10.0, 15.0];

    for &d in &test_values {
        let rust_result: u32 = FixT::from_f64_signed(d, max).into();
        let c_result = fix_t_to_u32(unsafe { ffi::Ptngc_d_to_fix_t(d, max) });
        assert_eq!(
            rust_result, c_result,
            "d_to_fix_t mismatch for d={d}, max={max}"
        );
    }
}

#[test]
fn fix_t_to_ud_matches_c() {
    let max = 10.0;
    // Use values produced by the unsigned encoder so they're realistic
    let test_doubles = [0.0, 3.0, 5.0, 7.5, 10.0];

    for &d in &test_doubles {
        let fix_val: u32 = FixT::from_f64_unsigned(d, max).into();
        let rust_result = FixT::from_f64_unsigned(d, max).to_f64_unsigned(max);
        let c_result = unsafe { ffi::Ptngc_fix_t_to_ud(fix_val as ffi::FixT, max) };
        assert!(
            (rust_result - c_result).abs() < 1e-9,
            "fix_t_to_ud mismatch for fix_val={fix_val}, max={max}: rust={rust_result}, c={c_result}"
        );
    }
}

#[test]
fn d_to_i32x2_matches_c() {
    let test_values = [
        -12.345678,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        12.345678,
        123456789.0,
    ];

    for &d in &test_values {
        let (rust_hi, rust_lo) = f64_to_fixt_pair(d);
        let rust_hi: u32 = rust_hi.into();
        let rust_lo: u32 = rust_lo.into();

        let mut c_hi: ffi::FixT = 0;
        let mut c_lo: ffi::FixT = 0;
        unsafe {
            ffi::Ptngc_d_to_i32x2(d, &mut c_hi, &mut c_lo);
        }

        assert_eq!(
            rust_hi,
            fix_t_to_u32(c_hi),
            "d_to_i32x2 hi mismatch for d={d}: rust=0x{rust_hi:08X}, c=0x{:08X}",
            fix_t_to_u32(c_hi)
        );
        assert_eq!(
            rust_lo,
            fix_t_to_u32(c_lo),
            "d_to_i32x2 lo mismatch for d={d}: rust=0x{rust_lo:08X}, c=0x{:08X}",
            fix_t_to_u32(c_lo)
        );
    }
}

#[test]
fn i32x2_roundtrip_matches_c() {
    let test_values = [-12.345678, -1.0, -0.5, 0.0, 0.5, 1.0, 12.345678];

    for &d in &test_values {
        // Encode with Rust
        let (rust_hi, rust_lo) = f64_to_fixt_pair(d);
        let hi_u32: u32 = rust_hi.into();
        let lo_u32: u32 = rust_lo.into();

        // Decode with C using the Rust-encoded values
        let c_result = unsafe { ffi::Ptngc_i32x2_to_d(hi_u32 as ffi::FixT, lo_u32 as ffi::FixT) };

        // Decode with Rust
        let rust_result = fixt_pair_to_f64(rust_hi, rust_lo);

        assert!(
            (rust_result - c_result).abs() < 1e-6,
            "i32x2 roundtrip mismatch for d={d}: rust={rust_result}, c={c_result}"
        );
    }
}

#[test]
fn unsigned_fixpoint_cross_roundtrip() {
    // Encode with Rust, decode with C and vice versa
    let test_values = [0.0, 5.0, 10.0, 3.14159, 7.777];
    let max = 10.0;

    for &d in &test_values {
        // Rust encode -> C decode
        let rust_fix: u32 = FixT::from_f64_unsigned(d, max).into();
        let c_decoded = unsafe { ffi::Ptngc_fix_t_to_ud(rust_fix as ffi::FixT, max) };
        let rust_decoded = FixT::from_f64_unsigned(d, max).to_f64_unsigned(max);
        assert!(
            (c_decoded - rust_decoded).abs() < 1e-9,
            "Rust encode -> C decode mismatch for d={d}"
        );

        // C encode -> Rust decode
        let c_fix = fix_t_to_u32(unsafe { ffi::Ptngc_ud_to_fix_t(d, max) });
        // Construct FixT from the C-produced value
        // from_f64_unsigned with same args should produce same value
        let rust_from_c = FixT::from_f64_unsigned(d, max);
        assert_eq!(
            u32::from(rust_from_c),
            c_fix,
            "C and Rust should encode to same value for d={d}"
        );
    }
}

#[test]
fn i32x2_cross_roundtrip() {
    // Encode with C, decode with Rust
    let test_values = [-12.345678, -1.0, 0.0, 1.0, 12.345678];

    for &d in &test_values {
        let mut c_hi: ffi::FixT = 0;
        let mut c_lo: ffi::FixT = 0;
        unsafe {
            ffi::Ptngc_d_to_i32x2(d, &mut c_hi, &mut c_lo);
        }

        // The Rust f64_to_fixt_pair should produce same hi/lo
        let (rust_hi, rust_lo) = f64_to_fixt_pair(d);
        assert_eq!(u32::from(rust_hi), fix_t_to_u32(c_hi));
        assert_eq!(u32::from(rust_lo), fix_t_to_u32(c_lo));

        // Decode the C-encoded values with Rust
        let rust_decoded = fixt_pair_to_f64(rust_hi, rust_lo);
        // Decode the C-encoded values with C
        let c_decoded = unsafe { ffi::Ptngc_i32x2_to_d(c_hi, c_lo) };

        assert!(
            (rust_decoded - c_decoded).abs() < 1e-6,
            "i32x2 cross roundtrip mismatch for d={d}: rust={rust_decoded}, c={c_decoded}"
        );
    }
}
