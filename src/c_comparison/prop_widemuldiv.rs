use crate::{
    ffi,
    widemuldiv::{ptngc_largeint_add, ptngc_largeint_div, ptngc_largeint_mul},
};
use proptest::prelude::*;
use std::ffi::c_int;

const N: usize = 4;

proptest! {
    #[test]
    fn prop_largeint_add_matches_c(
        v1 in 0u32..,
        input in prop::array::uniform4(0u32..)
    ) {
        let mut rust_out = input;
        ptngc_largeint_add(v1, &mut rust_out, N);

        let mut c_out = input;
        unsafe {
            ffi::Ptngc_largeint_add(v1, c_out.as_mut_ptr(), N as c_int);
        }

        prop_assert_eq!(rust_out, c_out, "add mismatch for v1={}, input={:?}", v1, input);
    }

    #[test]
    fn prop_largeint_mul_matches_c(
        v1 in 0u32..,
        input in prop::array::uniform4(0u32..)
    ) {
        let mut rust_out = [0u32; N];
        ptngc_largeint_mul(v1, &input, &mut rust_out, N);

        let mut c_in = input;
        let mut c_out = [0u32; N];
        unsafe {
            ffi::Ptngc_largeint_mul(v1, c_in.as_mut_ptr(), c_out.as_mut_ptr(), N as c_int);
        }

        prop_assert_eq!(rust_out, c_out, "mul mismatch for v1={}, input={:?}", v1, input);
    }

    #[test]
    fn prop_largeint_div_matches_c(
        divisor in 1u32..,
        input in prop::array::uniform4(0u32..)
    ) {
        let mut rust_out = [0u32; N];
        let rust_rem = ptngc_largeint_div(divisor, &input, &mut rust_out, N);

        let mut c_in = input;
        let mut c_out = [0u32; N];
        let c_rem = unsafe {
            ffi::Ptngc_largeint_div(divisor, c_in.as_mut_ptr(), c_out.as_mut_ptr(), N as c_int)
        };

        prop_assert_eq!(rust_out, c_out, "div quotient mismatch for divisor={}, input={:?}", divisor, input);
        prop_assert_eq!(rust_rem, c_rem, "div remainder mismatch for divisor={}, input={:?}", divisor, input);
    }

    #[test]
    fn prop_largeint_mul_div_roundtrip_c(
        multiplier in 1u32..,
        input in prop::array::uniform4(0u32..)
    ) {
        // Multiply in both Rust and C
        let mut rust_mul = [0u32; N];
        ptngc_largeint_mul(multiplier, &input, &mut rust_mul, N);

        let mut c_in = input;
        let mut c_mul = [0u32; N];
        unsafe {
            ffi::Ptngc_largeint_mul(multiplier, c_in.as_mut_ptr(), c_mul.as_mut_ptr(), N as c_int);
        }
        prop_assert_eq!(rust_mul, c_mul, "mul step mismatch");

        // Divide in both Rust and C
        let mut rust_div = [0u32; N];
        let rust_rem = ptngc_largeint_div(multiplier, &rust_mul, &mut rust_div, N);

        let mut c_div = [0u32; N];
        let c_rem = unsafe {
            ffi::Ptngc_largeint_div(multiplier, c_mul.as_mut_ptr(), c_div.as_mut_ptr(), N as c_int)
        };

        prop_assert_eq!(rust_div, c_div, "div step mismatch");
        prop_assert_eq!(rust_rem, c_rem, "div remainder mismatch");
        let _ = (rust_rem, c_rem);
    }
}
