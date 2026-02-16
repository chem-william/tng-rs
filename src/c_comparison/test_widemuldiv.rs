use std::ffi::c_int;

use crate::ffi;
use crate::widemuldiv::{ptngc_largeint_add, ptngc_largeint_mul};

#[test]
fn largeint_add_matches_c() {
    let test_cases: &[(u32, &[u32])] = &[
        (1, &[0, 0, 0, 0]),
        (1, &[u32::MAX, 0, 0, 0]),           // carry propagation
        (1, &[u32::MAX, u32::MAX, 0, 0]),     // double carry
        (100, &[50, 200, 300, 400]),
        (0, &[1, 2, 3, 4]),
        (u32::MAX, &[1, 0, 0, 0]),
    ];

    for &(v1, input) in test_cases {
        let n = input.len();

        let mut rust_largeint = input.to_vec();
        ptngc_largeint_add(v1, &mut rust_largeint, n);

        let mut c_largeint = input.to_vec();
        unsafe {
            ffi::Ptngc_largeint_add(v1, c_largeint.as_mut_ptr(), n as c_int);
        }

        assert_eq!(
            rust_largeint, c_largeint,
            "largeint_add mismatch for v1={v1}, input={input:?}"
        );
    }
}

#[test]
fn largeint_mul_matches_c() {
    let test_cases: &[(u32, &[u32])] = &[
        (0, &[1, 2, 3, 4]),
        (1, &[1, 2, 3, 4]),
        (3, &[2, 0, 0, 0]),
        (2, &[0x80000000, 0, 0, 0]),
        (2, &[0xFFFFFFFF, 0, 0, 0]),
        (2, &[0xFFFFFFFF, 0xFFFFFFFF, 0, 0]),
        (2, &[0, 0, 0, 0xFFFFFFFF]),
        (0x1000, &[0x12345678, 0x9ABCDEF0, 0, 0]),
        (0xFFFFFFFF, &[1, 0, 0, 0]),
        (0xFFFFFFFF, &[0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF]),
    ];

    for &(v1, input) in test_cases {
        let n = input.len();

        let mut rust_out = vec![0u32; n];
        ptngc_largeint_mul(v1, input, &mut rust_out, n);

        let mut c_in = input.to_vec();
        let mut c_out = vec![0u32; n];
        unsafe {
            ffi::Ptngc_largeint_mul(v1, c_in.as_mut_ptr(), c_out.as_mut_ptr(), n as c_int);
        }

        assert_eq!(
            rust_out, c_out,
            "largeint_mul mismatch for v1={v1}, input={input:?}"
        );
    }
}

#[test]
fn largeint_div_c_only_smoke_test() {
    // Verify the C div function works (no Rust equivalent yet to compare against,
    // but we can test mul then div roundtrips in C).
    let original = [42u32, 0, 0, 0];
    let multiplier = 7u32;
    let n = 4i32;

    let mut mul_out = vec![0u32; n as usize];
    let mut div_out = vec![0u32; n as usize];

    unsafe {
        let mut input = original.to_vec();
        ffi::Ptngc_largeint_mul(multiplier, input.as_mut_ptr(), mul_out.as_mut_ptr(), n);
        let remainder = ffi::Ptngc_largeint_div(multiplier, mul_out.as_mut_ptr(), div_out.as_mut_ptr(), n);
        assert_eq!(remainder, 0, "Division should have no remainder");
        assert_eq!(div_out, original, "mul then div should roundtrip");
    }
}

#[test]
fn largeint_mul_then_div_roundtrip_c() {
    let test_cases: &[(u32, &[u32])] = &[
        (3, &[100, 0, 0, 0]),
        (17, &[12345, 67890, 0, 0]),
        (256, &[1, 1, 1, 0]),
    ];

    for &(multiplier, input) in test_cases {
        let n = input.len() as c_int;
        let mut c_in = input.to_vec();
        let mut mul_out = vec![0u32; input.len()];
        let mut div_out = vec![0u32; input.len()];

        unsafe {
            ffi::Ptngc_largeint_mul(multiplier, c_in.as_mut_ptr(), mul_out.as_mut_ptr(), n);
            let remainder = ffi::Ptngc_largeint_div(multiplier, mul_out.as_mut_ptr(), div_out.as_mut_ptr(), n);
            assert_eq!(remainder, 0, "Remainder should be 0 for multiplier={multiplier}, input={input:?}");
            assert_eq!(
                div_out, input,
                "mul/div roundtrip failed for multiplier={multiplier}, input={input:?}"
            );
        }
    }
}
