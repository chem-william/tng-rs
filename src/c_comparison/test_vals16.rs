use std::ffi::c_int;
use std::sync::Once;

static INIT: Once = Once::new();

fn init_logger() {
    INIT.call_once(|| {
        env_logger::builder().is_test(true).try_init().ok();
    });
}

use crate::{
    bwlzh::{ptngc_comp_conv_from_vals16, ptngc_comp_conv_to_vals16},
    ffi,
};

#[test]
fn single_small_values() {
    init_logger();

    let vals = [0x1234, 0x7FFF, 0x0000, 0x0001];
    let mut rust_vals16 = [0u32; 20];
    let rust_nvals16 = ptngc_comp_conv_to_vals16(&vals, &mut rust_vals16);

    let mut c_vals = vals;
    let mut c_vals16 = [0u32; 20];
    let mut c_nvals16 = 0;
    unsafe {
        ffi::Ptngc_comp_conv_to_vals16(
            c_vals.as_mut_ptr(),
            c_vals.len() as c_int,
            c_vals16.as_mut_ptr(),
            &mut c_nvals16,
        )
    };

    assert_eq!(rust_nvals16, c_nvals16 as usize);
    assert_eq!(rust_vals16, c_vals16);
}

#[test]
fn two_chunk_values() {
    init_logger();

    // Test value 0x12345678
    // lo = (0x12345678 & 0x7FFF) | 0x8000 = 0x5678 | 0x8000 = 0xD678
    // hi = 0x12345678 >> 15 = 0x2468A (≤ 0x7FFF, so stored directly)
    let vals = [0x12345678];
    let mut rust_vals16 = [0u32; 20];
    let rust_nvals16 = ptngc_comp_conv_to_vals16(&vals, &mut rust_vals16);

    let mut c_vals = vals;
    let mut c_vals16 = [0u32; 20];
    let mut c_nvals16 = 0;
    unsafe {
        ffi::Ptngc_comp_conv_to_vals16(
            c_vals.as_mut_ptr(),
            c_vals.len() as c_int,
            c_vals16.as_mut_ptr(),
            &mut c_nvals16,
        )
    };

    assert_eq!(rust_nvals16, c_nvals16 as usize);
    assert_eq!(rust_vals16, c_vals16);
}

#[test]
fn three_chunk_values() {
    init_logger();

    // Test large value 0x80000000 (needs 3 chunks)
    // lo = (0x80000000 & 0x7FFF) | 0x8000 = 0x0000 | 0x8000 = 0x8000
    // hi = 0x80000000 >> 15 = 0x100000 (> 0x7FFF, needs splitting)
    // lohi = (0x100000 & 0x7FFF) | 0x8000 = 0x0000 | 0x8000 = 0x8000
    // hihi = 0x100000 >> 15 = 0x2000
    let vals = [0x80000000];
    let mut rust_vals16 = [0u32; 20];

    let rust_nvals16 = ptngc_comp_conv_to_vals16(&vals, &mut rust_vals16);

    let mut c_vals = vals;
    let mut c_vals16 = [0u32; 20];
    let mut c_nvals16 = 0;
    unsafe {
        ffi::Ptngc_comp_conv_to_vals16(
            c_vals.as_mut_ptr(),
            c_vals.len() as c_int,
            c_vals16.as_mut_ptr(),
            &mut c_nvals16,
        )
    };

    assert_eq!(rust_nvals16, c_nvals16 as usize);
    assert_eq!(rust_vals16, c_vals16);
}

#[test]
fn boundary_values() {
    init_logger();

    let vals = [
        0x7FFF,     // Maximum single chunk
        0x8000,     // Minimum two-chunk
        0x3FFFFFFF, // Maximum two-chunk (hi = 0x7FFF)
        0x40000000, // Minimum three-chunk (hi = 0x8000)
    ];
    let mut rust_vals16 = [0u32; 20];

    let _ = ptngc_comp_conv_to_vals16(&vals, &mut rust_vals16);

    let mut c_vals = vals;
    let mut c_vals16 = [0u32; 20];
    let mut c_nvals16 = 0;
    unsafe {
        ffi::Ptngc_comp_conv_to_vals16(
            c_vals.as_mut_ptr(),
            c_vals.len() as c_int,
            c_vals16.as_mut_ptr(),
            &mut c_nvals16,
        )
    };

    assert_eq!(rust_vals16, c_vals16);
}

#[test]
fn mixed_values() {
    init_logger();

    let vals = [
        0x1234,     // Single chunk
        0x12345678, // Two chunks
        0x5555,     // Single chunk
        0x80000001, // Three chunks
    ];
    let mut rust_vals16 = [0u32; 20];

    let rust_nvals16 = ptngc_comp_conv_to_vals16(&vals, &mut rust_vals16);

    let mut c_vals = vals;
    let mut c_vals16 = [0u32; 20];
    let mut c_nvals16 = 0;
    unsafe {
        ffi::Ptngc_comp_conv_to_vals16(
            c_vals.as_mut_ptr(),
            c_vals.len() as c_int,
            c_vals16.as_mut_ptr(),
            &mut c_nvals16,
        )
    };

    assert_eq!(rust_nvals16, c_nvals16 as usize);
    assert_eq!(rust_vals16, c_vals16);
}

#[test]
fn empty_input() {
    init_logger();

    let vals: [u32; 0] = [];
    let mut rust_vals16 = [0u32; 20];

    let rust_nvals16 = ptngc_comp_conv_to_vals16(&vals, &mut rust_vals16);

    let mut c_vals = vals;
    let mut c_vals16 = [0u32; 20];
    let mut c_nvals16 = 0;
    unsafe {
        ffi::Ptngc_comp_conv_to_vals16(
            c_vals.as_mut_ptr(),
            c_vals.len() as c_int,
            c_vals16.as_mut_ptr(),
            &mut c_nvals16,
        )
    };

    assert_eq!(rust_nvals16, c_nvals16 as usize);
    assert_eq!(rust_vals16, c_vals16);
}

#[test]
fn test_max_values() {
    init_logger();

    let vals = [0xFFFFFFFF];
    let mut rust_vals16 = [0u32; 20];

    let rust_nvals16 = ptngc_comp_conv_to_vals16(&vals, &mut rust_vals16);

    let mut c_vals = vals;
    let mut c_vals16 = [0u32; 20];
    let mut c_nvals16 = 0;
    unsafe {
        ffi::Ptngc_comp_conv_to_vals16(
            c_vals.as_mut_ptr(),
            c_vals.len() as c_int,
            c_vals16.as_mut_ptr(),
            &mut c_nvals16,
        )
    };

    assert_eq!(rust_nvals16, c_nvals16 as usize);

    // lo = (0xFFFFFFFF & 0x7FFF) | 0x8000 = 0x7FFF | 0x8000 = 0xFFFF
    // hi = 0xFFFFFFFF >> 15 = 0x1FFFF
    // lohi = (0x1FFFF & 0x7FFF) | 0x8000 = 0x7FFF | 0x8000 = 0xFFFF
    // hihi = 0x1FFFF >> 15 = 0x3
    assert_eq!(rust_vals16, c_vals16);
}

#[test]
fn compression_efficiency() {
    init_logger();

    // Test various ranges to show compression behavior
    let test_cases = [
        (0x1000, 1),     // Small value -> 1 chunk
        (0x10000, 2),    // Medium value -> 2 chunks
        (0x1000000, 2),  // Large 2-chunk value -> 2 chunks
        (0x40000000, 3), // Large 3-chunk value -> 3 chunks
    ];

    for (val, expected_chunks) in test_cases {
        let vals = [val];
        let mut rust_vals16 = [0u32; 10];

        let rust_nvals16 = ptngc_comp_conv_to_vals16(&vals, &mut rust_vals16);

        let mut c_vals = vals;
        let mut c_vals16 = [0u32; 10];
        let mut c_nvals16 = 0;
        unsafe {
            ffi::Ptngc_comp_conv_to_vals16(
                c_vals.as_mut_ptr(),
                c_vals.len() as c_int,
                c_vals16.as_mut_ptr(),
                &mut c_nvals16,
            )
        };

        assert_eq!(rust_nvals16, c_nvals16 as usize);
        assert_eq!(rust_vals16, c_vals16);
    }
}
