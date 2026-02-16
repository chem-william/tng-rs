use std::ffi::c_int;

use crate::ffi;
use crate::rle::{ptngc_comp_conv_from_rle, ptngc_comp_conv_to_rle};

/// Call C `Ptngc_comp_conv_to_rle`, return the output as a Vec<u32>.
fn c_rle_encode(input: &[u32], min_rle: i32) -> Vec<u32> {
    let mut c_output = vec![0u32; input.len() * 4 + 16];
    let mut c_len: c_int = 0;
    unsafe {
        ffi::Ptngc_comp_conv_to_rle(
            input.as_ptr().cast(),
            input.len() as c_int,
            c_output.as_mut_ptr().cast(),
            &mut c_len,
            min_rle as c_int,
        );
    }
    c_output.truncate(c_len as usize);
    c_output
}

/// Call C `Ptngc_comp_conv_from_rle`, return the output as a Vec<u32>.
fn c_rle_decode(rle: &[u32], output_len: usize) -> Vec<u32> {
    let mut c_output = vec![0u32; output_len];
    unsafe {
        ffi::Ptngc_comp_conv_from_rle(
            rle.as_ptr().cast(),
            c_output.as_mut_ptr().cast(),
            output_len as c_int,
        );
    }
    c_output
}

#[test]
fn rle_encode_matches_c() {
    let test_cases: &[(&[u32], i32)] = &[
        (&[1, 2, 3], 3),
        (&[7, 7, 7, 7, 7], 3),
        (&[9, 9, 9, 9], 4),
        (&[0, 0, 0, 0, 0, 0], 2),
        (&[1, 1, 2, 2, 3, 3], 2),
        (&[100, 200, 300], 3),
    ];

    for &(input, min_rle) in test_cases {
        let rust_result = ptngc_comp_conv_to_rle(input, min_rle as usize);
        let c_result = c_rle_encode(input, min_rle);
        assert_eq!(
            rust_result, c_result,
            "RLE encode mismatch for input={input:?}, min_rle={min_rle}"
        );
    }
}

#[test]
fn rle_decode_matches_c() {
    // Encode with C, then decode with both and compare
    let test_cases: &[(&[u32], i32)] = &[
        (&[1, 2, 3], 3),
        (&[7, 7, 7, 7, 7], 3),
        (&[0, 0, 0, 0, 0, 0], 2),
        (&[1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6], 2),
    ];

    for &(input, min_rle) in test_cases {
        let encoded = c_rle_encode(input, min_rle);
        let rust_decoded = ptngc_comp_conv_from_rle(&encoded, input.len());
        let c_decoded = c_rle_decode(&encoded, input.len());
        assert_eq!(
            rust_decoded,
            c_decoded,
            "RLE decode mismatch for encoded={encoded:?}, len={}",
            input.len()
        );
        assert_eq!(
            rust_decoded, input,
            "RLE roundtrip failed for input={input:?}"
        );
    }
}

#[test]
fn rle_cross_roundtrip_rust_encode_c_decode() {
    let test_cases: &[(&[u32], i32)] = &[
        (&[], 0),
        (&[1, 2, 3], 3),
        (&[7, 7, 7, 7, 7], 3),
        (&[5, 5, 5, 5, 5, 5, 5, 5], 2),
        (&[0, 1, 0, 1, 0, 1], 3),
    ];

    for &(input, min_rle) in test_cases {
        let rust_encoded = ptngc_comp_conv_to_rle(input, min_rle as usize);
        let c_decoded = c_rle_decode(&rust_encoded, input.len());
        assert_eq!(
            c_decoded.as_slice(),
            input,
            "Cross roundtrip (Rust encode -> C decode) failed for input={input:?}"
        );
    }
}

#[test]
fn rle_cross_roundtrip_c_encode_rust_decode() {
    let test_cases: &[(&[u32], i32)] = &[
        (&[], 0),
        (&[1, 2, 3], 3),
        (&[7, 7, 7, 7, 7], 3),
        (&[5, 5, 5, 5, 5, 5, 5, 5], 2),
        (&[0, 1, 0, 1, 0, 1], 3),
        (&[1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6], 2),
    ];

    for &(input, min_rle) in test_cases {
        let c_encoded = c_rle_encode(input, min_rle);
        let rust_decoded = ptngc_comp_conv_from_rle(&c_encoded, input.len());
        assert_eq!(
            rust_decoded.as_slice(),
            input,
            "Cross roundtrip (C encode -> Rust decode) failed for input={input:?}"
        );
    }
}

#[test]
fn rle_edge_case_single_element() {
    let input = &[42u32];
    let min_rle = 3;

    let rust_result = ptngc_comp_conv_to_rle(input, min_rle as usize);
    let c_result = c_rle_encode(input, min_rle);
    assert_eq!(rust_result, c_result);
}

#[test]
fn rle_edge_case_long_run() {
    let input: Vec<u32> = vec![3; 100];
    let min_rle = 2;

    let rust_result = ptngc_comp_conv_to_rle(&input, min_rle as usize);
    let c_result = c_rle_encode(&input, min_rle);
    assert_eq!(rust_result, c_result);

    // Verify roundtrip
    let rust_decoded = ptngc_comp_conv_from_rle(&rust_result, input.len());
    assert_eq!(rust_decoded, input);
}
