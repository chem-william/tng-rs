use crate::{
    ffi,
    rle::{ptngc_comp_conv_from_rle, ptngc_comp_conv_to_rle},
};
use proptest::prelude::*;
use std::ffi::c_int;

fn c_rle_encode(input: &[u32], min_rle: usize) -> Vec<u32> {
    let mut c_output = vec![0u32; input.len() * 4 + 16];
    let mut c_len: c_int = 0;
    unsafe {
        ffi::Ptngc_comp_conv_to_rle(
            input.as_ptr().cast(),
            input.len() as c_int,
            c_output.as_mut_ptr().cast(),
            &raw mut c_len,
            min_rle as c_int,
        );
    }
    c_output.truncate(c_len as usize);
    c_output
}

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

/// Generates a mix: 2/3 small-range values (0..=15, likely runs) and
/// 1/3 full-range (0..=0xFFFF, stress-tests value encoding).
fn rle_vals() -> impl Strategy<Value = Vec<u32>> {
    prop_oneof![
        2 => prop::collection::vec(0u32..=15, 0..=256usize),
        1 => prop::collection::vec(0u32..=0xFFFF, 0..=256usize),
    ]
}

proptest! {
    #[test]
    fn prop_encode_matches_c(
        vals in rle_vals(),
        min_rle in 1usize..=8,
    ) {
        let rust = ptngc_comp_conv_to_rle(&vals, min_rle);
        let c = c_rle_encode(&vals, min_rle);
        prop_assert_eq!(rust, c, "encode mismatch for min_rle={}", min_rle);
    }

    /// Uses small value range to ensure runs are generated and RLE is exercised.
    #[test]
    fn prop_roundtrip(
        vals in prop::collection::vec(0u32..=15, 1..=256usize),
        min_rle in 1usize..=8,
    ) {
        let encoded = ptngc_comp_conv_to_rle(&vals, min_rle);
        let decoded = ptngc_comp_conv_from_rle(&encoded, vals.len());
        prop_assert_eq!(decoded, vals);
    }

    #[test]
    fn prop_cross_rust_encode_c_decode(
        vals in prop::collection::vec(0u32..=15, 1..=256usize),
        min_rle in 1usize..=8,
    ) {
        let encoded = ptngc_comp_conv_to_rle(&vals, min_rle);
        let decoded = c_rle_decode(&encoded, vals.len());
        prop_assert_eq!(decoded, vals, "Rust->C roundtrip failed for min_rle={}", min_rle);
    }

    #[test]
    fn prop_cross_c_encode_rust_decode(
        vals in prop::collection::vec(0u32..=15, 1..=256usize),
        min_rle in 1usize..=8,
    ) {
        let encoded = c_rle_encode(&vals, min_rle);
        let decoded = ptngc_comp_conv_from_rle(&encoded, vals.len());
        prop_assert_eq!(decoded, vals, "C->Rust roundtrip failed for min_rle={}", min_rle);
    }
}
