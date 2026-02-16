use std::ffi::c_int;
use std::sync::Once;

static INIT: Once = Once::new();

fn init_logger() {
    INIT.call_once(|| {
        env_logger::builder().is_test(true).try_init().ok();
    });
}

use crate::{
    bwlzh::{inverse_bwt, ptngc_comp_to_bwt},
    ffi,
};

fn roundtrip_bwt(vals: &[u32]) {
    init_logger();
    let mut rust_output = vec![0; vals.len()];
    let rust_index = ptngc_comp_to_bwt(vals, vals.len(), &mut rust_output);

    let mut c_input = vals.to_owned().clone();
    let mut c_output = vec![0; vals.len()];
    let mut c_index = 0;
    unsafe {
        ffi::Ptngc_comp_to_bwt(
            c_input.as_mut_ptr(),
            c_input.len() as c_int,
            c_output.as_mut_ptr(),
            &mut c_index,
        );
    }
    assert_eq!(
        rust_index, c_index as usize,
        "BWT roundtrip failed for indices. Rust: {rust_index}, C: {c_index}"
    );

    assert_eq!(
        rust_output, c_output,
        "BWT roundtrip failed for output. Rust: {rust_output:?}, C: {c_output:?}"
    );

    let mut recovered = vec![0; vals.len()];
    inverse_bwt(&rust_output, rust_index, &mut recovered);
    assert_eq!(
        recovered, vals,
        "BWT roundtrip recovery for the Rust part failed for input: {vals:?}",
    );
}

#[test]
fn test_bwt_roundtrip_empty() {
    roundtrip_bwt(&[]);
}

#[test]
fn test_bwt_roundtrip_single() {
    roundtrip_bwt(&[42]);
}

#[test]
fn test_bwt_roundtrip_simple_ascii() {
    let input = b"banana".iter().map(|&b| b as u32).collect::<Vec<u32>>();
    roundtrip_bwt(&input);
}

#[test]
fn test_bwt_roundtrip_full_ascii_range() {
    let input = (0u8..=255u8).map(|b| b as u32).collect::<Vec<_>>();
    roundtrip_bwt(&input);
}

#[test]
fn test_bwt_roundtrip_repeating_pattern() {
    let input = b"ABABABABAB".iter().map(|&b| b as u32).collect::<Vec<_>>();
    roundtrip_bwt(&input);
}

#[test]
fn test_bwt_roundtrip_palindrome() {
    let input = b"racecar".iter().map(|&b| b as u32).collect::<Vec<_>>();
    roundtrip_bwt(&input);
}

#[test]
fn test_bwt_roundtrip_binary_data() {
    let input = vec![0u32, 1, 2, 3, 0, 1, 2, 3];
    roundtrip_bwt(&input);
}
