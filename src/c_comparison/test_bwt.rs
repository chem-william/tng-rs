use std::ffi::c_int;
use std::sync::Once;

static INIT: Once = Once::new();

fn init_logger() {
    INIT.call_once(|| {
        env_logger::builder().is_test(true).try_init().ok();
    });
}

use crate::{
    bwlzh::{bwt_sort, inverse_bwt, ptngc_comp_to_bwt},
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

// ----------------- sorting -----------------
// in the following, some of the tests use bitshifts
// to construct the `nrepeat` vecs. that's to easier test the
// repeat logic. so when we write
//            (repeat << 8) | k
//
//    [ repeat (…bits…) ][      k (8 bits) ]
//    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//          nrepeat entry

fn run_case(vals: &[u32], nrepeat: &[u32]) -> Vec<usize> {
    let n = vals.len();
    let mut rust_idx: Vec<usize> = (0..n).collect();
    bwt_sort(&mut rust_idx, n, vals, nrepeat);

    let mut c_indices: Vec<c_int> = (0..n as c_int).collect();
    let mut c_input = vals.to_vec();
    let mut c_nrepeat = nrepeat.to_vec();
    let nvals = n as c_int;
    let mut workarray: Vec<c_int> = vec![0; n];
    unsafe {
        ffi::Ptngc_bwt_merge_sort_inner(
            c_indices.as_mut_ptr(),
            nvals,
            c_input.as_mut_ptr(),
            0,
            nvals,
            c_nrepeat.as_mut_ptr(),
            workarray.as_mut_ptr(),
        );
    }

    let c_result: Vec<usize> = c_indices.iter().map(|&i| i as usize).collect();
    assert_eq!(
        rust_idx, c_result,
        "bwt_sort mismatch for vals={vals:?}, nrepeat={nrepeat:?}"
    );

    rust_idx
}

#[test]
fn simple_in_order() {
    let vals = [1, 2, 3];
    let nrepeat = vec![0u32; vals.len()]; // no repeats
    let sorted = run_case(&vals, &nrepeat);
    assert_eq!(sorted, vec![0, 1, 2]);
}

#[test]
fn rotated_sequence() {
    // rotations: [3,1,2], [1,2,3], [2,3,1]
    let vals = [3, 1, 2];
    let nrepeat = vec![0u32; vals.len()]; // no repeats
    let sorted = run_case(&vals, &nrepeat);
    // lex order: [1,2,3](@1), [2,3,1](@2), [3,1,2](@0)
    assert_eq!(sorted, vec![1, 2, 0]);
}

#[test]
fn all_equal() {
    let vals = [7, 7, 7, 7];
    let nrepeat = vec![0u32; vals.len()]; // no repeats
    let sorted = run_case(&vals, &nrepeat);
    // stable sort must preserve original [0,1,2,3]
    assert_eq!(sorted, vec![0, 1, 2, 3]);
}

#[test]
fn full_repeated_pattern() {
    let vals = [1, 2, 1, 2, 1, 2];
    let nrepeat = [(3 << 8) | 2; 6];
    let sorted = run_case(&vals, &nrepeat);
    assert_eq!(sorted, vec![0, 2, 4, 1, 3, 5]);
}

#[test]
fn partial_repeats_wrap() {
    let vals = [3, 4, 5, 3, 4];
    let nrepeat = [(2 << 8) | 3, 0, 0, (1 << 8) | 2, 0];
    let sorted = run_case(&vals, &nrepeat);
    assert_eq!(sorted, vec![3, 0, 4, 1, 2]);
}

#[test]
fn mismatched_k() {
    let vals = [0, 1, 0, 1, 0, 2, 0, 2];
    let nrepeat = [(3 << 8) | 2, 0, 0, 0, 0, (2 << 8) | 3, 0, 0];
    let sorted = run_case(&vals, &nrepeat);
    assert_eq!(sorted, vec![0, 2, 6, 4, 1, 3, 7, 5]);
}

#[test]
fn k_zero_malformed() {
    let vals = [5, 6, 7];
    let nrepeat = [(5 << 8), 0, 0];
    let sorted = run_case(&vals, &nrepeat);
    assert_eq!(sorted, vec![0, 1, 2]);
}

#[test]
fn overshoot_skip() {
    let vals = [9, 8, 7, 6];
    let nrepeat = [(10 << 8) | 1; 4];
    let sorted = run_case(&vals, &nrepeat);
    assert_eq!(sorted, vec![3, 2, 1, 0]);
}
