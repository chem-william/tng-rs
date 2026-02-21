use std::ffi::c_int;

use crate::{
    dict::ptngc_comp_make_dict_hist,
    ffi,
    mtf::{
        ptngc_comp_conv_from_mtf, ptngc_comp_conv_from_mtf_partial,
        ptngc_comp_conv_from_mtf_partial3, ptngc_comp_conv_to_mtf, ptngc_comp_conv_to_mtf_partial,
        ptngc_comp_conv_to_mtf_partial3,
    },
};

// ---------------------------------------------------------------------------
// C wrappers
// ---------------------------------------------------------------------------

fn c_to_mtf_partial3(vals: &[u32]) -> Vec<u8> {
    let nvals = vals.len();
    let mut out = vec![0u8; nvals * 3];
    unsafe {
        ffi::Ptngc_comp_conv_to_mtf_partial3(vals.as_ptr(), nvals as c_int, out.as_mut_ptr());
    }
    out
}

fn c_from_mtf_partial3(encoded: &[u8], nvals: usize) -> Vec<u32> {
    let mut out = vec![0u32; nvals];
    let mut enc_copy = encoded.to_vec();
    unsafe {
        ffi::Ptngc_comp_conv_from_mtf_partial3(
            enc_copy.as_mut_ptr(),
            nvals as c_int,
            out.as_mut_ptr(),
        );
    }
    out
}

fn c_to_mtf_partial(vals: &[u32]) -> Vec<u32> {
    let nvals = vals.len();
    let mut out = vec![0u32; nvals];
    unsafe {
        ffi::Ptngc_comp_conv_to_mtf_partial(vals.as_ptr(), nvals as c_int, out.as_mut_ptr());
    }
    out
}

fn c_from_mtf_partial(encoded: &[u32]) -> Vec<u32> {
    let nvals = encoded.len();
    let mut out = vec![0u32; nvals];
    unsafe {
        ffi::Ptngc_comp_conv_from_mtf_partial(encoded.as_ptr(), nvals as c_int, out.as_mut_ptr());
    }
    out
}

fn c_to_mtf(vals: &[u32], dict: &[u32]) -> Vec<u32> {
    let nvals = vals.len();
    let mut out = vec![0u32; nvals];
    unsafe {
        ffi::Ptngc_comp_conv_to_mtf(
            vals.as_ptr(),
            nvals as c_int,
            dict.as_ptr(),
            dict.len() as c_int,
            out.as_mut_ptr(),
        );
    }
    out
}

fn c_from_mtf(encoded: &[u32], dict: &[u32]) -> Vec<u32> {
    let nvals = encoded.len();
    let mut out = vec![0u32; nvals];
    unsafe {
        ffi::Ptngc_comp_conv_from_mtf(
            encoded.as_ptr(),
            nvals as c_int,
            dict.as_ptr(),
            dict.len() as c_int,
            out.as_mut_ptr(),
        );
    }
    out
}

// ---------------------------------------------------------------------------
// partial3 tests
// ---------------------------------------------------------------------------

#[test]
fn test_to_mtf_partial3_matches_c() {
    let test_cases: &[&[u32]] = &[
        &[0, 1, 2, 3],
        &[0xFF, 0xAB, 0x00, 0xFF],
        &[0x010203, 0x040506, 0x070809],
        &[7, 7, 7, 7, 7],
        &[0, 0, 0, 1],
        &[0xFFFFFF, 0, 0xABCDEF],
    ];
    for &input in test_cases {
        let nvals = input.len();
        let mut rust_out = vec![0u8; nvals * 3];
        ptngc_comp_conv_to_mtf_partial3(input, nvals, &mut rust_out);
        let c_out = c_to_mtf_partial3(input);
        assert_eq!(
            rust_out, c_out,
            "to_mtf_partial3 mismatch for input={input:?}"
        );
    }
}

#[test]
fn test_from_mtf_partial3_matches_c() {
    let test_cases: &[&[u32]] = &[
        &[0, 1, 2, 3],
        &[0xFF, 0xAB, 0x00, 0xFF],
        &[0x010203, 0x040506, 0x070809],
        &[7, 7, 7, 7, 7],
        &[0, 0, 0, 1],
        &[0xFFFFFF, 0, 0xABCDEF],
    ];
    for &input in test_cases {
        let nvals = input.len();
        // Encode with C, then decode with both and compare
        let encoded = c_to_mtf_partial3(input);

        let mut rust_out = vec![0u32; nvals];
        ptngc_comp_conv_from_mtf_partial3(&encoded, &mut rust_out);
        let c_out = c_from_mtf_partial3(&encoded, nvals);

        assert_eq!(
            rust_out, c_out,
            "from_mtf_partial3 Rust/C mismatch for input={input:?}"
        );
        assert_eq!(
            rust_out.as_slice(),
            input,
            "from_mtf_partial3 roundtrip failed for input={input:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// partial tests
// ---------------------------------------------------------------------------

#[test]
fn test_to_mtf_partial_matches_c() {
    let test_cases: &[&[u32]] = &[
        &[0, 1, 2, 3],
        &[0xFF, 0xAB, 0x00, 0xFF],
        &[0x010203, 0x040506, 0x070809],
        &[7, 7, 7, 7, 7],
        &[0, 0, 0, 1],
        &[0xFFFFFF, 0, 0xABCDEF],
    ];
    for &input in test_cases {
        let nvals = input.len();
        let mut rust_out = vec![0u32; nvals];
        ptngc_comp_conv_to_mtf_partial(input, &mut rust_out);
        let c_out = c_to_mtf_partial(input);
        assert_eq!(
            rust_out, c_out,
            "to_mtf_partial mismatch for input={input:?}"
        );
    }
}

#[test]
fn test_from_mtf_partial_matches_c() {
    let test_cases: &[&[u32]] = &[
        &[0, 1, 2, 3],
        &[0xFF, 0xAB, 0x00, 0xFF],
        &[0x010203, 0x040506, 0x070809],
        &[7, 7, 7, 7, 7],
        &[0, 0, 0, 1],
        &[0xFFFFFF, 0, 0xABCDEF],
    ];
    for &input in test_cases {
        let nvals = input.len();
        // Encode with C, then decode with both and compare
        let encoded = c_to_mtf_partial(input);

        let mut rust_out = vec![0u32; nvals];
        ptngc_comp_conv_from_mtf_partial(&encoded, &mut rust_out);
        let c_out = c_from_mtf_partial(&encoded);

        assert_eq!(
            rust_out, c_out,
            "from_mtf_partial Rust/C mismatch for input={input:?}"
        );
        assert_eq!(
            rust_out.as_slice(),
            input,
            "from_mtf_partial roundtrip failed for input={input:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// dict-based tests
// ---------------------------------------------------------------------------

#[test]
fn test_to_mtf_matches_c() {
    let test_cases: &[&[u32]] = &[
        &[3, 1, 3, 7, 1],
        &[0, 1, 2, 3, 0, 1, 2, 3],
        &[100, 200, 100, 150, 200, 100],
        &[42],
        &[5, 5, 5, 5],
    ];
    for &input in test_cases {
        let (dict, _) = ptngc_comp_make_dict_hist(input);
        let nvals = input.len();
        let mut rust_out = vec![0u32; nvals];
        ptngc_comp_conv_to_mtf(input, &dict, &mut rust_out);
        let c_out = c_to_mtf(input, &dict);
        assert_eq!(rust_out, c_out, "to_mtf mismatch for input={input:?}");
    }
}

#[test]
fn test_from_mtf_matches_c() {
    let test_cases: &[&[u32]] = &[
        &[3, 1, 3, 7, 1],
        &[0, 1, 2, 3, 0, 1, 2, 3],
        &[100, 200, 100, 150, 200, 100],
        &[42],
        &[5, 5, 5, 5],
    ];
    for &input in test_cases {
        let (dict, _) = ptngc_comp_make_dict_hist(input);
        let nvals = input.len();
        // Encode with Rust, then decode with both and compare
        let mut encoded = vec![0u32; nvals];
        ptngc_comp_conv_to_mtf(input, &dict, &mut encoded);

        let mut rust_out = vec![0u32; nvals];
        ptngc_comp_conv_from_mtf(&encoded, &dict, &mut rust_out);
        let c_out = c_from_mtf(&encoded, &dict);

        assert_eq!(
            rust_out, c_out,
            "from_mtf Rust/C mismatch for input={input:?}"
        );
        assert_eq!(
            rust_out.as_slice(),
            input,
            "from_mtf roundtrip failed for input={input:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Cross-direction roundtrip tests
// ---------------------------------------------------------------------------

#[test]
fn test_mtf_partial3_cross_roundtrip_rust_encode_c_decode() {
    let test_cases: &[&[u32]] = &[
        &[0, 1, 2, 3],
        &[0xAB, 0xCD, 0xEF, 0x01],
        &[0x010203, 0x040506],
    ];
    for &input in test_cases {
        let nvals = input.len();
        let mut encoded = vec![0u8; nvals * 3];
        ptngc_comp_conv_to_mtf_partial3(input, nvals, &mut encoded);
        let decoded = c_from_mtf_partial3(&encoded, nvals);
        assert_eq!(
            decoded.as_slice(),
            input,
            "partial3 cross roundtrip (Rust enc → C dec) failed for input={input:?}"
        );
    }
}

#[test]
fn test_mtf_partial3_cross_roundtrip_c_encode_rust_decode() {
    let test_cases: &[&[u32]] = &[
        &[0, 1, 2, 3],
        &[0xAB, 0xCD, 0xEF, 0x01],
        &[0x010203, 0x040506],
    ];
    for &input in test_cases {
        let nvals = input.len();
        let encoded = c_to_mtf_partial3(input);
        let mut decoded = vec![0u32; nvals];
        ptngc_comp_conv_from_mtf_partial3(&encoded, &mut decoded);
        assert_eq!(
            decoded.as_slice(),
            input,
            "partial3 cross roundtrip (C enc → Rust dec) failed for input={input:?}"
        );
    }
}

#[test]
fn test_mtf_partial_cross_roundtrip_rust_encode_c_decode() {
    let test_cases: &[&[u32]] = &[
        &[0, 1, 2, 3],
        &[0xFF, 0xAB],
        &[0x010203, 0x040506, 0x070809],
    ];
    for &input in test_cases {
        let nvals = input.len();
        let mut encoded = vec![0u32; nvals];
        ptngc_comp_conv_to_mtf_partial(input, &mut encoded);
        let decoded = c_from_mtf_partial(&encoded);
        assert_eq!(
            decoded.as_slice(),
            input,
            "partial cross roundtrip (Rust enc → C dec) failed for input={input:?}"
        );
    }
}

#[test]
fn test_mtf_partial_cross_roundtrip_c_encode_rust_decode() {
    let test_cases: &[&[u32]] = &[
        &[0, 1, 2, 3],
        &[0xFF, 0xAB],
        &[0x010203, 0x040506, 0x070809],
    ];
    for &input in test_cases {
        let nvals = input.len();
        let encoded = c_to_mtf_partial(input);
        let mut decoded = vec![0u32; nvals];
        ptngc_comp_conv_from_mtf_partial(&encoded, &mut decoded);
        assert_eq!(
            decoded.as_slice(),
            input,
            "partial cross roundtrip (C enc → Rust dec) failed for input={input:?}"
        );
    }
}

#[test]
fn test_mtf_cross_roundtrip_rust_encode_c_decode() {
    let test_cases: &[&[u32]] = &[&[3, 1, 3, 7, 1], &[0, 1, 2, 0, 1], &[99, 99, 99]];
    for &input in test_cases {
        let (dict, _) = ptngc_comp_make_dict_hist(input);
        let nvals = input.len();
        let mut encoded = vec![0u32; nvals];
        ptngc_comp_conv_to_mtf(input, &dict, &mut encoded);
        let decoded = c_from_mtf(&encoded, &dict);
        assert_eq!(
            decoded.as_slice(),
            input,
            "dict cross roundtrip (Rust enc → C dec) failed for input={input:?}"
        );
    }
}

#[test]
fn test_mtf_cross_roundtrip_c_encode_rust_decode() {
    let test_cases: &[&[u32]] = &[&[3, 1, 3, 7, 1], &[0, 1, 2, 0, 1], &[99, 99, 99]];
    for &input in test_cases {
        let (dict, _) = ptngc_comp_make_dict_hist(input);
        let nvals = input.len();
        let encoded = c_to_mtf(input, &dict);
        let mut decoded = vec![0u32; nvals];
        ptngc_comp_conv_from_mtf(&encoded, &dict, &mut decoded);
        assert_eq!(
            decoded.as_slice(),
            input,
            "dict cross roundtrip (C enc → Rust dec) failed for input={input:?}"
        );
    }
}
