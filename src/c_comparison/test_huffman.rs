use crate::dict::{DICT_SIZE, ptngc_comp_make_dict_hist};
use crate::ffi;
use crate::huffman::ptngc_comp_conv_to_huffman;
use std::ffi::c_int;
use std::sync::Once;

static INIT: Once = Once::new();
const HUFF_DICT_BUF_CAP: usize = DICT_SIZE + 3;

fn init_logger() {
    INIT.call_once(|| {
        env_logger::builder().is_test(true).try_init().ok();
    });
}

#[test]
fn single_symbol() {
    init_logger();
    let vals = vec![42, 42, 42, 42];
    let (dict, mut hist) = ptngc_comp_make_dict_hist(&vals);
    let mut rust_huffman = vec![0; 1000];
    let mut rust_huffman_len = 0;
    let mut rust_huffman_dictlen = 0;
    let mut rust_huffman_dict_unpackedlen = 0;
    let mut rust_huffman_dict = vec![0; HUFF_DICT_BUF_CAP];
    let mut rust_huffman_dict_unpacked = vec![0; HUFF_DICT_BUF_CAP];
    ptngc_comp_conv_to_huffman(
        &vals,
        &dict,
        &mut hist,
        &mut rust_huffman,
        &mut rust_huffman_len,
        &mut rust_huffman_dict,
        &mut rust_huffman_dictlen,
        &mut rust_huffman_dict_unpacked,
        &mut rust_huffman_dict_unpackedlen,
    );

    let mut c_hist = hist;
    let mut c_huffman = rust_huffman.clone();
    let mut c_huffman_len = rust_huffman_len as c_int;
    let mut c_huffman_dictlen = rust_huffman_dictlen as c_int;
    let mut c_huffman_dict_unpackedlen = rust_huffman_dict_unpackedlen as c_int;
    let mut c_huffman_dict = rust_huffman_dict.clone();
    let mut c_huffman_dict_unpacked = rust_huffman_dict_unpacked.clone();
    unsafe {
        ffi::Ptngc_comp_conv_to_huffman(
            vals.as_ptr(),
            vals.len() as c_int,
            dict.as_ptr(),
            dict.len() as c_int,
            c_hist.as_mut_ptr(),
            c_huffman.as_mut_ptr(),
            &mut c_huffman_len,
            c_huffman_dict.as_mut_ptr(),
            &mut c_huffman_dictlen,
            c_huffman_dict_unpacked.as_mut_ptr(),
            &mut c_huffman_dict_unpackedlen,
        );
    }

    assert_eq!(rust_huffman_len, c_huffman_len as usize);
    assert_eq!(rust_huffman_dictlen, c_huffman_dictlen as usize);
    assert_eq!(
        rust_huffman_dict_unpackedlen,
        c_huffman_dict_unpackedlen as usize
    );
    assert_eq!(
        rust_huffman[..rust_huffman_len],
        c_huffman[..c_huffman_len as usize]
    );
    assert_eq!(
        rust_huffman_dict[..rust_huffman_dictlen],
        c_huffman_dict[..c_huffman_dictlen as usize]
    );
    assert_eq!(
        rust_huffman_dict_unpacked[..rust_huffman_dict_unpackedlen],
        c_huffman_dict_unpacked[..c_huffman_dict_unpackedlen as usize]
    );
}

#[test]
fn two_symbols() {
    init_logger();

    let vals = vec![10, 20, 10, 20, 10];
    let (dict, mut hist) = ptngc_comp_make_dict_hist(&vals);
    let mut rust_huffman = vec![0; 1000];
    let mut rust_huffman_len = 0;
    let mut rust_huffman_dictlen = 0;
    let mut rust_huffman_dict_unpackedlen = 0;
    let mut rust_huffman_dict = vec![0; HUFF_DICT_BUF_CAP];
    let mut rust_huffman_dict_unpacked = vec![0; HUFF_DICT_BUF_CAP];
    ptngc_comp_conv_to_huffman(
        &vals,
        &dict,
        &mut hist,
        &mut rust_huffman,
        &mut rust_huffman_len,
        &mut rust_huffman_dict,
        &mut rust_huffman_dictlen,
        &mut rust_huffman_dict_unpacked,
        &mut rust_huffman_dict_unpackedlen,
    );

    let mut c_hist = hist;
    let mut c_huffman = rust_huffman.clone();
    let mut c_huffman_len = rust_huffman_len as c_int;
    let mut c_huffman_dictlen = rust_huffman_dictlen as c_int;
    let mut c_huffman_dict_unpackedlen = rust_huffman_dict_unpackedlen as c_int;
    let mut c_huffman_dict = rust_huffman_dict.clone();
    let mut c_huffman_dict_unpacked = rust_huffman_dict_unpacked.clone();
    unsafe {
        ffi::Ptngc_comp_conv_to_huffman(
            vals.as_ptr(),
            vals.len() as c_int,
            dict.as_ptr(),
            dict.len() as c_int,
            c_hist.as_mut_ptr(),
            c_huffman.as_mut_ptr(),
            &mut c_huffman_len,
            c_huffman_dict.as_mut_ptr(),
            &mut c_huffman_dictlen,
            c_huffman_dict_unpacked.as_mut_ptr(),
            &mut c_huffman_dict_unpackedlen,
        );
    }

    assert_eq!(rust_huffman_len, c_huffman_len as usize);
    assert_eq!(rust_huffman_dictlen, c_huffman_dictlen as usize);
    assert_eq!(
        rust_huffman_dict_unpackedlen,
        c_huffman_dict_unpackedlen as usize
    );
    assert_eq!(
        rust_huffman[..rust_huffman_len],
        c_huffman[..c_huffman_len as usize]
    );
    assert_eq!(
        rust_huffman_dict[..rust_huffman_dictlen],
        c_huffman_dict[..c_huffman_dictlen as usize]
    );
    assert_eq!(
        rust_huffman_dict_unpacked[..rust_huffman_dict_unpackedlen],
        c_huffman_dict_unpacked[..c_huffman_dict_unpackedlen as usize]
    );
}

#[test]
fn four_symbols() {
    init_logger();

    let vals = vec![1, 2, 3, 4, 1, 1, 2, 1];
    let (dict, mut hist) = ptngc_comp_make_dict_hist(&vals);
    let mut rust_huffman = vec![0; 1000];
    let mut rust_huffman_len = 0;
    let mut rust_huffman_dictlen = 0;
    let mut rust_huffman_dict_unpackedlen = 0;
    let mut rust_huffman_dict = vec![0; HUFF_DICT_BUF_CAP];
    let mut rust_huffman_dict_unpacked = vec![0; HUFF_DICT_BUF_CAP];
    ptngc_comp_conv_to_huffman(
        &vals,
        &dict,
        &mut hist,
        &mut rust_huffman,
        &mut rust_huffman_len,
        &mut rust_huffman_dict,
        &mut rust_huffman_dictlen,
        &mut rust_huffman_dict_unpacked,
        &mut rust_huffman_dict_unpackedlen,
    );

    let mut c_hist = hist;
    let mut c_huffman = rust_huffman.clone();
    let mut c_huffman_len = rust_huffman_len as c_int;
    let mut c_huffman_dictlen = rust_huffman_dictlen as c_int;
    let mut c_huffman_dict_unpackedlen = rust_huffman_dict_unpackedlen as c_int;
    let mut c_huffman_dict = rust_huffman_dict.clone();
    let mut c_huffman_dict_unpacked = rust_huffman_dict_unpacked.clone();
    unsafe {
        ffi::Ptngc_comp_conv_to_huffman(
            vals.as_ptr(),
            vals.len() as c_int,
            dict.as_ptr(),
            dict.len() as c_int,
            c_hist.as_mut_ptr(),
            c_huffman.as_mut_ptr(),
            &mut c_huffman_len,
            c_huffman_dict.as_mut_ptr(),
            &mut c_huffman_dictlen,
            c_huffman_dict_unpacked.as_mut_ptr(),
            &mut c_huffman_dict_unpackedlen,
        );
    }

    assert_eq!(rust_huffman_len, c_huffman_len as usize);
    assert_eq!(rust_huffman_dictlen, c_huffman_dictlen as usize);
    assert_eq!(
        rust_huffman_dict_unpackedlen,
        c_huffman_dict_unpackedlen as usize
    );
    assert_eq!(
        rust_huffman[..rust_huffman_len],
        c_huffman[..c_huffman_len as usize]
    );
    assert_eq!(
        rust_huffman_dict[..rust_huffman_dictlen],
        c_huffman_dict[..c_huffman_dictlen as usize]
    );
    assert_eq!(
        rust_huffman_dict_unpacked[..rust_huffman_dict_unpackedlen],
        c_huffman_dict_unpacked[..c_huffman_dict_unpackedlen as usize]
    );
}

#[test]
fn large_values() {
    init_logger();

    let vals = vec![1000, 1001, 1000, 1001, 1000];
    let (dict, mut hist) = ptngc_comp_make_dict_hist(&vals);
    let mut rust_huffman = vec![0; 1000];
    let mut rust_huffman_len = 0;
    let mut rust_huffman_dictlen = 0;
    let mut rust_huffman_dict_unpackedlen = 0;
    let mut rust_huffman_dict = vec![0; HUFF_DICT_BUF_CAP];
    let mut rust_huffman_dict_unpacked = vec![0; HUFF_DICT_BUF_CAP];
    ptngc_comp_conv_to_huffman(
        &vals,
        &dict,
        &mut hist,
        &mut rust_huffman,
        &mut rust_huffman_len,
        &mut rust_huffman_dict,
        &mut rust_huffman_dictlen,
        &mut rust_huffman_dict_unpacked,
        &mut rust_huffman_dict_unpackedlen,
    );

    let mut c_hist = hist;
    let mut c_huffman = rust_huffman.clone();
    let mut c_huffman_len = rust_huffman_len as c_int;
    let mut c_huffman_dictlen = rust_huffman_dictlen as c_int;
    let mut c_huffman_dict_unpackedlen = rust_huffman_dict_unpackedlen as c_int;
    let mut c_huffman_dict = rust_huffman_dict.clone();
    let mut c_huffman_dict_unpacked = rust_huffman_dict_unpacked.clone();
    unsafe {
        ffi::Ptngc_comp_conv_to_huffman(
            vals.as_ptr(),
            vals.len() as c_int,
            dict.as_ptr(),
            dict.len() as c_int,
            c_hist.as_mut_ptr(),
            c_huffman.as_mut_ptr(),
            &mut c_huffman_len,
            c_huffman_dict.as_mut_ptr(),
            &mut c_huffman_dictlen,
            c_huffman_dict_unpacked.as_mut_ptr(),
            &mut c_huffman_dict_unpackedlen,
        );
    }

    assert_eq!(rust_huffman_len, c_huffman_len as usize);
    assert_eq!(rust_huffman_dictlen, c_huffman_dictlen as usize);
    assert_eq!(
        rust_huffman_dict_unpackedlen,
        c_huffman_dict_unpackedlen as usize
    );
    assert_eq!(
        rust_huffman[..rust_huffman_len],
        c_huffman[..c_huffman_len as usize]
    );
    assert_eq!(
        rust_huffman_dict[..rust_huffman_dictlen],
        c_huffman_dict[..c_huffman_dictlen as usize]
    );
    assert_eq!(
        rust_huffman_dict_unpacked[..rust_huffman_dict_unpackedlen],
        c_huffman_dict_unpacked[..c_huffman_dict_unpackedlen as usize]
    );
}

#[test]
fn empty() {
    init_logger();

    // The C function segfaults on empty input (ndict=0 causes it to dereference
    // a zero-byte allocation and access codelength[-1]). Rust handles this correctly
    // by returning early. We just verify that the Rust side produces all-zero outputs.
    let vals: Vec<u32> = vec![];
    let (dict, mut hist) = ptngc_comp_make_dict_hist(&vals);
    let mut rust_huffman = vec![0; 1000];
    let mut rust_huffman_len = 0;
    let mut rust_huffman_dictlen = 0;
    let mut rust_huffman_dict_unpackedlen = 0;
    let mut rust_huffman_dict = vec![0; HUFF_DICT_BUF_CAP];
    let mut rust_huffman_dict_unpacked = vec![0; HUFF_DICT_BUF_CAP];
    ptngc_comp_conv_to_huffman(
        &vals,
        &dict,
        &mut hist,
        &mut rust_huffman,
        &mut rust_huffman_len,
        &mut rust_huffman_dict,
        &mut rust_huffman_dictlen,
        &mut rust_huffman_dict_unpacked,
        &mut rust_huffman_dict_unpackedlen,
    );

    assert_eq!(rust_huffman_len, 0);
    assert_eq!(rust_huffman_dictlen, 0);
    assert_eq!(rust_huffman_dict_unpackedlen, 0);
}
