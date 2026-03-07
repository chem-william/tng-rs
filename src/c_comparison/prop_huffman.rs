use crate::{
    dict::{DICT_SIZE, ptngc_comp_make_dict_hist},
    ffi,
    huffman::ptngc_comp_conv_to_huffman,
};
use proptest::prelude::*;
use std::ffi::c_int;

const HUFF_DICT_BUF_CAP: usize = crate::dict::DICT_SIZE + 3;
const MAX_DICT_SYMBOL: u32 = (crate::dict::DICT_SIZE - 1) as u32;

fn rust_encode(vals: &[u32]) -> (Vec<u8>, usize, Vec<u8>, usize, Vec<u32>, usize) {
    let (dict, mut hist) = ptngc_comp_make_dict_hist(vals);

    let mut huffman = vec![0u8; vals.len() * 4 + 16];
    let mut huffman_len = 0;
    let mut huffman_dict = vec![0; HUFF_DICT_BUF_CAP];
    let mut huffman_dict_len = 0;
    let mut huffman_dict_unpacked = vec![0; HUFF_DICT_BUF_CAP];
    let mut huffman_dict_unpacked_len = 0usize;

    ptngc_comp_conv_to_huffman(
        vals,
        &dict,
        &mut hist,
        &mut huffman,
        &mut huffman_len,
        &mut huffman_dict,
        &mut huffman_dict_len,
        &mut huffman_dict_unpacked,
        &mut huffman_dict_unpacked_len,
    );

    (
        huffman,
        huffman_len,
        huffman_dict,
        huffman_dict_len,
        huffman_dict_unpacked,
        huffman_dict_unpacked_len,
    )
}

fn c_encode(vals: &[u32]) -> (Vec<u8>, usize, Vec<u8>, usize, Vec<u32>, usize) {
    let mut dict = vec![0; DICT_SIZE];
    let mut hist = vec![0; DICT_SIZE];
    let mut ndict = 0;
    unsafe {
        ffi::Ptngc_comp_make_dict_hist(
            vals.as_ptr(),
            vals.len() as c_int,
            dict.as_mut_ptr(),
            &mut ndict,
            hist.as_mut_ptr(),
        );
    }

    let mut huffman = vec![0u8; vals.len() * 4 + 16];
    let mut huffman_len = 0 as c_int;
    let mut huffman_dict = vec![0; HUFF_DICT_BUF_CAP];
    let mut huffman_dict_len = 0 as c_int;
    let mut huffman_dict_unpacked = vec![0; HUFF_DICT_BUF_CAP];
    let mut huffman_dict_unpacked_len = 0 as c_int;

    unsafe {
        ffi::Ptngc_comp_conv_to_huffman(
            vals.as_ptr(),
            vals.len() as c_int,
            dict.as_ptr(),
            ndict,
            hist.as_mut_ptr(),
            huffman.as_mut_ptr(),
            &mut huffman_len,
            huffman_dict.as_mut_ptr(),
            &mut huffman_dict_len,
            huffman_dict_unpacked.as_mut_ptr(),
            &mut huffman_dict_unpacked_len,
        );
    }

    (
        huffman,
        huffman_len as usize,
        huffman_dict,
        huffman_dict_len as usize,
        huffman_dict_unpacked,
        huffman_dict_unpacked_len as usize,
    )
}

proptest! {
    #[test]
    fn prop_encode_matches_c(vals in prop::collection::vec(0..DICT_SIZE as u32, 1..=32)) {
        // The C implementation segfaults for empty input, so differential cases are non-empty.
        let (
            c_huffman,
            c_huffman_len,
            c_huffman_dict,
            c_huffman_dict_len,
            c_huffman_dict_unpacked,
            c_huffman_dict_unpacked_len,
        ) = c_encode(&vals);

        let (
            rust_huffman,
            rust_huffman_len,
            rust_huffman_dict,
            rust_huffman_dict_len,
            rust_huffman_dict_unpacked,
            rust_huffman_dict_unpacked_len,
        ) = rust_encode(&vals);


        prop_assert_eq!(rust_huffman_len, c_huffman_len, "huffman len mismatch");
        prop_assert_eq!(
            rust_huffman_dict_len,
            c_huffman_dict_len,
            "huffman dict len mismatch"
        );
        prop_assert_eq!(
            rust_huffman_dict_unpacked_len,
            c_huffman_dict_unpacked_len,
            "huffman dict unpacked len mismatch"
        );

        prop_assert_eq!(
            &rust_huffman[..rust_huffman_len],
            &c_huffman[..c_huffman_len],
            "huffman payload mismatch"
        );
        prop_assert_eq!(
            &rust_huffman_dict[..rust_huffman_dict_len],
            &c_huffman_dict[..c_huffman_dict_len],
            "huffman dict payload mismatch"
        );
        prop_assert_eq!(
            &rust_huffman_dict_unpacked[..rust_huffman_dict_unpacked_len],
            &c_huffman_dict_unpacked[..c_huffman_dict_unpacked_len],
            "huffman unpacked dict payload mismatch"
        );
    }
}
