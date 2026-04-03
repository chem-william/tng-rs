use crate::{
    bwlzh::{ptngc_comp_from_bwt, ptngc_comp_to_bwt},
    ffi,
};
use proptest::prelude::*;
use std::ffi::c_int;

proptest! {
    #[test]
    fn prop_bwt_matches_c(vals in prop::collection::vec(0u32..=255, 0..=256)) {
        let mut rust_output = vec![0u32; vals.len()];
        let rust_index = ptngc_comp_to_bwt(&vals, vals.len(), &mut rust_output);

        let mut c_input = vals.clone();
        let mut c_output = vec![0u32; vals.len()];
        let mut c_index: c_int = 0;
        unsafe {
            ffi::Ptngc_comp_to_bwt(
                c_input.as_mut_ptr(),
                c_input.len() as c_int,
                c_output.as_mut_ptr(),
                &raw mut c_index,
            );
        }

        prop_assert_eq!(rust_index, c_index as usize);
        prop_assert_eq!(rust_output, c_output);
    }

    #[test]
    fn prop_bwt_roundtrip(vals in prop::collection::vec(0u32..=255, 1..=256)) {
        let mut transformed = vec![0u32; vals.len()];
        let index = ptngc_comp_to_bwt(&vals, vals.len(), &mut transformed);
        let mut recovered = vec![0u32; vals.len()];
        ptngc_comp_from_bwt(&transformed, index, &mut recovered);
        prop_assert_eq!(recovered, vals);
    }
}
