use crate::{
    bwlzh::ptngc_comp_to_bwt,
    coder::Coder,
    compress::{
        TNG_COMPRESS_ALGO_BWLZH1, TNG_COMPRESS_ALGO_BWLZH2, TNG_COMPRESS_ALGO_POS_TRIPLET_INTRA,
        TNG_COMPRESS_ALGO_POS_TRIPLET_ONETOONE, TNG_COMPRESS_ALGO_POS_XTC2,
        TNG_COMPRESS_ALGO_POS_XTC3, TNG_COMPRESS_ALGO_TRIPLET,
    },
    ffi,
    xtc3::positive_int,
};
use proptest::prelude::*;
use std::ffi::c_int;

fn rust_run_pack(input: &[i32], algo: i32, natoms: usize, speed: usize) -> (Vec<u8>, usize) {
    let mut coder = Coder::default();
    let mut input = input.to_vec();
    let mut length = input.len();
    let mut speed = speed;
    let coding_param = 0;
    coder
        .pack_array(
            &mut input,
            &mut length,
            algo,
            coding_param,
            natoms,
            &mut speed,
        )
        .unwrap()
}

fn c_run_pack(input: &mut [i32], algo: c_int, natoms: i32, speed: c_int) -> (Vec<u8>, i32) {
    let c_coder = unsafe { ffi::Ptngc_coder_init() };
    let mut c_length = input.len() as c_int;
    let c_raw_output = unsafe {
        ffi::Ptngc_pack_array(
            c_coder,
            input.as_mut_ptr(),
            &mut c_length,
            algo,
            0 as c_int,
            natoms,
            speed,
        )
    };
    assert!(!c_raw_output.is_null(), "Ptngc_pack_array returned null");
    let mut c_output = vec![0; c_length as usize];
    unsafe { c_raw_output.copy_to(c_output.as_mut_ptr(), c_length as usize) };
    unsafe { libc::free(c_raw_output as *mut _) };
    unsafe { ffi::Ptngc_coder_deinit(c_coder) };

    (c_output, c_length)
}

fn arbitrary_algo() -> impl Strategy<Value = i32> {
    prop_oneof![
        // Just(TNG_COMPRESS_ALGO_BWLZH1),
        // Just(TNG_COMPRESS_ALGO_BWLZH2),
        // Just(TNG_COMPRESS_ALGO_POS_XTC2),
        // Just(TNG_COMPRESS_ALGO_POS_XTC3),
        Just(TNG_COMPRESS_ALGO_TRIPLET),
    ]
}

proptest! {
    #[test]
    fn pack_array_matches_c(
        speed in 1usize..6,
        algo in arbitrary_algo(),
        // we want the length of the array to match the dimension of atoms.
        // Range restricted so positive_int(v) < 2^31 for triplet encoder safety.
        // otherwise, an infinite loop gets triggered in the C code and the Rust code
        // overflows.
        vals in prop_oneof![
            prop::collection::vec(-1_073_741_823i32..=1_073_741_823i32, 3),
            prop::collection::vec(-1_073_741_823i32..=1_073_741_823i32, 6),
        ]
    ) {
        if algo == TNG_COMPRESS_ALGO_POS_XTC2 {
            const MAGIC_LAST_ENTRY: u64 = 3408917801; // MAGIC[91], the last entry
            for dim in 0..3 {
                let dim_vals: Vec<i32> = vals.iter().copied().skip(dim).step_by(3).collect();
                let maxint = *dim_vals.iter().max().unwrap() as i64;
                let minint = *dim_vals.iter().min().unwrap() as i64;
                prop_assume!(((maxint - minint + 1) as u64) < MAGIC_LAST_ENTRY);
            }
        }
        // Triplet encoder uses u32 max_base that doubles past 2^31, wrapping to 0 -> infinite loop
        // in C (and overflow panic in Rust debug). Skip inputs where positive_int(v) >= 2^31.
        if algo == TNG_COMPRESS_ALGO_TRIPLET
            || algo == TNG_COMPRESS_ALGO_POS_TRIPLET_INTRA
            || algo == TNG_COMPRESS_ALGO_POS_TRIPLET_ONETOONE
        {
            let intmax = vals.iter().copied().map(positive_int).max().unwrap_or(0);
            prop_assume!(intmax < (1u32 << 31));
        }
        let (c_output, _c_length) = c_run_pack(&mut vals.clone(), algo, (vals.len() / 3) as c_int, speed as c_int);
        let (rust_output, rust_output_length) = rust_run_pack(&vals, algo, vals.len() / 3, speed);

        prop_assert_eq!(&rust_output[..rust_output_length], c_output);
    }
}
