use crate::{
    coder::Coder,
    compress::{
        TNG_COMPRESS_ALGO_BWLZH1, TNG_COMPRESS_ALGO_BWLZH2, TNG_COMPRESS_ALGO_POS_TRIPLET_INTRA,
        TNG_COMPRESS_ALGO_POS_TRIPLET_ONETOONE, TNG_COMPRESS_ALGO_POS_XTC2,
        TNG_COMPRESS_ALGO_POS_XTC3, TNG_COMPRESS_ALGO_STOPBIT, TNG_COMPRESS_ALGO_TRIPLET,
        TNG_COMPRESS_ALGO_VEL_STOPBIT_INTER,
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
    coder
        .pack_array(&mut input, &mut length, algo, 0, natoms, &mut speed)
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

fn c_run_unpack(
    packed: &[u8],
    length: i32,
    coding: i32,
    coding_parameter: i32,
    natoms: i32,
) -> Vec<i32> {
    let c_coder = unsafe { ffi::Ptngc_coder_init() };
    let mut packed_copy = packed.to_vec();
    let mut output = vec![0i32; length as usize];
    let ret = unsafe {
        ffi::Ptngc_unpack_array(
            c_coder,
            packed_copy.as_mut_ptr(),
            output.as_mut_ptr(),
            length,
            coding,
            coding_parameter,
            natoms,
        )
    };
    unsafe { ffi::Ptngc_coder_deinit(c_coder) };
    assert_eq!(ret, 0, "Ptngc_unpack_array returned {ret}");
    output
}

fn rust_run_unpack(packed: &[u8], length: i32, coding: i32, coding_parameter: i32) -> Vec<i32> {
    let coder = Coder::default();
    let mut output = vec![0i32; length as usize];
    coder.unpack_array(
        packed,
        &mut output,
        length,
        coding,
        coding_parameter,
        (length / 3) as usize,
    );
    output
}

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

fn unpack_vals() -> impl Strategy<Value = Vec<i32>> {
    prop_oneof![
        prop::collection::vec(-1_073_741i32..=1_073_741i32, 6),
        prop::collection::vec(-1_073_741i32..=1_073_741i32, 30),
        prop::collection::vec(-1_073_741i32..=1_073_741i32, 99),
    ]
}

fn xtc3_vals() -> impl Strategy<Value = Vec<i32>> {
    prop_oneof![
        prop::collection::vec(-1_000i32..=1_000i32, 60),
        prop::collection::vec(-1_000i32..=1_000i32, 120),
    ]
}

// ---------------------------------------------------------------------------
// Guards
// ---------------------------------------------------------------------------

fn triplet_guard(vals: &[i32]) -> bool {
    vals.iter().copied().map(positive_int).max().unwrap_or(0) < (1u32 << 31)
}

fn xtc2_guard(vals: &[i32]) -> bool {
    const MAGIC_LAST_ENTRY: u64 = 3408917801; // MAGIC[91], the last entry
    for dim in 0..3 {
        let dim_vals: Vec<i32> = vals.iter().copied().skip(dim).step_by(3).collect();
        let maxint = *dim_vals.iter().max().unwrap() as i64;
        let minint = *dim_vals.iter().min().unwrap() as i64;
        if ((maxint - minint + 1) as u64) >= MAGIC_LAST_ENTRY {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Assertion helpers
// ---------------------------------------------------------------------------

/// Pack `vals` with C, unpack with both C and Rust, assert they match each other and the original.
fn assert_unpack_matches_c(vals: &[i32], algo: i32, speed: i32) -> Result<(), TestCaseError> {
    let natoms = (vals.len() / 3) as i32;
    let len = vals.len() as i32;
    let (c_packed, c_packed_len) = c_run_pack(&mut vals.to_vec(), algo, natoms, speed);
    let packed = &c_packed[..c_packed_len as usize];
    let c_unpacked = c_run_unpack(packed, len, algo, 0, natoms);
    let rust_unpacked = rust_run_unpack(packed, len, algo, 0);
    prop_assert_eq!(&rust_unpacked, &c_unpacked);
    prop_assert_eq!(&c_unpacked, vals);
    Ok(())
}

/// Pack with Rust -> unpack with C, and pack with C -> unpack with Rust. Assert roundtrip.
fn assert_pack_unpack_cross(vals: &[i32], algo: i32, speed: i32) -> Result<(), TestCaseError> {
    let natoms = (vals.len() / 3) as i32;
    let len = vals.len() as i32;
    let (rust_packed, rust_packed_len) = rust_run_pack(vals, algo, vals.len() / 3, speed as usize);
    let c_of_rust = c_run_unpack(&rust_packed[..rust_packed_len], len, algo, 0, natoms);
    prop_assert_eq!(&c_of_rust, vals);
    let (c_packed, c_packed_len) = c_run_pack(&mut vals.to_vec(), algo, natoms, speed);
    let rust_of_c = rust_run_unpack(&c_packed[..c_packed_len as usize], len, algo, 0);
    prop_assert_eq!(&rust_of_c, vals);
    Ok(())
}

proptest! {
    #[test]
    fn pack_array_matches_c(
        speed in 1usize..6,
        algo in prop_oneof![
            Just(TNG_COMPRESS_ALGO_BWLZH1),
            Just(TNG_COMPRESS_ALGO_BWLZH2),
            Just(TNG_COMPRESS_ALGO_POS_XTC2),
            Just(TNG_COMPRESS_ALGO_POS_XTC3),
            Just(TNG_COMPRESS_ALGO_TRIPLET),
        ],
        // Range restricted so positive_int(v) < 2^31 for triplet encoder safety.
        // Array sizes must be divisible by 3 (representing 3D atom coordinates).
        vals in prop_oneof![
            prop::collection::vec(-1_073_741i32..=1_073_741i32, 30),
            prop::collection::vec(-1_073_741i32..=1_073_741i32, 99),
        ]
    ) {
        // XTC3 base_compress has a heap buffer overflow in C with large values — tested separately.
        if algo == TNG_COMPRESS_ALGO_POS_XTC3 {
            prop_assume!(false);
        }
        if algo == TNG_COMPRESS_ALGO_POS_XTC2 {
            prop_assume!(xtc2_guard(&vals));
        }
        // Triplet encoder uses u32 max_base that doubles past 2^31, wrapping to 0 -> infinite loop
        // in C (and overflow panic in Rust debug). Skip inputs where positive_int(v) >= 2^31.
        if algo == TNG_COMPRESS_ALGO_TRIPLET
            || algo == TNG_COMPRESS_ALGO_POS_TRIPLET_INTRA
            || algo == TNG_COMPRESS_ALGO_POS_TRIPLET_ONETOONE
        {
            prop_assume!(triplet_guard(&vals));
        }
        let (c_output, _c_length) = c_run_pack(&mut vals.clone(), algo, (vals.len() / 3) as c_int, speed as c_int);
        let (rust_output, rust_output_length) = rust_run_pack(&vals, algo, vals.len() / 3, speed);
        prop_assert_eq!(&rust_output[..rust_output_length], c_output);
    }
}

// Separate test for XTC3 with smaller values to avoid C base_compress buffer overflow
proptest! {
    #[test]
    fn pack_array_xtc3_matches_c(
        speed in 1usize..6,
        vals in xtc3_vals()
    ) {
        let algo = TNG_COMPRESS_ALGO_POS_XTC3;
        let (c_output, _c_length) = c_run_pack(&mut vals.clone(), algo, (vals.len() / 3) as c_int, speed as c_int);
        let (rust_output, rust_output_length) = rust_run_pack(&vals, algo, vals.len() / 3, speed);
        prop_assert_eq!(&rust_output[..rust_output_length], c_output);
    }
}

proptest! {
    #[test]
    fn unpack_stopbit_matches_c(vals in unpack_vals()) {
        assert_unpack_matches_c(&vals, TNG_COMPRESS_ALGO_STOPBIT, 1)?;
    }
}

proptest! {
    #[test]
    fn stopbit_pack_unpack_cross(vals in unpack_vals()) {
        assert_pack_unpack_cross(&vals, TNG_COMPRESS_ALGO_STOPBIT, 1)?;
    }
}

// VEL_STOPBIT_INTER uses the same unpack path as STOPBIT but with a different coding constant.
proptest! {
    #[test]
    fn unpack_vel_stopbit_inter_matches_c(vals in unpack_vals()) {
        assert_unpack_matches_c(&vals, TNG_COMPRESS_ALGO_VEL_STOPBIT_INTER, 1)?;
    }
}

proptest! {
    #[test]
    fn unpack_triplet_matches_c(vals in unpack_vals()) {
        prop_assume!(triplet_guard(&vals));
        assert_unpack_matches_c(&vals, TNG_COMPRESS_ALGO_TRIPLET, 1)?;
    }
}

proptest! {
    #[test]
    fn triplet_pack_unpack_cross(vals in unpack_vals()) {
        prop_assume!(triplet_guard(&vals));
        assert_pack_unpack_cross(&vals, TNG_COMPRESS_ALGO_TRIPLET, 1)?;
    }
}

proptest! {
    #[test]
    fn unpack_triplet_intra_matches_c(vals in unpack_vals()) {
        prop_assume!(triplet_guard(&vals));
        assert_unpack_matches_c(&vals, TNG_COMPRESS_ALGO_POS_TRIPLET_INTRA, 1)?;
    }
}

proptest! {
    #[test]
    fn unpack_triplet_onetoone_matches_c(vals in unpack_vals()) {
        prop_assume!(triplet_guard(&vals));
        assert_unpack_matches_c(&vals, TNG_COMPRESS_ALGO_POS_TRIPLET_ONETOONE, 1)?;
    }
}

proptest! {
    #[test]
    fn unpack_xtc2_matches_c(vals in unpack_vals()) {
        prop_assume!(xtc2_guard(&vals));
        assert_unpack_matches_c(&vals, TNG_COMPRESS_ALGO_POS_XTC2, 0)?;
    }
}

proptest! {
    #[test]
    fn xtc2_pack_unpack_cross(vals in unpack_vals()) {
        prop_assume!(xtc2_guard(&vals));
        assert_pack_unpack_cross(&vals, TNG_COMPRESS_ALGO_POS_XTC2, 1)?;
    }
}

proptest! {
    #[test]
    fn unpack_xtc3_matches_c(vals in xtc3_vals()) {
        assert_unpack_matches_c(&vals, TNG_COMPRESS_ALGO_POS_XTC3, 1)?;
    }
}

proptest! {
    #[test]
    fn xtc3_pack_unpack_cross(vals in xtc3_vals()) {
        assert_pack_unpack_cross(&vals, TNG_COMPRESS_ALGO_POS_XTC3, 1)?;
    }
}

proptest! {
    #[test]
    fn unpack_bwlzh1_matches_c(vals in unpack_vals()) {
        assert_unpack_matches_c(&vals, TNG_COMPRESS_ALGO_BWLZH1, 9)?;
    }
}

proptest! {
    #[test]
    fn bwlzh1_pack_unpack_cross(vals in unpack_vals()) {
        assert_pack_unpack_cross(&vals, TNG_COMPRESS_ALGO_BWLZH1, 9)?;
    }
}

proptest! {
    #[test]
    fn unpack_bwlzh2_matches_c(vals in unpack_vals()) {
        assert_unpack_matches_c(&vals, TNG_COMPRESS_ALGO_BWLZH2, 9)?;
    }
}

proptest! {
    #[test]
    fn bwlzh2_pack_unpack_cross(vals in unpack_vals()) {
        assert_pack_unpack_cross(&vals, TNG_COMPRESS_ALGO_BWLZH2, 9)?;
    }
}
